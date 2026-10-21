import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mist.data.data import Spectra
from src.mist.data.featurizers import PeakFormula
from src.mist.models.checkpoint import (
    configure_encoder_finetuning,
    load_mist_checkpoint,
)
from src.mist.models.scaled import ScaledMIST

CONFIG = dict(hidden_size=32, peak_attn_layers=2, num_heads=4,
              refine_layers=3, spectra_dropout=0.0, no_diffs=True)


def spectrum_batch():
    forms = torch.zeros(2, 5, 18)
    forms[:, :, 0] = 1
    forms[:, :, 1] = 2
    forms[0, 2, [0, 1, 13]] = torch.tensor([2., 6., 1.])
    forms[1, 3, [0, 1, 13]] = torch.tensor([2., 6., 1.])
    return dict(form_vec=forms, ion_vec=torch.zeros(2, 5, dtype=torch.long),
                intens=torch.tensor([[0.5, 0.7, 1., 0., 0.], [0.2, 0.8, 0.5, 1., 0.]]),
                types=torch.tensor([[0, 0, 3, 0, 0], [0, 0, 0, 3, 0]]),
                num_peaks=torch.tensor([3, 4]), instruments=torch.tensor([0, 0]))


@pytest.fixture
def release(tmp_path):
    torch.manual_seed(7)
    model = ScaledMIST(**CONFIG).eval()
    checkpoint = dict(
        checkpoint_kind="mist_ema_release", step=10,
        config={"model": CONFIG, "data": {"dataset_name": "canopus_all"}},
        ema_model=model.state_dict(),
        model={key: torch.zeros_like(value) for key, value in model.state_dict().items()},
    )
    path = tmp_path / "mist_canopus_aug.ckpt"
    torch.save(checkpoint, path)
    return path, model, checkpoint


def test_loads_checkpoint_architecture_and_ema(release):
    path, reference, _ = release
    model = load_mist_checkpoint(path)
    assert model.hidden_size == 32
    assert model.dataset_name == "canopus_all"
    assert not model.training
    batch = spectrum_batch()
    with torch.inference_mode():
        expected, _ = reference(batch)
        actual, aux = model(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 4096)
    assert aux['h0'].shape == (2, 32)
    assert [item.shape[-1] for item in aux['int_preds']] == [512, 1024, 2048]
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('corruption', ['missing', 'extra', 'shape'])
def test_incompatible_weights_fail_strictly(release, corruption):
    path, _, checkpoint = release
    if corruption == 'missing':
        checkpoint['ema_model'].pop('encoder.intermediate_layer.input_layer.weight')
    elif corruption == 'extra':
        checkpoint['ema_model']['not_a_parameter'] = torch.tensor(0.)
    else:
        checkpoint['config']['model'] = {**CONFIG, 'hidden_size': 64}
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError):
        load_mist_checkpoint(path)


def test_missing_checkpoint_and_wrong_format_fail(tmp_path):
    path = tmp_path / 'missing.ckpt'
    with pytest.raises(FileNotFoundError):
        load_mist_checkpoint(path)
    torch.save({'state_dict': {}}, path)
    with pytest.raises(ValueError, match='coleygroup/MIST'):
        load_mist_checkpoint(path)


def test_masked_padding_does_not_change_predictions(release):
    path, _, _ = release
    model = load_mist_checkpoint(path)
    batch = spectrum_batch()
    with torch.inference_mode():
        before, _ = model(batch)
        batch['form_vec'][0, 3:] = 0
        batch['form_vec'][1, 4:] = 0
        batch['intens'][0, 3:] = 0.9
        batch['intens'][1, 4:] = 0.9
        after, _ = model(batch)
    torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('count', [0, 1, 10, 12])
def test_peak_limit_counts_fragments_not_cls(tmp_path, count):
    tree = {'cand_form': 'C2H6O', 'cand_ion': '[M+H]+', 'output_tbl': None}
    if count:
        tree['output_tbl'] = {
            'formula': ['CH3O'] * count,
            'ms2_inten': np.linspace(0.1, 0.9, count).tolist(),
            'ions': ['[M+H]+'] * count,
        }
    (tmp_path / 'sample.json').write_text(json.dumps(tree))
    feature = PeakFormula(subform_folder=str(tmp_path), max_peaks=10).featurize(
        Spectra(spectra_name='sample', spectra_formula='C2H6O', instrument='Unknown'))
    assert len(feature['peak_type']) == min(count, 10) + 1
    assert feature['peak_type'][-1] == 3
    assert (feature['peak_type'] == 3).sum() == 1


@pytest.mark.parametrize('strategy,prefix', [('ft-unfold', 'decoder.'), ('ft-transformer', 'encoder.')])
def test_finetuning_selects_the_new_encoder_layers(strategy, prefix):
    model = ScaledMIST(**CONFIG)
    configure_encoder_finetuning(model, strategy)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(name.startswith(prefix) for name in trainable)


def test_predictions_match_coleygroup_mist(tmp_path):
    source = os.environ.get('MIST_SOURCE')
    if not source:
        pytest.skip('Set MIST_SOURCE to compare against coleygroup/MIST')
    inputs = tmp_path / 'inputs.pt'
    output = tmp_path / 'reference.pt'
    torch.save({'batch': spectrum_batch(), 'config': CONFIG}, inputs)
    code = """
import sys
import torch
from mist.models.scaled_mist import ScaledMIST
inputs = torch.load(sys.argv[1], weights_only=False)
torch.manual_seed(17)
model = ScaledMIST(**inputs['config']).eval()
with torch.inference_mode():
    predicted, aux = model(inputs['batch'])
    hidden = model.encoder(inputs['batch'])
torch.save({'config': {'model': inputs['config']}, 'ema_model': model.state_dict(),
            'predicted': predicted, 'hidden': hidden}, sys.argv[2])
"""
    env = {**os.environ, 'PYTHONPATH': str(Path(source) / 'src'), 'OMP_NUM_THREADS': '1'}
    subprocess.run([sys.executable, '-c', code, str(inputs), str(output)],
                   cwd=source, env=env, check=True, timeout=60)
    reference = torch.load(output, weights_only=False)
    model = load_mist_checkpoint(output)
    with torch.inference_mode():
        predicted, aux = model(spectrum_batch())
    torch.testing.assert_close(predicted, reference['predicted'], rtol=0, atol=0)
    torch.testing.assert_close(aux['h0'], reference['hidden'], rtol=0, atol=0)
