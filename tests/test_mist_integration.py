from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from test_mist_checkpoint import release as release
from test_mist_checkpoint import spectrum_batch

from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.mist.models.scaled import ScaledMIST


def make_model(checkpoint=None, config=None, load_pretrained_encoder=True):
    root = Path(__file__).resolve().parents[1]
    if config is None:
        config = OmegaConf.create({
            'general': OmegaConf.load(root / 'configs/general/general_default.yaml'),
            'model': OmegaConf.load(root / 'configs/model/model_default.yaml'),
            'dataset': OmegaConf.load(root / 'configs/dataset/canopus.yaml'),
        })
        config.general.encoder = str(checkpoint) if checkpoint else None
        config.general.val_samples_to_generate = 2
        config.general.test_samples_to_generate = 2
        config.model.n_layers = 1
        config.model.encoder_hidden_dim = 32
        config.model.hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 32}
        config.model.hidden_dims = {
            'dx': 32, 'de': 16, 'dy': 32, 'n_head': 4,
            'dim_ffX': 32, 'dim_ffE': 16, 'dim_ffy': 32,
        }
    info = SimpleNamespace(
        input_dims={'X': 8, 'E': 5, 'y': 2048},
        output_dims={'X': 8, 'E': 5, 'y': 0},
        nodes_dist=None, node_types=torch.ones(8), edge_types=torch.ones(5),
    )
    return Spec2MolDenoisingDiffusion(
        config, info, None, None, None, None,
        load_pretrained_encoder=load_pretrained_encoder,
    ).eval()


def test_combined_checkpoint_restores_without_external_mist_file(release):
    path, reference, _ = release
    model = make_model(path)
    assert isinstance(model.encoder, ScaledMIST)
    assert model.cfg.model.mist_model_config.hidden_size == 32
    assert model.cfg.general.encoder is None
    path.unlink()
    restored = make_model(config=deepcopy(model.cfg), load_pretrained_encoder=False)
    checkpoint = {'state_dict': model.state_dict()}
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    with torch.inference_mode():
        expected, _ = reference(spectrum_batch())
        actual, _ = restored.encoder(spectrum_batch())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_explicit_mist_overrides_bundled_encoder_and_preserves_decoder(release):
    path, reference, _ = release
    legacy = make_model()
    model = make_model(path)
    checkpoint = {'state_dict': legacy.state_dict()}
    decoder_weights = {name: value.clone() for name, value in legacy.decoder.state_dict().items()}
    model.on_load_checkpoint(checkpoint)
    assert not any(name.startswith('encoder.spectra_encoder.') for name in checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    for name, value in model.decoder.state_dict().items():
        torch.testing.assert_close(value, decoder_weights[name], rtol=0, atol=0)
    with torch.inference_mode():
        expected, _ = reference(spectrum_batch())
        actual, _ = model.encoder(spectrum_batch())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_restore_ignores_stale_pretrained_encoder_path():
    initial = make_model()
    config = deepcopy(initial.cfg)
    config.general.encoder = '/does/not/exist/mist.ckpt'
    restored = make_model(config=config, load_pretrained_encoder=False)
    restored.load_state_dict(initial.state_dict(), strict=True)
