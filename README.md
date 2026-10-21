# DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra

![teaser](./figs/diffms-animation.gif)

This is the codebase for our preprint [DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra](https://arxiv.org/abs/2502.09571).

The DiffMS codebase is adapted from [DiGress](https://github.com/cvignac/DiGress). 

## Environment installation
This code was tested with PyTorch 2.3.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a conda environment with rdkit:
    
    ```
    conda create -y -c conda-forge -n diffms rdkit=2024.09.4 python=3.9
    conda activate diffms
    ```

  - OR for a faster installation, you can use mamba:

    ```
    mamba create -y -n diffms rdkit=2024.09.4 python=3.9
    mamba activate diffms
    ```
    
  - Install a corresponding version of pytorch, for example: 
    
    ```pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118```

  - Run:
    
    ```pip install -e .```


## Dataset Download/Processing

We provide a series of scripts to download/process the pretraining and finetuning datasets. To download/setup the datasets, run the scripts in the data_processing/ folder in order:

```
bash data_processing/00_download_fp2mol_data.sh
bash data_processing/01_download_canopus_data.sh
bash data_processing/02_download_msg_data.sh
bash data_processing/03_preprocess_fp2mol.sh
```

These scripts use unzip, which can be installed with ```sudo apt-get install unzip``` on Linux. If you are on a different OS, you many need to edit these scripts or run the command manually.  

## Run the code
  
For fingerprint-molecule pretraining run [fp2mol_main.py](src/fp2mol_main.py). You will need to set the dataset in [config.yaml](configs/config.yaml) to 'fp2mol'. The primary pretraining dataset in our paper is referred to as 'combined' in the [fp2mol.yaml](configs/dataset/fp2mol.yaml) config. 

To finetune the end-to-end model on spectra-molecule generation, run [spec2mol_main.py](src/spec2mol_main.py). You will also need to set the dataset in [config.yaml](configs/config.yaml) to 'msg' for MassSpecGym or 'canopus' for NPLIB1. 

## Pretrained Checkpoints

Pretrained DiffMS end-to-end models and decoders are available [here](https://zenodo.org/records/15122968). MIST encoder checkpoints are available [here](https://zenodo.org/records/23303289), with the complete training code in [coleygroup/MIST](https://github.com/coleygroup/MIST). The MIST checkpoints, `mist_{canopus,msg}_{noaug,aug}.ckpt`, cover CANOPUS and MassSpecGym, with (`aug`) or without (`noaug`) synthetic training spectra.

To use a pretrained end-to-end model, set `general.load_weights` in [general_default.yaml](configs/general/general_default.yaml) to its checkpoint path. To initialize the encoder and decoder separately, set `general.encoder` and `general.decoder` to their respective checkpoint paths.

## License

DiffMS is released under the [MIT](LICENSE.txt) license.

## Contact

If you have any questions, please reach out to mbohde@tamu.edu or liuhx25@mit.edu.

## Reference
If you find this codebase useful in your research, please kindly cite the following manuscript
```
@article{bohde2025diffms,
  title={DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra},
  author={Bohde, Montgomery and Manjrekar, Mrunali and Wang, Runzhong and Ji, Shuiwang and Coley, Connor W},
  journal={arXiv preprint arXiv:2502.09571},
  year={2025}
}
```
