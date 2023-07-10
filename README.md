<p align="center">
  <img src="assets/logo.png" align="center" width="80%">
</p>

--------------------------------------------------------------------------------

[![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/YyzHarry/SubpopBench/blob/main/LICENSE)
![](https://img.shields.io/github/stars/YyzHarry/SubpopBench)
![](https://img.shields.io/github/forks/YyzHarry/SubpopBench)
![](https://visitor-badge.laobi.icu/badge?page_id=YyzHarry.SubpopBench&right_color=%23FFA500)

## Overview

**SubpopBench** is a benchmark of _subpopulation shift_.
It is a living PyTorch suite containing benchmark datasets and algorithms for subpopulation shift, as introduced in [Change is Hard: A Closer Look at Subpopulation Shift](https://arxiv.org/abs/2302.12254) (Yang et al., ICML 2023).


## Contents

Currently we support [13 datasets](./subpopbench/dataset/datasets.py) and [~20 algorithms](./subpopbench/learning/algorithms.py) that span different learning strategies.
Feel free to send us a PR to add your algorithm / dataset for subpopulation shift.

### Available Algorithms

The [currently available algorithms](./subpopbench/learning/algorithms.py) are:

* Empirical Risk Minimization (**ERM**, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
* Invariant Risk Minimization (**IRM**, [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
* Group Distributionally Robust Optimization (**GroupDRO**, [Sagawa et al., 2020](https://arxiv.org/abs/1911.08731))
* Conditional Value-at-Risk Distributionally Robust Optimization (**CVaRDRO**, [Duchi and Namkoong, 2018](https://arxiv.org/abs/1810.08750))
* Mixup (**Mixup**, [Zhang et al., 2018](https://arxiv.org/abs/1710.09412))
* Just Train Twice (**JTT**, [Liu et al., 2021](http://proceedings.mlr.press/v139/liu21f.html))
* Learning from Failure (**LfF**, [Nam et al., 2020](https://proceedings.neurips.cc/paper/2020/file/eddc3427c5d77843c2253f1e799fe933-Paper.pdf))
* Learning Invariant Predictors with Selective Augmentation (**LISA**, [Yao et al., 2022](https://arxiv.org/abs/2201.00299))
* Deep Feature Reweighting (**DFR**, [Kirichenko et al., 2022](https://arxiv.org/abs/2204.02937))
* Maximum Mean Discrepancy (**MMD**, [Li et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf))
* Deep Correlation Alignment (**CORAL**, [Sun and Saenko, 2016](https://arxiv.org/abs/1607.01719))
* Data Re-Sampling (**ReSample**, [Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
* Cost-Sensitive Re-Weighting (**ReWeight**, [Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
* Square-Root Re-Weighting (**SqrtReWeight**, [Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
* Focal Loss (**Focal**, [Lin et al., 2017](https://arxiv.org/abs/1708.02002))
* Class-Balanced Loss (**CBLoss**, [Cui et al., 2019](https://arxiv.org/abs/1901.05555))
* Label-Distribution-Aware Margin Loss (**LDAM**, [Cao et al., 2019](https://arxiv.org/abs/1906.07413))
* Balanced Softmax (**BSoftmax**, [Ren et al., 2020](https://arxiv.org/abs/2007.10740))
* Classifier Re-Training (**CRT**, [Kang et al., 2020](https://arxiv.org/abs/1910.09217))

Send us a PR to add your algorithm! Our implementations use the hyper-parameter grids [described here](./subpopbench/hparams_registry.py).

### Available Datasets

The [currently available datasets](./subpopbench/dataset/datasets.py) are:

* ColoredMNIST ([Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
* Waterbirds ([Wah et al., 2011](https://authors.library.caltech.edu/27452/))
* CelebA ([Liu et al., 2015](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
* MetaShift ([Liang and Zou, 2022](https://arxiv.org/abs/2202.06523))
* CivilComments ([Borkan et al., 2019](https://arxiv.org/abs/1903.04561)) from the [WILDS benchmark](https://arxiv.org/abs/2012.07421)
* MultiNLI ([Williams et al., 2017](https://arxiv.org/abs/1704.05426))
* MIMIC-CXR ([Johnson et al., 2019](https://www.nature.com/articles/s41597-019-0322-0))
* CheXpert ([Irvin et al., 2019](https://arxiv.org/abs/1901.07031))
* CXRMultisite ([Puli et al., 2021](https://openreview.net/forum?id=12RoR2o32T))
* MIMICNotes ([Johnson et al., 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4878278/))
* NICO++ ([Zhang et al., 2022](https://arxiv.org/abs/2204.08040))
* ImageNetBG ([Xiao et al., 2020](https://openreview.net/forum?id=gl3D-xY7wLq))
* Living17 ([Santurkar et al., 2020](https://openreview.net/pdf?id=mQPBmvyAuk)) from the [BREEDS benchmark](https://openreview.net/pdf?id=mQPBmvyAuk)

Send us a PR to add your dataset! You can follow the dataset format [described here](./subpopbench/dataset/datasets.py).

### Model Architectures & Pretraining Methods

The [supported image architectures](./subpopbench/models/networks.py) are:

* ResNet-50 on ImageNet-1K using supervised pretraining (`resnet_sup_in1k`)
* ResNet-50 on ImageNet-21K using supervised pretraining (`resnet_sup_in21k`, [Ridnik et al., 2021](https://arxiv.org/pdf/2104.10972v4.pdf))
* ResNet-50 on ImageNet-1K using SimCLR (`resnet_simclr_in1k`, [Chen et al., 2020](https://arxiv.org/abs/2002.05709))
* ResNet-50 on ImageNet-1K using Barlow Twins (`resnet_barlow_in1k`, [Zbontar et al., 2021](https://arxiv.org/abs/2103.03230))
* ResNet-50 on ImageNet-1K using DINO (`resnet_dino_in1k`, [Caron et al., 2021](https://arxiv.org/abs/2104.14294))
* ViT-B on ImageNet-1K using supervised pretraining (`vit_sup_in1k`, [Steiner et al., 2021](https://arxiv.org/abs/2106.10270))
* ViT-B on ImageNet-21K using supervised pretraining (`vit_sup_in21k`, [Steiner et al., 2021](https://arxiv.org/abs/2106.10270))
* ViT-B from OpenAI CLIP (`vit_clip_oai`, [Radford et al., 2021](https://arxiv.org/abs/2103.00020))
* ViT-B pretrained using CLIP on LAION-2B (`vit_clip_laion`, [OpenCLIP](https://github.com/mlfoundations/open_clip))
* ViT-B on SWAG using weakly supervised pretraining (`vit_sup_swag`, [Singh et al., 2022](https://arxiv.org/abs/2201.08371))
* ViT-B on ImageNet-1K using DINO (`vit_dino_in1k`, [Caron et al., 2021](https://arxiv.org/abs/2104.14294))

The [supported text architectures](./subpopbench/models/networks.py) are:

* BERT-base-uncased (`bert-base-uncased`, [Devlin et al., 2018](https://arxiv.org/abs/1810.04805))
* GPT-2 (`gpt2`, [Radford et al., 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf))
* RoBERTa-base-uncased (`xlm-roberta-base`, [Liu et al., 2019](https://arxiv.org/abs/1907.11692))
* SciBERT (`allenai/scibert_scivocab_uncased`, [Beltagy et al., 2019](https://arxiv.org/abs/1903.10676))
* DistilBERT-uncased (`distilbert-base-uncased`, [Sanh et al., 2019](https://arxiv.org/abs/1910.01108))

Note that text architectures are only compatible with `CivilComments`.

### Subpopulation Shift Scenarios

We characterize four basic types of subpopulation shift using our framework, and categorize each dataset into its most dominant shift type.

* **Spurious Correlations (SC)**: certain $a$ is spuriously correlated with $y$ in training but not in testing.
* **Attribute Imbalance (AI)**: certain attributes are sampled with a much smaller probability than others in $p_{\text{train}}$, but not in $p_{\text{test}}$.
* **Class Imbalance (CI)**: certain (minority) classes are underrepresented in $p_{\text{train}}$, but not in $p_{\text{test}}$.
* **Attribute Generalization (AG)**: certain attributes can be totally missing in $p_{\text{train}}$, but present in $p_{\text{test}}$.

### Evaluation Metrics

We include [a variety of metrics](./subpopbench/utils/eval_helper.py) aiming for a thorough evaluation from different aspects:

* Average Accuracy & Worst Accuracy
* Average Precision & Worst Precision
* Average F1-score & Worst F1-score
* Adjusted Accuracy
* Balanced Accuracy
* AUROC & AUPRC
* Expected Calibration Error (ECE) 

### Model Selection Criteria

We highlight the impact of whether attribute is known in (1) _training set_ and (2) _validation set_,
where the former is specified by `--train_attr` in [`train.py`](./subpopbench/train.py),
and the latter is specified by [model selection criteria](./subpopbench/learning/model_selection.py).
We show a few important selection criteria:

* `OracleWorstAcc`: Picks the best test-set worst-group accuracy (oracle)
* `ValWorstAccAttributeYes`: Picks the best val-set worst-group accuracy (attributes _known_ in validation)
* `ValWorstAccAttributeNo`: Picks the best val-set worst-class accuracy (attributes _unknown_ in validation; group degenerates to class)


## Getting Started

### Installation

#### Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```bash
git clone git@github.com:YyzHarry/SubpopBench.git
cd SubpopBench/
conda env create -f environment.yml
conda activate subpop_bench
```

#### Downloading Data
Download the original datasets and generate corresponding metadata in your `data_path`:

```bash
python -m subpopbench.scripts.download --data_path <data_path> --download
```

For `MIMICNoFinding`, `CheXpertNoFinding`, `CXRMultisite`, and `MIMICNotes`, see [MedicalData.md](./MedicalData.md) for instructions for downloading the datasets manually.


### Code Overview

#### Main Files
- [`train.py`](./subpopbench/train.py): main training script
- [`sweep.py`](./subpopbench/sweep.py): launch a sweep with all selected algorithms (provided in `subpopbench/learning/algorithms.py`) on all subpopulation shift datasets
- [`collect_results.py`](./subpopbench/scripts/collect_results.py): collect sweep results to automatically generate result tables (as in the paper)


#### Main Arguments
- __train.py__:
    - `--dataset`: name of chosen subpopulation dataset
    - `--algorithm`: choose algorithm used for running
    - `--train_attr`: whether attributes are known or not during training (`yes` or `no`)
    - `--data_dir`: data path
    - `--output_dir`: output path
    - `--output_folder_name`: output folder name (under `output_dir`) for the current run
    - `--hparams_seed`: seed for different hyper-parameters
    - `--seed`: seed for different runs
    - `--stage1_folder` & `--stage1_algo`: arguments for two-stage algorithms
    - `--image_arch` & `--text_arch`: model architecture and source of initial model weights (text architectures only compatible with `CivilComments`)
- __sweep.py__:
    - `--n_hparams`: how many hparams to run for each <dataset, algorithm> pair
    - `--best_hp` & `--n_trials`: after sweeping hparams, fix best hparam and run trials with different seeds


### Usage

#### Train a single model (with unknown attributes)
```bash
python -m subpopbench.train \
       --algorithm <algo> \
       --dataset <dset> \
       --train_attr no \
       --data_dir <data_path> \
       --output_dir <output_path> \
       --output_folder_name <output_folder_name>
```

#### Train a model using 2-stage methods, e.g., DFR (with known attributes)
```bash
python -m subpopbench.train \
       --algorithm DFR \
       --dataset <dset> \
       --train_attr yes \
       --data_dir <data_path> \
       --output_dir <output_path> \
       --output_folder_name <output_folder_name> \
       --stage1_folder <stage1_model_folder> \
       --stage1_algo <stage1_algo>
```

#### Launch a sweep with different hparams (with unknown attributes)
```bash
python -m subpopbench.sweep launch \
       --algorithms <...> \
       --dataset <...> \
       --train_attr no \
       --n_hparams <num_of_hparams> \
       --n_trials 1
```

#### Launch a sweep after fixing hparam with different seeds (with unknown attributes)
```bash
python -m subpopbench.sweep launch \
       --algorithms <...> \
       --dataset <...> \
       --train_attr no \
       --best_hp \
       --input_folder <...> \
       --n_trials <num_of_trials>
```

#### Collect the results of your sweep
```bash
python -m subpopbench.scripts.collect_results --input_dir <...>
```


## Updates
- __[07/2023]__ Check out the [Oral talk video](https://www.youtube.com/watch?v=WiSrCWAAUNI) (10 mins) for our ICML paper.
- __[05/2023]__ Paper accepted to [ICML 2023](https://icml.cc/Conferences/2023).
- __[02/2023]__ [arXiv version](https://arxiv.org/abs/2302.12254) posted. Code is released.


## Acknowledgements
This code is partly based on the open-source implementations from [DomainBed](https://github.com/facebookresearch/DomainBed), [spurious_feature_learning](https://github.com/izmailovpavel/spurious_feature_learning), and [multi-domain-imbalance](https://github.com/YyzHarry/multi-domain-imbalance).


## Citation
If you find this code or idea useful, please cite our work:
```bib
@inproceedings{yang2023change,
  title={Change is Hard: A Closer Look at Subpopulation Shift},
  author={Yang, Yuzhe and Zhang, Haoran and Katabi, Dina and Ghassemi, Marzyeh},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```


## Contact
If you have any questions, feel free to contact us through email (yuzhe@mit.edu & haoranz@mit.edu) or GitHub issues. Enjoy!
