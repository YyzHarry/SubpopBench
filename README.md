# Subpopulation Shift Benchmark

This repository contains the implementation for paper: [Change is Hard: A Closer Look at Subpopulation Shift](https://arxiv.org/abs/xxx) (Yang et al., 2023).
It is also a living PyTorch suite containing benchmark datasets and algorithms for subpopulation shift.


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
* MultiNLI ([Williams et al., 2017](https://arxiv.org/abs/1704.05426)
* MIMIC-CXR ([Johnson et al., 2019](https://www.nature.com/articles/s41597-019-0322-0))
* CheXpert ([Irvin et al., 2019](https://arxiv.org/abs/1901.07031))
* CXRMultisite ([Puli et al., 2021](https://openreview.net/forum?id=12RoR2o32T))
* MIMICNotes ([Johnson et al., 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4878278/))
* NICO++ ([Zhang et al., 2022](https://arxiv.org/abs/2204.08040))
* ImageNetBG ([Xiao et al., 2020](https://openreview.net/forum?id=gl3D-xY7wLq))
* Living17 ([Santurkar et al., 2020](https://openreview.net/pdf?id=mQPBmvyAuk)) from the [BREEDS benchmark](https://openreview.net/pdf?id=mQPBmvyAuk)

Send us a PR to add your dataset!

### Model Architectures & Pretraining Methods

[**TODO: Haoran**]

### Subpopulation Shift Scenarios

We characterize four basic types of subpopulation shift using our framework, and categorize each dataset into its most dominant shift type.

* **Spurious Correlations (SC)** happen when certain $a$ is spuriously correlated with $y$ in training but not in testing.
* **Attribute Imbalance (AI)** happens when certain attributes are sampled with a much smaller probability than others in $p_{\text{train}}$, but not in $p_{\text{test}}$.
* **Class Imbalance (CI)** happens when certain classes are biased in $p_{\text{train}}$, leading to higher prediction confidence for majority classes.
* **Attribute Generalization (AG)** happens when certain attributes can be totally missing in $p_{\text{train}}$, but present in $p_{\text{test}}$.

### Evaluation Metrics

We include a variety of metrics aiming for a thorough evaluation from different aspects:

* Average Accuracy & Worst Accuracy
* Average Precision & Worst Precision
* Average F1-score & Worst F1-score
* Adjusted Accuracy
* Balanced Accuracy
* AUROC & AUPRC
* Expected Calibration Error (ECE) 

### Model Selection Criteria

We systematically investigate whether attribute is known in both (1) _training set_ and (2) _validation set_,
where the former case is specified by argument `--train_attr` in [`train.py`](./subpopbench/train.py),
while the latter case is specified by [model selection criteria](./subpopbench/learning/model_selection.py).
There are many [currently available model selection criteria](./subpopbench/dataset/datasets.py) using different metrics defined above; we highlight a few important ones:

* `OracleWorstAcc`: Picks the best test-set worst-group accuracy (oracle)
* `ValWorstAccAttributeYes`: Picks the best validation-set worst-group accuracy (attributes _known_ in validation set)
* `ValWorstAccAttributeNo`: Picks the best validation-set worst-class accuracy (attributes _unknown_ in validation set; group degenerates to class)


## Getting Started

### Installation

#### Prerequisites
Download the original datasets and generate corresponding metadata in your `data_path`
```bash
python -m subpopbench.scripts.download --data_path <data_path> --download
```

[**TODO: Haoran**] Note that for ..., you will need to download manually... please refer to XXX for details 

#### Dependencies

You can install the dependencies for SubpopBench using

[**TODO: Haoran** - change to env.yml and delete .txt file] 
```bash
pip install -r requirements.txt
```


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
- __[02/2023]__ [arXiv version](https://arxiv.org/abs/xxx) posted. Code is released.


## Acknowledgements
This code is partly based on the open-source implementations from [DomainBed](https://github.com/facebookresearch/DomainBed) and [multi-domain-imbalance](https://github.com/YyzHarry/multi-domain-imbalance).


## Citation
If you find this code or idea useful, please cite our work:
```bib
@article{yang2023change,
  title={Change is Hard: A Closer Look at Subpopulation Shift},
  author={Yang, Yuzhe and Zhang, Haoran and Katabi, Dina and Ghassemi, Marzyeh},
  journal={arXiv preprint arXiv:xxx},
  year={2023}
}
```


## Contact
If you have any questions, feel free to contact us through email (yuzhe@mit.edu & haoranz@mit.edu) or Github issues. Enjoy!
