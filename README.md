# PPML-TSA

This repository provides all code necessary to reproduce the results reported in our paper `Evaluating Privacy-Preserving Machine Learning in Critical Infrastructures: A Case Study on Time-Series Classification`[[IEEE](#)][[arXiv](#)].

<strong>Abstract</strong>: With the advent of machine learning in applications of critical infrastructure such as healthcare and energy, privacy is a growing concern in the minds of stakeholders.It is pivotal to ensure that neither the model nor the data can be used to extract sensitive information used by attackers against individuals or to harm whole societies through the exploitation of critical infrastructure. The applicability of machine learning in these domains is mostly limited due to a lack of trust regarding the transparency and the privacy constraints. Various safety-critical use cases (mostly relying on time-series data) are currently underrepresented in privacy-related considerations.By evaluating several privacy-preserving methods regarding their applicability on time-series data, we validated the inefficacy of encryption for deep learning, the strong dataset dependence of differential privacy, and the broad applicability of federated methods. 

## Requirements

An appropriate Python environment can be set up using the `src/requirements.txt` files provided in the repo. The respective datasets can be downloaded from the [UEA & UCR Time Series Classification Repository](https://www.timeseriesclassification.com/dataset.php) and should be placed in the `/data/` folder.

## Basic Usage

Results can be reproduced by running the corresponding bash scripts located in the subfolders of `/src/bash_scripts/` as outlined in the table below. Models are savd in `/models/` and resulting evaluation files are placed under `/results/`.

Experiment|Scripts
---|:--
Experiment 0 - Train the baselines|`execute_baseline.sh` <br /> `execute_baseline_architecture.sh`
Experiment 1 - Performance Benchmarking|`DM/execute_differential_privacy.sh` <br /> `DM/execute_federated_learning.sh` <br /> `AL/execute_federated_ensemble.sh`
Experiment 2 - Architecture comparison|`DM/execute_differential_privacy_architecture.sh` <br /> `DM/execute_federated_learning_architecture.sh` <br /> `AL/execute_federated_ensemble_architecture.sh`
Experiment 3 - Differential Privacy: Hyperparameter Evaluation|`DM/execute_differential_privacy_detailed.sh`
Experiment 4 - Federated Ensemble: Ensemble Size Evaluation|`AL/execute_federated_ensemble_detailed.sh`
Experiment 5 - Differential Privacy in a Federated Setting|`AL/execute_DPFE.sh`
Experiment 6 - Secret Sharing Runtime Evaluation|`DM/execute_crypten_timing.sh`

## Citation

Please consider citing our associated [paper](#):
```
    @article{mercier2021evaluating,
        title={Evaluating Privacy-Preserving Machine Learning in Critical Infrastructures: A Case Study on Time-Series Classification},
        author={Mercier, Dominique and Lucieri, Adriano and Munir, Mohsin and Dengel, Andreas and Ahmed, Sheraz},
        journal={IEEE Transactions on Industrial Informatics},
        year={2021}
    }
```