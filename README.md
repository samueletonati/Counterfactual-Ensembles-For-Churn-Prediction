## Description
The repository provides a comprehensive framework for generating interpretable counterfactual explanations black-box outcomes, with a focus on churn prediction. It includes implementations of various explanation methods such as DiCE, GS, T-LACE, and CFRL, along with a script for ensemble creation and selection of the best counterfactuals based on multiple evaluation metrics, as well as a comparison between real and synthetic explanations and formulation of a minimization problem to minimize kl divergence between real and synthetic ensembles.

## Contents
- **Classifiers:** This folder contains the trained black box model used for churn prediction.
- **Counterfactual Explanations:** This folder contains local packages for generating counterfactual explanations for Growing Spheres.
- **cfrl.py:** This script implements the Counterfactual Reinforcement Learning method for generating counterfactual explanations.
- **tlace.py:** This script implements the T-LACE method for generating counterfactual explanations.
- **dice.py:** This script implements the DiCE method for generating counterfactual explanations.
- **growing_spheres.py:** This script implements the Growing Spheres method for generating counterfactual explanations.
- **ensemble.py:** This script implements an ensemble approach of counterfactual explanations from the above methods, DiCE, GS, T-LACE, and CFRL. It then selects the top counterfactuals for each instance based on a weighted combination of metrics such as Proximity, Sparsity, Plausibility, and Diversity. The selected counterfactuals are chosen to provide the most informative and diverse explanations for model predictions.
- **Minimize KL on clusters.ipynb:** This notebook contains the minimization problem and cluster analysis on real vs synth ensembles.
- **requirements.txt:** This file contains dependencies required to set up the environment for running the code in this repository.

