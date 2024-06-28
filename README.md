## Description
The repository provides a comprehensive framework for generating interpretable counterfactual explanations black-box outcomes, with a focus on churn prediction. It includes implementations of various explanation methods such as DiCE, GS, CP-ILS, and CFRL, along with a script for ensemble creation and selection of the best counterfactuals based on multiple evaluation metrics.

## Contents
- **Classifiers:** This folder contains the trained black box model used for churn prediction.
- **Counterfactual Explanations:** This folder contains local packages for generating counterfactual explanations for LORE_SA and Growing Spheres.
- **Unicredit_Churn.ipynb:** This notebook provides the data preprocessing and machine learning pipeline for training the churn prediction model.
- **cfrl.py:** This script implements the Counterfactual Reinforcement Learning method for generating counterfactual explanations.
- **cp_ils_new.py:** This script implements the CP-ILS method for generating counterfactual explanations.
- **dice.py:** This script implements the DiCE method for generating counterfactual explanations.
- **growing_spheres.py:** This script implements the Growing Spheres method for generating counterfactual explanations.
- **ensemble.py:** This script implements an ensemble approach of counterfactual explanations from the above methods, DiCE, GS, T-LACE, and CFRL. It then selects the top counterfactuals for each instance based on a weighted combination of metrics such as Proximity, Sparsity, Plausibility, and Diversity. The selected counterfactuals are chosen to provide the most informative and diverse explanations for model predictions.
- **environment.yml:** This file contains dependencies required to set up the environment for running the code in this repository.

## Usage
1. Ensure that the required dependencies are installed by setting up the environment using the environment.yml file.
2. Run the Churn.ipynb notebooks to preprocess the data and train the churn prediction model.
3. Utilize the 4 scripts provided in the repository to generate counterfactual explanations from the test set.
4. Run ensemble.py script for evaluation and selection of top-performing counterfactual examples.

