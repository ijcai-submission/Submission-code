# Submission-code

### Dependencies
Python 2.7

Pytorch

Matlab



**Data**
  - Download the ASVspoof 2017 data (Version 1) for [training, validation and evaluation](https://datashare.is.ed.ac.uk/handle/10283/2778) and extract it to `data`.
  - Run the Matlab script `Feature_extraction.m` to generate GD-grams for training, validation and evaluation data.


**Training**
  - Train Stage-I network `python Train_stage1.py`
  - Generate attentionally weighted train, validation and evaluation set `python resnetCAMtraingen.py`
  `python resnetCAMtrainspo.py` `python resnetCAMdevgen.py` `python resnetCAMdevspo.py` `python resnetCAMevalgen.py` `python resnetCAMevalspo.py`
  - Train Stage-II network `python Train_stage2.py`





**Evaluating**
- Generate predictions for evaluation set `python Generate_predictions2.py`
- Evaluate model by running the Matlab script `Evaluation.m`

