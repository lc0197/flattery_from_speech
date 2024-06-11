# Flattery Prediction from Speech and Text 

[TODO add paper link](TODO)

## Getting started 

First, install all requirements in ``requirements.txt``.

We fine-tune RoBERTa-base with 5 different seeds. The best of these models, fine-tuned with the gold standard transcripts, is available at [the huggingface hub](TODO).
Analogously, we fine-tune [Wav2Vec2-MSP](TODO) with 5 different seeds and make the best of these models [available]().

This repository contains the code to reproduce the SVM-based experiments. For this, the features must be downloaded [from zenodo](https://zenodo.org/records/11561487) and placed in the ``data`` directory like this:

```
|-- data
    |-- features
        |-- w2v2_msp_hidden_11.csv
        |-- ...
    |--- final_db.csv
    |--- split.csv
|-- src
    |-- ...
|--- ...
```

## Simple SVM experiments 
I.e., experiments using only one set of features. The script ``svm_experiments.py`` implements a grid search for SVM-based classification.
Example call:

`` 
python3 svm_experiments_simple.py --feature w2v2_base_960_hidden_7_ --Cs 0.1 1. --kernels linear rbf
``

This call would run a grid search for the given feature using the 
two given parameters for ``C`` and ``kernel`` in ``sklearn.svm.SVC``. The logs would be placed in 
a folder under ``logs/svm/w2v2_base_960_hidden_7_``.  

For all parameters and details, see ``parse_args()`` or call the script with the ``--help`` option. 


## Early Fusion SVM experiments

I.e., experiments based on feature-level fusion for finetuned audio and text Transformers.
These experiments differ from the "simple" ones in that they utilise a range of different seeds w.r.t. the 
models to be fused. This is best illustrated via an example call:

``` 
python svm_early_fusion.py --feature_a w2v_ft --feature_t gold_roberta --Cs 0.01 0.1 1. --seed 101 --n_seeds 3
```

Here, ``--n_seeds``=3 rows of experiments (grid searches over the given ``C`` values) 
will be run and their average results reported: 
1) based on the features (``gold_roberta_101(.csv)``, ``w2v_ft_101(.csv)``), 
2) based on  (``gold_roberta_102(.csv)``, ``w2v_ft_102(.csv)``) and 
3) based on  (``gold_roberta_103(.csv)``, ``w2v_ft_103(.csv)``).

The logs are to be found in ``logs/svm_early_fusion/{feature_a}_{feature_t}``.

## Late Fusion 
Late Fusion simply fuses two sets of predictions of our finetuned models, as can be found in the ``predictions``
directory. 
Similar to the early fusion experiments, it expects these predictions to be organised based on the seeds the models were 
trained with. See the directories in ``predictions`` and an example call for clarification:

`` 
python late_fusions.py --method unweighted --audio w2v_msp --text roberta_base_gold
``

This would look up and fuse all the predictions given in ``predictions/audio/w2v_msp`` and ``predictions/textual/roberta_base_gold``, respectively.
Similar to early fusion, the script runs an experiment for every seed (i.e. 101, 102,...) it finds in the predictions directory and reports average results.
The script support unweighted and weighted (based on development set performance) late fusion via the parameter ``--method``
