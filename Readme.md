# Flattery Prediction from Speech and Text 

## Getting started 

First, install all requirements in ``requirements.txt``.

We fine-tune RoBERTa-base with 5 different seeds. The best of these models, fine-tuned with the gold standard transcripts, is available at [the huggingface hub](TODO).
Analogously, we fine-tune [Wav2Vec2-MSP](TODO) with 5 different seeds and make the best of these models [available]().

This repository contains the code to reproduce the SVM-based experiments. For this, the features must be downloaded:
[zenodo-link here](TODO) and placed in the top-level directory.  

## Simple SVM experiments 
I.e., experiments using only one set of features. The script ``svm_experiments.py`` implements a grid search for SVM-based classification.
Example call:

`` 
TODO
``

For details, see ``parse_args()`` or call the script with the ``--help`` option. 


## Early Fusion SVM experiments

I.e., experiments based on feature-level fusion for finetuned audio and text Transformers.
These experiments differ from the "simple" ones in that they utilise a range of different seeds w.r.t. the 
models to be fused. This is best illustrated via an example call:

``` 
TODO
```

Here, ``--n_seeds``=3 rows of experiments will be run and their average results reported: 
1) based on (``TODO_t1``, ``TODO_t2``), 2) based on etc. TODO 

## Late Fusion 
Late Fusion simply fuses two sets of predictions of our finetuned models, as can be found in the ``predictions``
directory. 
Similar to the early fusion experiments, it expects these predictions to be organised based on the seeds the models were 
trained with. See the directories in ``predictions`` and an example call for clarification:

`` 
TODO
``

The script support unweighted and weighted (based on development set performance) late fusion via the parameter ``--method``