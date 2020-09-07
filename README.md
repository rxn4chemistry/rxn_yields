# Predicting Chemical Reaction Yields
> Predicting the yield of a chemical reaction from a reaction SMILES using Transformers


Artificial intelligence is driving one of the most important revolutions in organic chemistry. Multiple platforms, including tools for reaction prediction and synthesis planning based on machine learning, successfully became part of the organic chemists’ daily laboratory, assisting in domain-specific synthetic problems. Unlike reaction prediction and retrosynthetic models, reaction yields models have been less investigated, despite the enormous potential of accurately predicting them. Reaction yields models, describing the percentage of the reactants that is converted to the desired products, could guide chemists and help them select high-yielding reactions and score synthesis routes, reducing the number of attempts. So far, yield predictions have been predominantly performed for high-throughput experiments using a categorical (one-hot) encoding of reactants, concatenated molecular fingerprints, or computed chemical descriptors. Here, we extend the application of natural language processing architectures to predict reaction properties given a text-based representation of the reaction, using an encoder transformer model combined with a regression layer. We demonstrate outstanding prediction performance on two high-throughput experiment reactions sets. An analysis of the yields reported in the open-source USPTO data set shows that their distribution differs depending on the mass scale, limiting the dataset applicability in reaction yields predictions.

This repository complements our study on predicting chemical reaction yields, which can currently be found on [ChemRxiv](https://chemrxiv.org/articles/preprint/Prediction_of_Chemical_Reaction_Yields_using_Deep_Learning/12758474). 

## Install

As the library is based on the chemoinformatics toolkit [RDKit](http://www.rdkit.org) it is best installed using the [Anaconda](https://docs.conda.io/en/latest/miniconda.html) package manager. Once you have conda, you can simply run:

```
conda create -n yields python=3.6 -y
conda activate yields
conda install -c rdkit rdkit=2020.03.3.0 -y
conda install -c tmap tmap -y
```

```
git clone https://github.com/rxn4chemistry/rxn_yields.git
cd rxn_yields
pip install -e .
```

## Approach - predicting yields from reaction SMILES

Transformer models have recently revolutionised Natural Language Processing and were also successfully applied to task in chemistry, using a text-based representation of molecules and chemical reactions called Simplified molecular-input line-entry system (SMILES). 

Sequence-2-Sequence transformers as in [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need) were used for:
- Chemical Reaction Prediction
    - [Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction](https://pubs.acs.org/doi/full/10.1021/acscentsci.9b00576)
    - [Carbohydrate Transformer: Predicting Regio- and Stereoselective Reactions Using Transfer Learning](http://dx.doi.org/10.26434/chemrxiv.11935635)
- Multi-step retrosynthesis
    - [Predicting retrosynthetic pathways using a combined linguistic model and hyper-graph exploration strategy](http://dx.doi.org/10.1039/c9sc05704h)
    - [Unassisted Noise-Reduction of Chemical Reactions Data Sets](https://chemrxiv.org/articles/Unassisted_Noise-Reduction_of_Chemical_Reactions_Data_Sets/12395120/1)
    
Encoder Transformers like [BERT](https://openreview.net/forum?id=SkZmKmWOWH) and [ALBERT](https://openreview.net/forum?id=H1eA7AEtvS) for:
- Reaction fingerprints and classification
    - [Mapping the Space of Chemical Reactions using Attention-Based Neural Networks](https://chemrxiv.org/articles/Data-Driven_Chemical_Reaction_Classification_with_Attention-Based_Neural_Networks/9897365)
- Atom rearrangements during chemical reactions
    - [Unsupervised Attention-Guided Atom-Mapping](https://chemrxiv.org/articles/Unsupervised_Attention-Guided_Atom-Mapping/12298559)
    
Those studies show that Transformer models are able to learn organic chemistry and chemical reactions from SMILES.

Here we asked the question, how well a **BERT** model would perform when applied to a **yield prediction** task:


<div style="text-align: center">
<img src="nbs/images/pipeline.jpg" width="800">
<p style="text-align: center;"> <b>Figure:</b> Pipeline and task description. </p>
</div>

To do so, we started with the reaction fingerprint models from the [rxnfp](https://rxn4chemistry.github.io/rxnfp/) library and added a fine-tuning regression head through [SimpleTransformers.ai](https://simpletransformers.ai). As we don't need to change the hyperparameters of the base model, we only tune the learning rate for the training and the dropout probability. 

We explored two high-throughput experiment (HTE) data sets and then also the yields data found in the USPTO data base.

## Buchwald-Hartwig HTE data set

One of the best studied reaction yield is the one that was published by Ahneman et al. in [Predicting reaction performance in C–N cross-coupling using machine learning](https://science.sciencemag.org/content/360/6385/186.full), where the authors have used DFT-computed descriptors as inputs to different machine learning descriptors. There best model was a random forest model. More recently, [one-hot encodings](https://science.sciencemag.org/content/362/6416/eaat8603) and [multi-fingerprint features (MFF)](https://www.sciencedirect.com/science/article/pii/S2451929420300851) as input representations were investigated. Here, we show competitive results starting simply from a text-based reaction SMILES input to our models.

<div style="text-align: center">
<img src="nbs/images/buchwald_hartwig.jpg" width="800">
<p style="text-align: center;"> <b>Figure:</b> a) Summary of the results on the Buchwald–Hartwig data set. b) Example regression plot for the first random-split. </p>
</div>

## Suzuki-Miyaura HTE data set

Another yield data set is the one of Perera et al. from [A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow](https://science.sciencemag.org/content/359/6374/429). Using 10 random splits, we demonstrate that the hyperparameters optimised on the Buchwald-Hartwig were transferable to a different HTE reaction data set.

<div style="text-align: center">
<img src="nbs/images/suzuki_miyaura.jpg" width="600">
<p style="text-align: center;"> <b>Figure:</b> Summary of the results on the Suzuki-Miyaura data set, using the hyperparameters of the Buchwald-Hartwig reactions (3.1) and those tune on one of the random splits. </p>
</div>

## Reaction discovery

Training on a reduced part of the training set containing 5%, 10% 20% of the data, we show that the models are already able to find high-yielding reactions in the remaining data set. 

<div style="text-align: center">
<img src="nbs/images/reaction_discovery.jpg" width="700">
<p style="text-align: center;"> <b>Figure:</b> Average and standard deviation of the yields for the 10, 50, and 100 reactions predicted to have the highest yields after training on a fraction of the data set (5%, 10%, 20%). The ideal reaction selection and a random selection are plotted for comparison. </p>
</div>

## USPTO data sets 

The [largest public reaction data](https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873) set was text-mined from the US patents by Lowe. There are numerous reasons why yield data in patents is noisy. Therefore, it the data set is not ideal for a yield prediction task. Using Reaction Atlases, made with the tutorial shown in [rxnfp](https://rxn4chemistry.github.io/rxnfp/), we show that while general yield trends exists. The local reaction neighbourhoods are very noisy and better quality data would be required.

<div style="text-align: center">
<img src="nbs/images/tmaps_uspto.jpg" width="700">
<p style="text-align: center;"> <b>Figure:</b> Superclasses, yields and yield distribution of the reactions in the USPTO data set divided into gram and subgram scale. </p>
</div>

We performed different experiments using random and time splits on the reaction data. As a sanity check, we compared the results to the one where we randomise the yields of the reactions. In one of the experiments, we smoothed the yield data taking the average of twice its own yield plus the yields of the three nearest-neighbours. This procedure could potentially improve the data set quality by smoothing out originating from (human) errors. Accordingly, the results of the Yield-BERT on the smoothed data set are better than on the original yields. 

<div style="text-align: center">
<img src="nbs/images/uspto.jpg" width="500">
<p style="text-align: center;"> <b>Figure:</b> Summary of the results on the USPTO data sets. </p>
</div>

## Citation

If you found this repo useful, please cite our [manuscript](https://chemrxiv.org/articles/preprint/Prediction_of_Chemical_Reaction_Yields_using_Deep_Learning/12758474).

```
@article{Schwaller2020yields,
author = "Philippe Schwaller and Alain C. Vaucher and Teodoro Laino and Jean-Louis Reymond",
title = "{Prediction of Chemical Reaction Yields using Deep Learning}",
year = "2020",
month = "8",
url = "https://chemrxiv.org/articles/preprint/Prediction_of_Chemical_Reaction_Yields_using_Deep_Learning/12758474",
doi = "10.26434/chemrxiv.12758474.v1"
}
```

The models used in our work are based on the [Huggingface Transformers](https://github.com/huggingface/transformers) library and interfaced through [SimpleTransformers.ai](https://simpletransformers.ai).
