# Hierarchical Multi-label Classification
## About

This directory contains the code and resources of the project: **Hierarchical Multi-label Classification (HMC)**. It is a final course project of *Natural Language Processing and Deep Learning, 2022 Fall*.

1. In this project, a DeBERTa-based model is implemented to solve the HMC problem via hierarchical label retrive. 
2. Since the data of this task are long documents (the longest document has more than 10,000 tokens after tokenization), a slide-window based hierarchical encoder is implemented using LSTM.
3. Please contact Haowei Lin (linhaowei@pku.edu.cn) if you have issues using this software.

## Overview of the Model
<p align="center">
<img  src="figure/model.pdf" width="700"> 
</p>

### Step 1. Passage Split
The procedure is aimed at spliting one long document into small pieces. Since the transformer model is processing the context with $\mathcal O(n^2)$ complexity, we constrain the input of DeBERTa model to small blocks with length 256 (tokens). Here we preserve some overlapping text (32 tokens) between different blocks, and simply run the sliding window algorithm.

### Step 2. Passage Embedding
In this step, we use a DeBERTa model to extract features from passage blocks. We use the same context pooling method from DeBERTa original paper [1]. To aggregate the features of split blocks, a one-layer Bidirectional LSTM is applied to capture the temporal information of all blocks. Then the average pooling of the last hidden state of LSTM is used as the document feature.

### Step 3. Label Retrieve
After get the representation of the entire document, we use the feature to retrieve its corresponding label. The label set is organized as a hierarchical structure, for example, the first level contains labels *A,B,C,D* and the second level contains *A-a, A-b, A-c, B-a, B-b, C-a, C-b, C-c, C-d...*. We introduce an 'unknown' token to represent the terminal of label extension, for example, *A-unknown* means the sample has only the first level label and we don't need to consider its second level label. 

To take the advantages of the label's name, which is meaningful, we don't use one-hot vector to represent each class. Instead, we use the inner-product similarity to get the logit of each class. Then we decode the label from the first level to higher level, and stop when we meet the 'unknown' token.

## Sub-directories

  - [approaches] contains the trainer class of training and predicting.
  - [dataloader] contains the implementation of data loader.
  - [networks] contains the implementation of HMC model.
  - [scripts] contains the training and predicting scripts.
  - [tools] contians some example materials of the class labels and test data.
  - [utils] contains some helper functions.

## Code Usage

To run the code, it's suggested to install the relevant python packages in a isolated conda environment. The `requirements.txt` is provided for a quick start.
- Run `pip install -r requirements.txt` after creating a conda environment to get necessary packages installed.

- run the following command to train the model.

  ```bash
  bash scripts/train.sh
  ```

- Get the prediction results. 

  ```bash
  bash scripts/pred.sh
  ```

Then the scripts will automatically save results to `./log` dir.

## License

HMC is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0.

[1]. He, Pengcheng, et al. "Deberta: Decoding-enhanced bert with disentangled attention." arXiv preprint arXiv:2006.03654 (2020).


