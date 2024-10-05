
# MedRoute.AI

Our model predicts the requirement for treatment of a patient by analyzing his/her symptoms , it sends the predicted treatment and symptoms and the required doctor names(doctors info is given as input to the model) to all the hospitals within the accessible distance, the hospital sends if the facility and gives the reason for its analysis and the required doctor is vacant now, if yes then the hospital sends back an accept signal , the ambulance must navigate to the nearest hospital from the n number of hospital who gave the accept signal , implement pretrained bert model on this

## Table of Contents

- [About the Project](#about-the-project)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Modeling and Algorithms](#modeling-and-algorithms)
- [Results](#results)


## About the Project

Explain the purpose of the project. What problem does it solve or investigate? What methods or techniques were applied? Include a brief summary of the notebook.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Cavin-chandran/Alt-F4
   ```
2. Install the required dependencies:
    import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


## Usage

Dataset is uploaded

### Example for Google Colab:
- Open the notebook in Colab, upload your dataset, and execute the code cells.
- Modify paths in the notebook to fit your file structure.

## Dataset

- Source of the dataset (if applicable).
- Brief description of the dataset (number of features, types of data, etc.).
- How to load the dataset.

## Modeling and Algorithms

1. **BERT (Bidirectional Encoder Representations from Transformers)**:
   - **Description**: BERT is a transformer-based model designed to pre-train deep bidirectional representations from text. It is particularly powerful for natural language processing (NLP) tasks such as classification, named entity recognition, and question answering.
   - **Use in the Project**: BERT was used for **sequence classification**, specifically in a healthcare-related task to predict patient conditions based on symptoms, medical history, and other information. The model learns to understand the context and relationships within sentences to make accurate predictions.
   - **Pre-processing Steps**:
     - **Tokenization**: The input text (e.g., symptoms and medical history) was tokenized using the BERT tokenizer, which splits the text into tokens and adds special tokens (e.g., `[CLS]`, `[SEP]`) required by the BERT model.
     - **Padding and Truncation**: Since BERT expects fixed-length inputs, the text sequences were padded to a maximum length (256 tokens) or truncated if they exceeded the limit.

2. **T5 (Text-To-Text Transfer Transformer)**:
   - **Description**: T5 is a transformer model where every NLP problem is framed as a text-to-text task. For instance, classification is framed as generating a label from input text, and summarization involves generating a summary from a larger text.
   - **Use in the Project**: T5 was used to **generate medical reports** based on the predicted condition (from BERT) and patient information. The model was fine-tuned to produce concise, professional, and medically accurate text based on the provided inputs.
   - **Pre-processing Steps**:
     - **Text Preprocessing**: The input to T5 was structured as “**Condition: X. Symptoms: Y.**”, ensuring the input format was consistent and informative for the model to generate meaningful reports.
     - **Tokenization**: Similar to BERT, T5 uses its tokenizer to convert text to tokenized input, including padding/truncating sequences to ensure a fixed length.



### Machine Learning Models Used:
Bert and T5



### Example:
- **Modeling Techniques:** Bert,T5
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.

## Results

The symptoms are used as input and condition is found as target , ambulance travels to the best option of hospital successfully
FastAPI has been integrated
Flask backend has been implemnted

### Example:
- The model achieved an accuracy of 92% with a precision of 0.88 on the test dataset.



