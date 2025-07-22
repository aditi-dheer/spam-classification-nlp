# Spam Email Classification with Word Embeddings  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-Text%20Processing-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Word2Vec-339933?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
</p>  

## Overview  
This project performs **spam email classification** using **Natural Language Processing (NLP)** and **Word2Vec embeddings**.  

The pipeline:  
- Preprocesses emails into clean tokenized text  
- Learns **dense vector embeddings** for each word with **Word2Vec**  
- Averages embeddings to create **email-level feature vectors**  
- Trains a **Logistic Regression classifier**  
- Evaluates model performance using **ROC-AUC**  

## Key Points  
- **Loads email dataset** from `data/spamDataset.csv`  
- **Cleans and tokenizes text** with `gensim.utils.simple_preprocess`  
- **Learns Word2Vec embeddings** to represent each word as a vector  
- **Generates feature vectors** by averaging embeddings per email  
- **Trains Logistic Regression** on the averaged vectors  
- **Evaluates performance** with ROC-AUC score  

## Evaluation  

| Metric   |      Score      |
|----------|-----------------|
| ROC-AUC  | ~0.99           |

---

## Installation and Usage  

Clone the repository and install the required dependencies:  

```bash
git clone https://github.com/aditi-dheer/spam-classification-nlp.git
cd spam-classification-nlp
pip install gensim scikit-learn pandas numpy seaborn matplotlib
```

## Usage

Run the spam classification pipeline:

```bash
jupyter notebook classifying_spam_nlp.ipynb
```

It will output the **ROC-AUC score** on the test set.

---

## Project Structure

- **spamDataset.csv** – Dataset of book reviews  
- **classifying_spam_nlp.ipynb** – Main script for training and evaluation  
- **README.md** – Project documentation  

