# Twitter Sentiment Analysis with Transformer Encoder

This project implements a **Transformer Encoder** for **Twitter comment sentiment analysis**, inspired by the [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) paper (Vaswani et al., NeurIPS 2017).  
On top of the encoder, a simple **Artificial Neural Network (ANN) classifier** is added to predict the sentiment of tweets.

---

## 📊 Dataset
- **Source:** [Twitter Entity Sentiment Analysis Dataset (Kaggle)](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)  
- Includes tweets with annotated **sentiment labels**: Positive, Negative, Neutral.

---

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Tokenization & padding
   - Vocabulary building
   - Train/validation split  

2. **Model Architecture**
   - Transformer Encoder (multi-head self-attention + feed-forward layers)
   - ANN classifier for sentiment classification  

3. **Training**
   - CrossEntropy loss  
   - Adam optimizer  
   - Accuracy as the evaluation metric  

4. **Evaluation**
   - Model performance on validation set  
   - Sample predictions on unseen tweets  

---

## 🚀 Tech Stack
- Python  
- PyTorch  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 📌 Results
- Successfully trained a Transformer Encoder for **multi-class sentiment classification**.  
- Demonstrated how transformers can be adapted for smaller NLP tasks beyond LLMs.  

---

## 🔮 Future Work
- Integrate pre-trained embeddings (Word2Vec / GloVe / BERT embeddings)  
- Hyperparameter tuning for better accuracy  
- Experiment with full Transformer (Encoder-Decoder) or pretrained LLMs  

---
