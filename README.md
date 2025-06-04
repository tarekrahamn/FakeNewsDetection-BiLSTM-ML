# 📰 Fake News Detection using Machine Learning and Deep Learning

This project focuses on detecting fake news using multiple machine learning classifiers and a BiLSTM (Bidirectional LSTM) deep learning model. It uses the **True.csv** and **Fake.csv** datasets for training and evaluation.

## 📁 Dataset

- **True.csv**: Contains real news articles.
- **Fake.csv**: Contains fake news articles.
- Both datasets have been labeled (`1` for real, `0` for fake).
-  **Dataset Link ** : 

## 📌 Features

- Preprocessing of news text (cleaning HTML tags, URLs, numbers, etc.).
- Classical ML models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Passive Aggressive Classifier.
- Deep Learning model: Bidirectional LSTM using Keras/TensorFlow.
- TF-IDF Vectorization for classical models.
- Tokenizer + Padding for deep learning.
- Manual prediction for user-input news.

## 🛠️ Setup

1. Mount Google Drive to access dataset files:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2. Install required packages:
    ```bash
    !pip install -q scikit-learn tensorflow tqdm
    ```
3. Load datasets:
    ```python
    true = pd.read_csv('/content/drive/MyDrive/True.csv')
    fake = pd.read_csv('/content/drive/MyDrive/Fake.csv')
    ```

## 🧪 Models and Accuracy

| Model                            | Accuracy     |
|----------------------------------|--------------|
| Logistic Regression              | 98.83%       |
| Decision Tree                    | 99.49%       |
| Random Forest                    | 99.78%       |
| Gradient Boosting                | 99.48%       |
| Passive Aggressive Classifier    | 99.56%       |
| **BiLSTM (Deep Learning)**       | **99.89%**   |

### 🔍 Classification Report Example (Logistic Regression)
          precision    recall  f1-score   support

       0       0.99      0.99      0.99      7047
       1       0.99      0.99      0.99      6423

accuracy                           0.99     13470
macro avg 0.99 0.99 0.99 13470
weighted avg 0.99 0.99 0.99 13470


## 🔎 Manual Testing

You can test any custom news article:

```python
print(manual_testing("The president met with the UN delegation to discuss climate change."))
📊 Deep Learning Model (BiLSTM)
Tokenizer: 10,000 most frequent words

Padding: 300 max length

Architecture:

Embedding Layer

Bidirectional LSTM

Dropout

Dense layers with ReLU and Sigmoid

Trained with:

Epochs: 20

Batch Size: 64

Optimizer: Adam

📌 Future Improvements
Implement attention mechanism for the BiLSTM.

Build a web UI using Flask or Streamlit.

Add support for multi-language news detection.
## 🛠️ How to Run

### 🖥️ Option 1: Google Colab
1. Open the notebook in Colab: `[Your Colab Link]`
2. Upload the `True.csv` and `Fake.csv` files to your Google Drive
3. Update the file path accordingly
4. Run all cells to train and evaluate models

### 💻 Option 2: Local Jupyter Notebook
1. Clone the repo:
   ```bash
   git clone https://github.com/tarekrahamn/fake-news-detection.git
   cd fake-news-detection

pip install -r requirements.txt

