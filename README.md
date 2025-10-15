# Bitcoin Price Prediction using Twitter Sentiment (BERT + VADER + RF + LR + LSTM)

## Overview
This project explores how **social media sentiment** influences **Bitcoin price movements**.  
By analyzing thousands of **Bitcoin-related tweets**, we extract **sentiment signals** using both **VADER** and **BERT**, then train **three models** — **Random Forest**, **Logistic Regression**, and **LSTM** — to predict whether Bitcoin’s price will rise or fall.

This project combines **Natural Language Processing (NLP)**, **Machine Learning**, and **Deep Learning** to uncover the connection between **public opinion** and **cryptocurrency market behavior**.

---

##  Objectives
- Clean and preprocess Bitcoin-related tweets 🧹  
- Perform **sentiment analysis** using **VADER** and **BERT**  
- Merge sentiment data with **Bitcoin price trends**  
- Train three models:
  -  Random Forest  
  -  Logistic Regression  
  -  LSTM (Long Short-Term Memory)
- Compare model performances and visualize results  

---

##  Dataset Description

###  1. Bitcoin Tweets Dataset
- File: `Bitcoin_tweets.csv`
- Source: [Bitcoin_tweets](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)
- Contains tweets related to Bitcoin.
- Key columns:
  - `text` — tweet content  
  - `date` — tweet timestamp  

###  2. Bitcoin Price Dataset
- File: `BTC-USD.csv`
- Source: [Bitcoin Price Dataset](https://www.kaggle.com/datasets/pavan9065/bitcoin-price-history)
- Historical Bitcoin prices with columns:
  - `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

---

##  Workflow

###  Step 1: Data Preprocessing
- Load and clean both datasets using **pandas**
- Remove URLs, hashtags, mentions, emojis, and special characters
- Merge tweets with Bitcoin price data by date
- Handle missing values and normalize numeric features

###  Step 2: Sentiment Extraction
#### 🔹 VADER Sentiment Analyzer
- A **rule-based sentiment analyzer** optimized for social media.
- Generates:
  - `pos` (positive), `neu` (neutral), `neg` (negative), and `compound` (overall score between -1 to +1)

#### 🔹 BERT Embeddings
- **BERT (Bidirectional Encoder Representations from Transformers)** extracts **deep contextual representations** of tweets.
- Converts each tweet into a numerical vector capturing semantic meaning.

### Step 3: Feature Engineering
- Combine **VADER scores**, **BERT embeddings**, and **Bitcoin price features**
- Create a **target variable** indicating if Bitcoin’s price increased (1) or decreased (0) the next day
- Split data into training and test sets

###  Step 4: Model Training

| Model | Type | Description | Strength |
|--------|------|--------------|-----------|
|  **Random Forest** | Ensemble ML | Combines many decision trees for robust predictions | Handles complex data, prevents overfitting |
|  **Logistic Regression** | Classical ML | Simple and interpretable linear model | Fast baseline for comparison |
|  **LSTM (Long Short-Term Memory)** | Deep Learning (RNN) | Learns time-based dependencies | Great for sequential data (tweets/time-series) |

###  Step 5: Evaluation & Visualization
- Evaluate models using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Plot:
  - Average daily sentiment vs. Bitcoin closing price
  - Model comparison charts (RF, LR, LSTM)
  - Confusion matrices

---

##  Libraries Used

| Library | Purpose |
|----------|----------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualization |
| `vaderSentiment` | VADER sentiment analysis |
| `transformers` | BERT tokenizer & embeddings |
| `scikit-learn` | ML models (RF, LR) & metrics |
| `tensorflow / keras` | Deep learning (LSTM implementation) |
| `tqdm` | Progress tracking |

---

##  Results & Insights
- Tweets with **positive sentiment** generally align with **Bitcoin price increases** 📈  
- **BERT-based features** provided deeper context understanding  
- **Model comparison summary:**

| Model | Accuracy | Observation |
|--------|-----------|--------------|
|  Random Forest | ⭐ Highest | Balanced and stable results |
|  Logistic Regression | Good | Fast and interpretable baseline |
|  LSTM | High (on time-sequence data) | Captures temporal sentiment patterns |

---

##  How to Run the Project

```bash

git clone https://github.com/<your-username>/Bitcoin-Sentiment-Analysis.git
cd Bitcoin-Sentiment-Analysis

🧱 Step 2: Install Dependencies

Ensure you have Python 3.8+ installed. Then run:

pip install -r requirements.txt


If you don’t have a requirements.txt, install manually:

pip install pandas numpy matplotlib scikit-learn tensorflow keras transformers vaderSentiment tqdm

📂 Step 3: Place Datasets

Ensure the following files are in the same folder as the notebook:

Bitcoin_tweets.csv
BTC-USD.csv

▶️ Step 4: Run the Notebook

Open and execute the Jupyter Notebook:

jupyter notebook Team21_ML_Assignment.ipynb


Or use Google Colab:

Upload the notebook and both datasets to your Colab environment and make sure the naming conventions are correct. (⚠️ NOTE: The link to the datasets is given in the DATASET DESCRIPTION section of the README.)

Run all cells sequentially.

📊 Step 5: View Outputs

Model performance metrics will be displayed in output cells.

Graphs will show sentiment vs. price trends and model comparisons.

🪙 Step 6 — Bitcoin Tweet Sentiment Analyzer (Gradio)

This step creates a simple web app using Gradio and VADER Sentiment Analysis to quickly analyze the sentiment of any Bitcoin-related tweet or message.

This mini-app lets you enter a Bitcoin tweet or headline (like “Bitcoin is skyrocketing!! This is amazing!!”) and instantly get a sentiment label — Positive, Negative, or Neutral — along with its numeric VADER compound score. 
(⚠️ NOTE: Please open the link generated in the cell to run the GUI better, DONOT run it in the code cell.)

🔮 Future Work

Integrate real-time Twitter streaming via Twitter API (X)

Explore Transformer-based classifiers (e.g., BERT fine-tuning)

Add macro-economic indicators for more robust forecasting

Build a Streamlit dashboard for real-time visualization

CONCLUSION:
This project demonstrates that social media sentiment can serve as a valuable indicator of cryptocurrency price movements.
By combining VADER, BERT, and diverse models (RF, LR, LSTM), we blend classical and deep learning approaches to bridge human emotion and financial analytics.
