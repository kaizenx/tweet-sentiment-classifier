# Twitter Sentiment Analysis

Data found here https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweetsaHere’s 

---

# Twitter Sentiment Analysis using LSTM

## Overview
This project aims to classify tweets into three sentiment categories—Negative, Neutral, and Positive—using an LSTM (Long Short-Term Memory) model, a type of Recurrent Neural Network (RNN). The model processes a dataset of tweets, tokenizes and pads sequences, and utilizes an LSTM architecture to capture sequential patterns and context, essential for accurate sentiment analysis.

## Project Structure
- **sentiment.ipynb**: Jupyter Notebook containing the full implementation, including data preprocessing, model building, training, evaluation, and results analysis.
- **data/**: Directory containing the tweet dataset used for training and validation.
- **models/**: Directory to save the trained models and checkpoints (if applicable).

## Features
- **Data Preprocessing**: Tokenization, padding, and handling of informal or noisy text (e.g., null rows and float values).
- **LSTM Model**: Sequential model with embedding, LSTM layers, and dropout for regularization, optimized for tweet sentiment classification.
- **Hyperparameter Tuning**: Includes tuning of embedding dimensions, dropout rates, LSTM units, and more.
- **Evaluation Metrics**: Uses accuracy, confusion matrix, and classification report (precision, recall, F1-score) to evaluate model performance.
- **Visualization**: Plots of sentiment distribution, tweet length distribution, loss, and accuracy trends during training.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis-lstm.git
   cd twitter-sentiment-analysis-lstm
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or place your tweet dataset in the `data/` directory.

## Usage
1. **Run the Notebook**:
   - Open `sentiment.ipynb` in Jupyter Notebook or JupyterLab.
   - Follow the steps in the notebook to preprocess the data, build and train the model, and evaluate the results.

2. **Customize Hyperparameters**:
   - Modify parameters like `embedding_dim`, `lstm_units`, `dropout` rates, and `learning_rate` directly in the notebook to experiment with different configurations.

3. **Save and Load Models**:
   - The notebook includes sections for saving the trained model and loading it for future predictions or further analysis.

## Results
- The model achieves moderate accuracy, with a macro-average F1-score around 0.66, indicating balanced performance across sentiment classes.
- Evaluation highlights some challenges with Positive sentiment classification, suggesting areas for future improvement such as bidirectional LSTMs and attention mechanisms.

## Challenges
- **Data Quality**: Tweets often contain informal language, null values, and noisy data, requiring extensive preprocessing.
- **Sentiment Nuances**: Differentiating between similar sentiments, especially Positive and Disaster, due to overlapping language and context.

## Future Improvements
- Implement bidirectional LSTMs and attention mechanisms for improved context capturing.
- Consider systematic hyperparameter optimization (e.g., Grid Search or Bayesian Optimization).
- Explore data augmentation techniques to enhance model robustness.

