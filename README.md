# NLP-Based Sentiment Analysis of Product Reviews

## Project Overview

This project performs sentiment analysis on customer product reviews using Natural Language Processing (NLP) and Machine Learning.

The reviews are classified into two categories:

- Positive
- Negative

The project uses TF-IDF for text feature extraction and Logistic Regression for sentiment classification.

## Dataset

The project uses the **Women's E-Commerce Clothing Reviews** dataset from Kaggle.

The dataset contains customer reviews, ratings, product information, and recommendation information.

For sentiment classification:

- Ratings 1 and 2 → Negative
- Ratings 4 and 5 → Positive
- Rating 3 → Removed as neutral

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Methodology

The project follows these steps:

1. Load the dataset
2. Explore the dataset
3. Handle missing review text
4. Create Positive and Negative sentiment labels
5. Preprocess the review text
6. Split the dataset into training and testing data
7. Convert text into numerical features using TF-IDF
8. Train a Logistic Regression model
9. Evaluate the model
10. Predict sentiment for new reviews

## Text Preprocessing

The following preprocessing techniques are applied:

- Convert text to lowercase
- Remove HTML tags
- Remove URLs
- Remove punctuation
- Remove numbers
- Remove extra spaces

## Model

**Algorithm:** Logistic Regression

**Feature Extraction:** TF-IDF

The dataset is divided into:

- 80% training data
- 20% testing data

The TF-IDF vectorizer uses up to 10,000 features and includes unigrams and bigrams.


## Results

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The Logistic Regression model achieved the following results:

| Metric | Score |
|--------|-------|
| Accuracy | 92.43% |
| Precision | 92.81% |
| Recall | 99.08% |
| F1-Score | 95.84% |

The model was also tested with new product reviews and successfully predicted Positive and Negative sentiments.

## Example

**Input:**

> I absolutely love this dress. The material is excellent and it fits perfectly.

**Prediction:**

Positive

**Input:**

> I am very disappointed with this product. The quality is poor and I do not like it.

**Prediction:**

Negative

## Project Files

- `NLP_Sentiment_Analysis_Product_Reviews.ipynb` – Complete Google Colab notebook
- `requirements.txt` – Required Python libraries

## How to Run

1. Open the notebook in Google Colab.
2. Upload the Women's E-Commerce Clothing Reviews CSV dataset.
3. Run the notebook cells in order.
4. The model will train and evaluate the sentiment classification.
5. New product reviews can be entered for sentiment prediction.

## Conclusion

This project demonstrates how NLP and machine learning can be used to analyze customer opinions from product reviews. TF-IDF effectively converts review text into numerical features, while Logistic Regression is used to classify the reviews into Positive and Negative sentiments.
