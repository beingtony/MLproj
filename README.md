# Twitter Sentiment Analysis Using Machine Learning

## Problem Statement

The goal of this project is to develop a sentiment analysis model for Twitter data, specifically focusing on classifying tweets into three sentiment categories: **Positive**, **Negative**, and **Neutral**. Sentiment analysis on Twitter data is essential for understanding public opinions, trends, and feedback.

## Introduction

In the digital era, social media platforms like Twitter play a vital role in shaping public opinion. Twitter is a hub of diverse emotions and opinions, which makes it an excellent source for analyzing the sentiment of individuals on various topics. By understanding how people feel about political events, products, or global issues, we can provide businesses, political leaders, and organizations with valuable insights for better decision-making.

This project leverages **machine learning techniques** to perform sentiment analysis on Twitter data, helping to understand the general mood or opinion expressed in the tweets.

## Dataset Information

The dataset used for this project is the **Twitter Sentiment Analysis Dataset**, which consists of labeled tweet data with the sentiment labeled as **positive**, **negative**, or **neutral**. This dataset helps in training machine learning models to recognize sentiment based on tweet content.

### Dataset Columns:
- `tweet`: Contains the text of the tweet.
- `sentiment`: Sentiment label of the tweet (`positive`, `negative`, `neutral`).

## Tools and Libraries Used

- **Natural Language Toolkit (NLTK)**: A Python library for processing and analyzing human language data. It is used for text preprocessing tasks such as tokenization, stop words removal, and lemmatization.
- **Pandas**: A powerful data manipulation and analysis library, used to load, clean, and process the dataset.
- **Scikit-learn**: A machine learning library used to implement various algorithms, including Multinomial Naive Bayes and Random Forest Classifier.
- **Matplotlib**: A library used for creating visualizations, such as graphs and charts.
- **WordCloud**: A visualization tool that generates word clouds from text data, representing the frequency of words.

## Approach

1. **Data Preprocessing**:
   - Tokenization: Break the tweet text into smaller units (words).
   - Stop Word Removal: Eliminate common words like 'the', 'is', 'and', etc., that do not contribute significant meaning.
   - Lemmatization: Convert words to their base or root form (e.g., "running" → "run").

2. **Vectorization**:
   - We use **CountVectorizer** from scikit-learn to convert the processed tweet text into a numerical format that can be fed into machine learning models.

3. **Sentiment Classification**:
   - **Multinomial Naive Bayes**: This algorithm is ideal for text classification tasks and works by estimating the likelihood of each class (positive, negative, or neutral) based on the word counts.
   - **Random Forest Classifier**: A powerful ensemble learning algorithm that builds multiple decision trees to classify data based on majority voting.

4. **Visualization**:
   - **Word Cloud**: Visualizes the most frequent words from positive, negative, and neutral tweets to understand common themes and topics.

## Results

- The **Random Forest Classifier** achieved **92%** accuracy in predicting tweet sentiment.
- The **Multinomial Naive Bayes** classifier performed well with an accuracy of **79%**.
- Word cloud visualizations revealed common words associated with each sentiment category, helping to identify popular themes in the tweets.

### Example Word Cloud Visualization:
The word clouds for positive, negative, and neutral tweets provide insight into what words are frequently associated with each sentiment. Positive tweets often contain words like "good", "happy", and "love", while negative tweets frequently mention words like "bad", "hate", and "disappointed".

## Conclusion

This project demonstrates how machine learning techniques can be used to perform sentiment analysis on Twitter data. By utilizing preprocessing techniques such as tokenization, stop word removal, and lemmatization, combined with powerful classifiers like Random Forest and Naive Bayes, we were able to accurately predict tweet sentiments with high precision.

Additionally, the word cloud visualizations give a deeper understanding of the emotions expressed in tweets and highlight popular terms in different sentiment categories.

## References

1. **Naive Bayes** — [scikit-learn 1.4.2 documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
2. **Random Forest Classifier** — [scikit-learn 1.4.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. **NLTK** — [Natural Language Toolkit Documentation](https://www.nltk.org/)
4. **WordCloud** — [WordCloud Documentation](https://github.com/amueller/word_cloud)

## Installation

To run this project, you will need to install the following Python libraries:

```bash
pip install nltk sklearn matplotlib wordcloud pandas
