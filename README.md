MACHINE LEARNING MINI PROJECT


PROBLEM STATEMENT :  The goal of this project is to employ machine learning techniques for sentiment analysis, specifically targeting tweets. Sentiment analysis involves determining whether a given tweet expresses a positive, negative, or neutral sentiment. 


INTRODUCTION: In today's digital age, the rapid dissemination of information through social media platforms like Twitter has become ubiquitous. Users rely on these platforms for news and updates, making them powerful tools for understanding public sentiment. However, alongside genuine content, social media is flooded with tweets expressing diverse emotions ranging from positivity to negativity. Analysing this sentiment can offer valuable insights into public perception, brand sentiment, and social trends. Moreover, as elections loom closer, the analysis of tweets becomes even more critical, providing a lens into public sentiment towards political figures and parties. By examining the sentiments expressed in tweets, we can gain a deeper understanding of people's opinions, preferences, and concerns, thus informing political strategies and public discourse.


DATASET INFORMATION :  
Twitter Sentiment Analysis Dataset


AIM : To develop a sentiment analysis model using machine learning techniques tailored for Twitter data.

The Natural Language Toolkit (NLTK) :  A platform used for building Python programs that work with human language data for application in statistical natural language processing (NLP). It contains text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning.

STOP WORDS : Stop words are common words like ‘the’, ‘and’, ‘I’, etc. that are very frequent in text, and so don’t convey insights into the specific topic of a document. We can remove these stop words from the text in a given corpus to clean up the data, and identify words that are more rare and potentially more relevant to what we’re interested in.
Text may contain stop words like ‘the’, ‘is’, ‘are’. Stop words can be filtered from the text to be processed.
There is no universal list of stop words in nlp research, however the nltk module contains a list of stop words.

WordNet Lemmatizer:  Often referred to simply as the WordNet Lemmatizer, is a tool commonly used in NLP tasks. Its primary function is to reduce words to their base or root form, known as the lemma. For example, the lemma of the word "running" is "run," and the lemma of "better" is "good."
The main purpose of lemmatization is to normalize words so that variations of the same word are treated as the same token, which can improve the accuracy and efficiency of machine learning algorithms.

Preprocess_text:
I/P  —>  text = "Hello, World! This is an example text with @symbols and\newlines."
O/P  —>  hello world this is an example text with symbols and newlines

PUNKET: The 'punkt' resource refers to the Punkt tokenizer models provided by NLTK. Tokenization is the process of breaking text into smaller units, typically words or sentences, which are called tokens. The Punkt tokenizer is a pre-trained tokenizer that can tokenize text into sentences or words. It is particularly useful for tasks such as text segmentation, where you need to divide a paragraph into individual sentences.


COUNTVECTORIZER:
Ex – food is good but food is not good

Tokenized version of the sentence:
["the", "food", "is", "good", "but", "food", "is", "not", "good"]

Vocabulary:
{'the': 0, 'food': 1, 'is': 2, 'good': 3, 'but': 4, 'not': 5}

Count matrix:
|    | the | food | is | good | but | not |
|----|-----|------|----|------|-----|-----|
| 0  | 1   | 1    | 2  | 2    | 1   | 0   |
| 1  | 1   | 2    | 2  | 1    | 1   | 1   |

CSR format representation:
(0, 0) 1
(0, 1) 1
(0, 2) 2
(0, 3) 2
(0, 4) 1
(1, 0) 1
(1, 1) 2
(1, 2) 2
(1, 3) 1
(1, 4) 1
(1, 5) 1







O/P:


MULTINOMIALNB:
y_train[0]: "Positive"                                                                       
y_train[1]: "Negative"
y_train[2]: "Positive"

X_train[0]: "i am good"
X_train[1]: "food is bad"
X_train[2]: "drink is cool"

(0, 0)    1  # "am" appears once in the first document
(0, 1)    1  # "good" appears once in the first document
(0, 2)    1  # "i" appears once in the first document
(1, 3)    1  # "bad" appears once in the second document
(1, 4)    1  # "food" appears once in the second document
(1, 5)    1  # "is" appears once in the second document
(2, 6)    1  # "cool" appears once in the third document
(2, 5)    1  # "drink" appears once in the third document
(2, 7)    1  # "is" appears once in the third document

Multinomial Naive Bayes classifies text by representing documents as word count vectors, assuming independence between words. It estimates class priors and conditional probabilities from training data. During classification, it calculates posterior probabilities for each class using Bayes' theorem and selects the class with the highest probability. It handles unseen words through techniques like Laplace smoothing. Overall, it's efficient for text classification due to its simplicity and effectiveness in dealing with high-dimensional feature spaces.



WORD CLOUD:
A word cloud is a visualization technique that displays the frequency of words in a text data set, where the size of each word represents its frequency. It's commonly used to quickly identify the most prominent terms in a corpus and their relative importance. Word clouds are visually appealing and useful for gaining insights into the main themes or topics within a collection of documents. 

USING RANDOM FOREST CLASSIFIER:


CONCLUSION:
Our sentiment analysis project delves into the world of Twitter discussions, utilising a mix of powerful computer techniques to understand the emotions behind tweets. We carefully prepared and organised the tweet data, then trained a special computer model called RandomForestClassifier. This model is very accurate, correctly predicting feelings in tweets 92% of the time!
Also we use Naive Bayes classifier which gives accuracy upto 79%

But our project goes beyond just guessing feelings. We also create colourful word cloud pictures to show which words are most common in positive, negative, or neutral tweets. These pictures help us see what topics people often talk about when they're feeling different emotions.

Overall, our project not only helps us understand feelings in tweets better but also gives us insights into digital sentiment trends. By using these techniques, we can make smarter decisions and better understand what's happening in the online world.


REFERENCES : 
1.9. Naive Bayes — scikit-learn 1.4.2 documentation
sklearn.ensemble.RandomForestClassifier — scikit-learn 1.4.2 documentation

Colab link:
final.ipynb




