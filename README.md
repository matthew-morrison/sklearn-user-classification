# README

## Table of Contents  
+ [Preamble](#Preamble)
+ [Requirements](#Requirements)
+ [What Do All Our Files Do? What Do I Run?](#What_Do_All_Our_Files_Do?_What_Do_I_Run?)
+ [Conclusion](#Conclusion)

<a name="Preamble"/>

## Preamble

After taking an introductory Data Science course in late of 2017, I wondered if a collection of user's messages on one forum could help identify that same user based on their messages in a different forum. 

Specifically, this projects compares user messages from one public discord server against another user messages on a different public discord server. The goal of this project is to see if any basic Machine Learning algorithm could be trained to identify that same user on a different server.  

<a name="Requirements"/>

## Requirements

The following things need to be installed to run this project: 
- [Python3](https://www.python.org/downloads/)
- [sklearn](https://scikit-learn.org/stable/install.html)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)

<a name="What_Do_All_Our_Files_Do?_What_Do_I_Run?"/>

## What Do All Our Files Do? What Do I Run?

#### Important files: ####
- [shortcuts.py](shortcuts.py) finds users on multiple servers with more than 1500 messages. The messages are concatenated before being outputted to csv files called train.csv and test.csv.
- [rungrouped.py](rungrouped.py) reads the train.csv and test.csv files and runs several Machine Learning algorithms on the data set from the train.csv and test.csv.

#### Other files: ####
- [clean.py](clean.py) takes the raw database files and filters out unwanted data, writing to a new db file.
- [find.py](find.py) finds and prints out users who are on multiple servers. This python file is used for debugging.
- [findmore.py](findmore.py) does the same finding as find.py but creates a new clean database based on the findings.
- [superfilter.py](superfilter.py) is a "work in progress" python file that did not produce any good results.

<a name="Conclusion"/>

## Conclusion

### Summary

The user messages were aggregated from several discord servers and then cleaned. The data was then split into a training and test data set. Then, the Machine Learning algorithms from SKLearn were used to create models for predictions. The prediction results were generated by using different Machine Learning models without [hyperparameterization](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)). 

### Data Results

**method**|**score**
:-----:|:-----:
knn\_hash|0.243243
knn\_count|0.270270
sgdc\_count|0.351351
nb\_count|0.378378
nb\_count\_tfidf|0.405405
sgdc\_hash|0.459459
knn\_count\_tfidf|0.513514
nb\_hash\_tfidf|0.621622
knn\_hash\_tfidf|0.621622
nb\_hash|0.702703
sgdc\_hash\_tfidf|0.729730
sgdc\_count\_tfidf|0.729730

### Explanations

The most accurate of those results were produced by using the [Hashing Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) or the [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) with the [Term Frequency – Inverse Document Frequency (TF-IDF) Transformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html). Text feature extraction can be done with the Hashing Vectorizer or Count Vectorizer. They both utilize different methods to extract features from a text. The Hashing Vectorizer transforms a data set into matrix of hashed token occurrences for terms. On the other hand, the Count Vectorizer transforms a data set to a matrix of token counts. The resulting matrices can then be used on the TF-ID transformer. The TF-ID Transformer transforms the matrix into a weighting scheme that scales down frequent occurrences. Frequent occurrences could give bias towards features that are less important to features that occur less. The TF-IDF classification is essentially a [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html).
