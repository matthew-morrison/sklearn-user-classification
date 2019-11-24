# README

## Table of Contents  
+ [Requirements](#Requirements)
+ [What Do All Our Files Do? What Do I Run?](#What_Do_All_Our_Files_Do?_What_Do_I_Run?)
+ [What is Going On Here?](#What_is_Going_On_Here?)
+ [Results](#Results)

<a name="Requirements"/>

## Requirements

The following things need to be installed to run this project: 
- [Python3](https://www.python.org/downloads/)
- [sklearn](https://scikit-learn.org/stable/install.html)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)

<a name="What_Do_All_Our_Files_Do?_What_Do_I_Run?"/>

## What Do All Our Files Do? What Do I Run?

#### Important files: ####
- [shortcuts.py](shortcuts.py) finds users on multiple servers with more than 1500 messages. It concatenates all their messages together before printing to csv files called train.csv and test.csv.
- [rungrouped.py](rungrouped.py) reads the train.csv and test.csv files and runs several Machine Learning algorithms on the data set from the train.csv and test.csv.

#### Other files: ####
- [clean.py](clean.py) takes the raw database files and filters out unwanted data, writing to a new db file.
- [find.py](find.py) finds and prints out users who are on multiple servers. This python file is used for debugging.
- [findmore.py](findmore.py) does the same finding as find.py but creates a new clean database based on the findings.
- [superfilter.py](superfilter.py) is a "work in progress" python file that did not produce any good results.

<a name="What_is_Going_On_Here?"/>

## What is Going On Here?

After taking an introductory Data Science course in early of 2018, I wondered if a collection of user's messages on one forum could help identify that same user based on their messages in a different forum. 

More specifically, I compared user messages on one public discord server against another user messages on a different public discord server. I wanted to see if any basic Machine Learning algorithm could be trained to identify that same user on a different server.

<a name="Results"/>

## Results

I used different models without hyperparameterization. Surprisingly, I got an accurate result with [sgdc_tfidf](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). The sgdc_tfidf classification is essentially a [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html).

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