README.md

## Requirements

python3
sklearn
pandas



## What do all our files do? What do I run?

Important files:
shortcuts.py - this python file finds users on multiple servers with more than 1500 messages and concatenates all their messages together before printing to train.csv and test.csv.
rungrouped.py - this python file read the train.csv and test.csv files and runs several ML algorithms on it.


Other files:
clean.py takes the raw database files and filters out unwanted data, writing to a new db file.
find.py finds and prints out users who are on multiple servers. Used for debugging.
findmore.py does the same finding as find.py but additionally creates a new clean db based on the findings.
superfilter.py - a WIP python file which never gave good results


## What is actually going on here?

This is a project to answer a question of mine that I asked myself after taking an introductory Data Science course. This is during late 2017 early 2018.

If you had a collection of a user's messages on one forum, could you detect that user based on their messages on a different forum?

More specifically, I compared user messages on one public discord server against user messages on different public discord servers. I attempted to see if any basic ML algorithm could be trained to detect the user on a differnt server.


## Results

I used several different models without hyperparameterization. I got a surprisingly accurate result with sgdc_tfidf. (Essentially a State Vector Machine.)

              method     score
9           knn_hash  0.243243
11         knn_count  0.270270
7         sgdc_count  0.351351
3           nb_count  0.378378
2     nb_count_tfidf  0.405405
5          sgdc_hash  0.459459
10   knn_count_tfidf  0.513514
0      nb_hash_tfidf  0.621622
8     knn_hash_tfidf  0.621622
1            nb_hash  0.702703
4    sgdc_hash_tfidf  0.729730
6   sgdc_count_tfidf  0.729730

