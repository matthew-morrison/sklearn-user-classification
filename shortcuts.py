import sqlite3
import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter


# grab comments where the author has more than 1500 total comments.
select_more = """
SELECT * FROM messages
where messages.authorID IN
	(
	SELECT messages.authorID FROM messages
	GROUP BY messages.authorID
	HAVING COUNT(messages.authorID) > 1500
	)
"""

# connect to sql and dump into a dataframe

db = sqlite3.connect('dumpdb.sql')
c = db.cursor()
c.execute(select_more)

data = pd.DataFrame(c.fetchall(), columns=['messageID', 'authorID', 'channelID', 'serverID', 'timestamp', 'tts', 'content', 'embeds', 'mention_everyone', 'mentions_list', 'channel_mentions_list', 'role_mentions_list', 'pinned', 'reactions'])


# pare down to the necessary columns
data = data[['authorID', 'serverID', 'content']]
# convert to str so concatenation doesn't fail.
data['content'] = data['content'].astype(np.str)

#NEED TO FILTER OUT AUTHORS WHO DON'T TALK MUCH IN INDIVIDUAL SERVERS
grouped = data.groupby(['authorID', 'serverID'])
aggreg = grouped.agg({'content': ['sum', 'count']}) # counts as well as sums content rows.
filt1 = aggreg[aggreg['content']['count'] > 500] # filter out author/server combos with less than 500 comments.
# create boolean array of authors who are in multiple servers
aim = filt1.reset_index().groupby(['authorID'])['serverID'].count() > 1 
# create a dataframe which only contains authors who are in multiple servers
filt2 = filt1.reset_index()[filt1.reset_index()['authorID'].isin(aim[aim].reset_index()['authorID'])]

grouped2 = filt2.groupby(['authorID', 'serverID']).apply(lambda x: x.sum())

# create training group by grabbing the first server from each author
train = grouped2.groupby(['authorID']).head(1)
#train = grouped2.reset_index().groupby(['authorID'])['serverID'].head(1)



# create boolean array which represents authors for testing set.
boolarr = list(map(lambda item: item in train.index, grouped2.index.tolist())) #bool array for indices in train which are in group1

# create testing set which contains only authors/server/comments which do not exist in train set.
test = grouped2[[not i for i in boolarr]]

train.to_json("train.json")
test.to_json("test.json")

train.to_csv("train.csv")
test.to_csv("test.csv")

#====================================================
"""
# create boolean array of authors who are in multiple servers
aim = data.groupby(['authorID','serverID']).sum().reset_index().groupby(['authorID'])['serverID'].count() > 1 #aim = authors in multiple servers

# create a dataframe which only contains authors who are in multiple servers
filtered = data[data['authorID'].isin(aim[aim].reset_index()['authorID'])]

# create a dataframe which contains all an authors comments from a server concatenated together.
group1 = filtered.groupby(['authorID', 'serverID'])['content'].apply(lambda x: x.sum()) #author's comments per server

# create training group by grabbing the first server from each author
train = group1.reset_index().groupby(['authorID'])['serverID'].head(1)

# create boolean array which represents authors for testing set.
boolarr = list(map(lambda item: item in train.index, group1.reset_index().index.tolist())) #bool array for indices in train which are in group1

# create testing set which contains only authors/server/comments which do not exist in train set.
test = group1[[not i for i in boolarr]]

"""
