import sqlite3
import numpy as np
import pandas as pd
#import math
#import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter

select into (new table) (servername) select * from bigtable where #message is big.


select_stmt = "select * from messages"

select_more = """
SELECT * FROM messages
where messages.authorID IN
	(
	SELECT messages.authorID FROM messages
	GROUP BY messages.authorID
	HAVING COUNT(messages.authorID) > 1500
	)
"""


select_more_old = """
SELECT * FROM messages
GROUP BY messages.authorID
HAVING COUNT(messages.authorID) > 1000
"""

db = sqlite3.connect('dumpdb.sql')
c = db.cursor()
c.execute(select_more)

data = pd.DataFrame(c.fetchall(), columns=['messageID', 'authorID', 'channelID', 'serverID', 'timestamp', 'tts', 'content', 'embeds', 'mention_everyone', 'mentions_list', 'channel_mentions_list', 'role_mentions_list', 'pinned', 'reactions'])


#data = pd.DataFrame(yada) #straight from the dump table

servers = data['serverID'].unique() # list of servers


sframes = [] #contains an array of messages unique to servers
for s in servers:
   # s = data[data['serverID'] == s]
    sframes.append(data[data['serverID'] == s])
    
    
#for each "server table" we just need a unique list of author names.
#we also need to filter out users with less than a certain number of messages.



#this line was in here but it doesn't work   
#pivot0 = sframes[0].pivot(index='authorID')

fframes = [] # contains sorted/filtered array of messages unique to servers
for f in sframes:
   f1 = f.groupby('authorID').count()
   fframes.append(f1[f1['reactions'] > 500]) # filter out authors with less than 500 messages
   
#authors = fframes[0].index.values # returns array


alist = [] # contains raw 2dim arr of authors from each server
for f in fframes:
    alist.append(f.index.values)


fulllist = list(chain.from_iterable(alist)) #1d array of all authors
cnt = Counter(fulllist)
uniques = [k for k, v in cnt.items() if v > 1] #unique array of authors who are in more than one server

# from this point we need two different sets.
# set1 is the training set: the comments of a unique list of authors split by server.
# set2 is the testing set: the comments of a non-unique list of authors per server not found in set1.


setx = pd.DataFrame(columns=[0,1])
sety = pd.DataFrame(columns=[0,1])


# there is likely an error for iterating over items more than once.
for idx in range(len(servers)):
    for au in fulllist:
        if au in uniques: #if author is in more than one server
            if au not in setx[0].values:
                setx = setx.append([[au, servers[idx]]])
                print("adding" + str(au) + " to setx")
            else:
                sety = sety.append([[au, servers[idx]]])            
                print("adding" + str(au) + " to sety")


for s in alist:
    for a in s:
        if a in uniques:
            if a not in setx[0].values:
                print('for x adding ' + str(a) + ' in ' + str(servers[c1]))
                setx = setx.append([[a, servers[c1]]])
            else:
                print('for y adding ' + str(a) + ' in ' + str(servers[c1]))
                sety = sety.append([[a, servers[c1]]])
            c3 = c3 + 1
        else:
            c2 = c2 + 1 
    print('this was the ' + str(c1) + ' iteration')
    c1 = c1 + 1


# Now we need to add the grouped comments to these
# but we need to pare down the data first... otherwise it will take forever to groupby
filtered_data = data[data['authorID'].isin(uniques)]

# in my testing I found this brought us to 372k values compared to the original 8,427k... a considerable reduction.
filtered_data['content'] = filtered_data['content'].astype(np.str)
grouped_data = filtered_data.groupby(['authorID','serverID'])['content'].apply(lambda x: x.sum())
gp = gp.reset_index(level=[0,1]) # undo the multiindex from the previous function.

# there error might be explained by the filter on user+server combos of 500 messages. On one side they are filtered out whereas the filtered_data table only checks to see if the author itself is in the uniques list.



#why is the left side filter not showing people in 3 and 4 servers?

setx.isin(gp.index.get_level_values(0))

# FILTER OUT PEOPLE WHO AREN'T IN MORE THAN ONE SERVER BY FINDING data.uniques() AND USING ~data.isin(uniques) TO FIND PEOPLE WHO AREN'T UNIQUE

# instead of all that, can we get there using just the grouped data?
# you can easily "see" which index0 items have only 1 index1 for them

fd = data[data['authorID'].isin(data['authorID'].unique())] #

#Check out the actual multiindex levels...
# it has nice structure...
# levels=[[num,num,num],
#	[num,num,num],
#	[num,num,num,...
	
	
#	]]
# so just iterate, for array in levels... take first one.


# or... do two groupings.
group2 = data.groupby(['authorID','serverID'])['content'].apply(lambda x: x.sum())
group2.reset_index()
group3 = group2.groupby(['authorID'])['serverID'].count()
group4 = group3[group3 > 1]

# only has authorID as index and comments as column


group2 = group2.reset_index(drop_levels=False)
group3 = group2.groupby(['authorID', 'serverID']).count()????

filtered = data['authorID'].isin(data.groupby(['authorID','serverID']).sum().reset_index().groupby(['authorID'])['serverID'].count() > 1)

authors_in_multi = data.groupby(['authorID','serverID']).sum().reset_index().groupby(['authorID'])['serverID'].count() > 1
authors_in_multi[authors_in_multi]
filtered = data.isin(authors_in_multi[authors_in_multi])


data = data['authorID'].isin(data.groupby(['authorID'])['serverID'].count() > 1)




