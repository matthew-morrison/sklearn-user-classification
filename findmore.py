import sqlite3
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# find users who are on multiple servers
# also cleans datum

select_stmt = """select * from messages"""

insert_stmt2 = """
INSERT INTO {0} 
SELECT messages.* FROM cdb.messages 
INNER JOIN cdb.authors on cdb.messages.authorID = cdb.authors.authorID 
WHERE cdb.messages.authorID in 
	(SELECT cdb.messages.authorID 
	FROM cdb.messages 
	GROUP BY cdb.messages.authorID 
	HAVING COUNT(cdb.messages.authorID) > 1000) 
AND cdb.messages.embeds == 0 
GROUP BY cdb.messages.timestamp
"""

tb_msg = '''CREATE TABLE IF NOT EXISTS {0}(
            messageID integer primary key,
            authorID integer,
            channelID integer,
            serverID integer,
            timestamp datetime,
            tts boolean,
            content string,
            embeds text,
            mention_everyone boolean,
            mentions_list text, 
            channel_mentions_list text, 
            role_mentions_list text,
            pinned boolean,
            reactions text
            )'''


schema_stmt = """SELECT sql FROM cdb.sqlite_master WHERE type='table' AND name='messages'"""


db = sqlite3.connect('superclean.sql') # create superclean
c = db.cursor()
c.execute(schema_stmt)





c.execute(select_stmt)
servers = c.execute("select * from servers").fetchall()


multiserver = sqlite3.connect('multiserver.sql')
for sid in servers:
    print(sid[0])
    create_table = tb_msg.format(sid[0])
    c.execute(create_table)
    c.execute('insert into '+sid[0]+' values where yada yada')



data = pd.DataFrame(c.fetchall(), columns=['messageID', 'authorID', 'channelID', 'serverID', 'timestamp', 'tts', 'content', 'embeds', 'mention_everyone', 'mentions_list', 'channel_mentions_list', 'role_mentions_list', 'pinned', 'reactions'])

#data = pd.DataFrame(c.fetchall(), columns=['content', 'displayname', 'id'])
data['content'] = data['content'].astype(np.str) # fix wrong inferred type.
data['messageID'] = data['messageID'].astype(np.int64)
data['channelID'] = data['channelID'].astype(np.int64)
data['serverID'] = data['serverID'].astype(np.int64)
print(data)
