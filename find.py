import sqlite3
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# find users who are on multiple servers

select_stmt = """select * from messages"""

stmt2 = """
select o,
s1.serverID as "81384788765712384",
s2.serverID as "127338035850248193",
s3.serverID as "222078108977594368",
s4.serverID as "228761314644852736",
s5.serverID as "244187921228234762",
s6.serverID as "266818573337362433",
s7.serverID as "270279881294741514",
s8.serverID as "286168815585198080",
s9.serverID as "286429969104764928",
s10.serverID as "297744523910578176",
s11.serverID as "298460124341862401",
s12.serverID as "316759222207643649",
s13.serverID as "318970898050973697"

from (
	select authors.authorID as o from authors 
	group by authors.authorID
	having count(authors.authorID) > 1
	)
	
left outer join authors s1 on s1.authorID = o and s1.serverID = 81384788765712384
left outer join authors s2 on s2.authorID = o and s2.serverID = 127338035850248193
left outer join authors s3 on s3.authorID = o and s3.serverID = 222078108977594368
left outer join authors s4 on s4.authorID = o and s4.serverID = 228761314644852736
left outer join authors s5 on s5.authorID = o and s5.serverID = 244187921228234762
left outer join authors s6 on s6.authorID = o and s6.serverID = 266818573337362433
left outer join authors s7 on s7.authorID = o and s7.serverID = 270279881294741514
left outer join authors s8 on s8.authorID = o and s8.serverID = 286168815585198080
left outer join authors s9 on s9.authorID = o and s9.serverID = 286429969104764928
left outer join authors s10 on s10.authorID = o and s10.serverID = 297744523910578176
left outer join authors s11 on s11.authorID = o and s11.serverID = 298460124341862401
left outer join authors s12 on s12.authorID = o and s12.serverID = 316759222207643649
left outer join authors s13 on s13.authorID = o and s13.serverID = 318970898050973697
"""





db = sqlite3.connect('cleandb.sql')
c = db.cursor()
c.execute(select_stmt)

data = pd.DataFrame(c.fetchall(), columns=['messageID', 'authorID', 'channelID', 'serverID', 'timestamp', 'tts', 'content', 'embeds', 'mention_everyone', 'mentions_list', 'channel_mentions_list', 'role_mentions_list', 'pinned', 'reactions'])

#data = pd.DataFrame(c.fetchall(), columns=['content', 'displayname', 'id'])
data['content'] = data['content'].astype(np.str) # fix wrong inferred type.
data['messageID'] = data['messageID'].astype(np.int64)
data['channelID'] = data['channelID'].astype(np.int64)
data['serverID'] = data['serverID'].astype(np.int64)
print(data)
