import sqlite3


# database compactor.
# so that you only have to run it once.
# this program reduces the amount of messages stored in the temp database based on
# criteria that is used for the ML algorithm.
# example: users having less than 1000 messages are filtered out as there is not enough data on those users.
# example: users that do not exist in at least 2 servers are filtered out, as we can only compare users that exist on at least 2 servers.


# get authors who belong to 2 servers who talk a lot.
insert_stmt = """
INSERT INTO main.messages
SELECT messages.* 
FROM cdb.messages
INNER JOIN cdb.authors ON cdb.messages.authorID = cdb.authors.authorID
WHERE cdb.messages.authorID IN 
	(
	SELECT cdb.authors.authorID FROM cdb.authors 
	GROUP BY cdb.authors.authorID
	HAVING COUNT(cdb.authors.authorID) > 1
	)
AND cdb.messages.authorID IN
	(
	SELECT cdb.messages.authorID 
	FROM cdb.messages 
	GROUP BY cdb.messages.authorID 
	HAVING COUNT(cdb.messages.authorID) > 1000
	)
AND cdb.messages.embeds == 0
GROUP BY cdb.messages.timestamp
"""


# get authors who talk a lot
insert_stmt2 = """
INSERT INTO main.messages 
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

schema_stmt = """SELECT sql FROM cdb.sqlite_master WHERE type='table' AND name='messages'"""

dbd = sqlite3.connect('cleandb.sql')
dbd.execute('ATTACH DATABASE "dumpdb.sql" as cdb')
c = dbd.cursor()
c.execute(schema_stmt)
c.execute(c.fetchone()[0]) #contains the sql for creating table schema
c.execute(insert_stmt)
dbd.commit()
dbd.close()
