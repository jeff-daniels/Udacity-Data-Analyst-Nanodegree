'''
CREATE TABLE nodes, nodes_tags, ways, ways_tags, ways_nodes

'''

# FILE PATHS and GLOBAL VARIABLES

OSM_FILE = "district-of-columbia-latest.osm"  # Replace this with your osm file
SAMPLE_FILE = "dc_sample.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

WORKING_FILE = OSM_FILE

if WORKING_FILE == SAMPLE_FILE:
    DB_FILE = 'dc_sample.db'
else:
    DB_FILE = 'dc.db'


import sqlite3
import csv
from pprint import pprint

# Connect to the database
conn = sqlite3.connect(DB_FILE)
conn.text_factory = str

# Get a cursor object
cur = conn.cursor()

###### CREATE TABLE nodes ######
cur.execute('''DROP TABLE IF EXISTS nodes''')
conn.commit()

# Create the table, specifying the column names and data types:
cur.execute('''
    CREATE TABLE nodes (
    id INTEGER PRIMARY KEY NOT NULL,
    lat REAL,
    lon REAL,
    user TEXT,
    uid INTEGER,
    version INTEGER,
    changeset INTEGER,
    timestamp TEXT);
    ''')
conn.commit()

# Read in the csv file as a dictionary, format the data as a list of tuples
with open(NODES_PATH, 'rb') as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['lat'], i['lon'], i['user'], i['uid'], i['version'], i['changeset'], 
              i['timestamp'])
             for i in dr]
    
# insert the formatted data
cur.executemany("INSERT INTO nodes (id, lat, lon, user, uid, version, changeset, timestamp)                 VALUES (?,?,?,?,?,?,?,?);", to_db)
conn.commit()

###### CREATE TABLE nodes_tags ######
cur.execute('''DROP TABLE IF EXISTS nodes_tags''')
conn.commit()

# Create the table, specifying the column names and data types:
cur.execute('''
    CREATE TABLE nodes_tags (
    id INTEGER,
    key TEXT,
    value TEXT,
    type TEXT,
    FOREIGN KEY (id) REFERENCES nodes(id));
    ''')
conn.commit()

# Read in the csv file as a dictionary, format the data as a list of tuples
with open(NODE_TAGS_PATH, 'rb') as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['key'], i['value'].decode("utf-8"), i['type']) for i in dr]
    
# insert the formatted data
cur.executemany("INSERT INTO nodes_tags(id, key, value, type) VALUES (?,?,?,?);", to_db)
conn.commit()

##### CREATE TABLE ways #####
cur.execute('''DROP TABLE IF EXISTS ways''')
conn.commit()

# Create the table, specifying the column names and data types:
cur.execute('''
    CREATE TABLE ways (
        id INTEGER PRIMARY KEY NOT NULL,
        user TEXT,
        uid INTEGER,
        version TEXT,
        changeset INTEGER,
        timestamp TEXT
    );
    ''')
conn.commit()

# Read in the csv file as a dictionary, format the data as a list of tuples
with open(WAYS_PATH, 'rb') as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['user'], i['uid'], i['version'], i['changeset'], i['timestamp']) for i in dr]
    
# insert the formatted data
cur.executemany("INSERT INTO ways(id, user, uid, version, changeset, timestamp) VALUES (?,?,?,?,?,?);", to_db)
conn.commit()

##### CREATE TABLE ways_tags #####
cur.execute('''DROP TABLE IF EXISTS ways_tags''')
conn.commit()

# Create the table, specifying the column names and data types:
cur.execute('''
    CREATE TABLE ways_tags (
    id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT,
    FOREIGN KEY (id) REFERENCES ways(id));
    ''')
conn.commit()

# Read in the csv file as a dictionary, format the data as a list of tuples
with open(WAY_TAGS_PATH, 'rb') as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['key'], i['value'], i['type']) for i in dr]
    
# insert the formatted data
cur.executemany("INSERT INTO ways_tags(id, key, value, type) VALUES (?,?,?,?);", to_db)
conn.commit()

##### CREATE TABLE ways_nodes #####
cur.execute('''DROP TABLE IF EXISTS ways_nodes''')
conn.commit()

# Create the table, specifying the column names and data types:
cur.execute('''
    CREATE TABLE ways_nodes (
    id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY (id) REFERENCES ways(id),
    FOREIGN KEY (node_id) REFERENCES nodes(id));
    ''')
conn.commit()

# Read in the csv file as a dictionary, format the data as a list of tuples
with open(WAY_NODES_PATH, 'rb') as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['node_id'], i['position']) for i in dr]
    
# insert the formatted data
cur.executemany("INSERT INTO ways_nodes (id, node_id, position) VALUES (?,?,?);", to_db)
conn.commit()

conn.close()

