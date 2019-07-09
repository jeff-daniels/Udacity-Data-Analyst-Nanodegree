'''
Audits the osm data to find and list common issues in the data set

'''

OSM_FILE = "district-of-columbia-latest.osm"  # Replace this with your osm file
SAMPLE_FILE = "dc_sample.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

WORKING_FILE = SAMPLE_FILE

if WORKING_FILE == SAMPLE_FILE:
    DB_FILE = 'dc_sample.db'
else:
    DB_FILE = 'dc.db'



import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

# OSMFILE = "example.osm"
# OSMFILE = WORKING_FILE
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = { "St": "Street",
            "St.": "Street",
            "Rd": "Road",
            "Rd.": "Road",
            "Ave": "Avenue",
            "Ave.": "Avenue",
            "N": "North",
            "N.": "North"
            }


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


def test():
    st_types = audit(WORKING_FILE)
    pprint.pprint(dict(st_types))



if __name__ == '__main__':
    test()
