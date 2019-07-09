"""
This module fixes fields of the type addr:
    street : by spelling out the street type and quadrant
    housenumber : eliminates any suffixes.  Limits length to 1-4 digits
    postcode : accepts only zip codes that begin with '20' and limits them to 5 digits
    
There is also a function for fixing phone numbers:
    phone : converts all phone numbers to the +1 123 456 7890 format
    
Functions try to convert values so that they match the XXXX_FORMAT, if after conversion they fail to match, 
the original value is returned unaltered.
"""
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
quadrant_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
PHONE_FORMAT = re.compile(r'\+1\s\d{3}\s\d{3}\s\d{4}')
POSTCODE_FORMAT = re.compile(r'20\d{3}')
HOUSENUMBER_FORMAT = re.compile(r'\d{1,4}')


street_mapping = { "St": "Street",
                  "St.": "Street",
                  "Rd": "Road",
                  "Rd.": "Road",
                  "Ave": "Avenue",
                  "Ave.": "Avenue"
                 }

quadrant_mapping = {"NW" : "Northwest",
                    "NE" : "Northeast", 
                    "SW" : "Southwest",
                    "SE" : "Southeast"
                   }

def is_street_name(elem):
    return (elem.attrib['k'] == 'addr:street')

def is_phone(elem):
    return (elem.attrib['k'] == 'phone')

def is_housenumber(elem):
    return (elem.attrib['k'] == 'addr:housenumber')

def is_postcode(elem):
    return (elem.attrib['k'] == 'addr:postcode')

def update_name(name):
    old_name = name
    # Get rid of any parenthetical descriptions, ie. "K Street NW (access road)"
    name = re.sub(r'\s\(.*\)', '', name)
    
    # Get rid of that pesky full address
    name = re.sub(r'\, Washington, DC \d*', '', name)
    name = re.sub(r'5601 E Capitol', 'East Capitol', name)
    
    # Update Quadrant and Street Abbreviations
    name = name.split(' ')
    for i in range(len(name)):
        if name[i] in quadrant_mapping:
            name[i] = quadrant_mapping[name[i]]
        if name[i] in street_mapping:
            name[i] = street_mapping[name[i]]
    name = ' '.join(name)
    
    return name

def update_phone(phone):
    m = PHONE_FORMAT.match(phone)
    if m is None:
        # grab all the digits and plus sign
        digits = re.findall(r'\+?\d', phone)
        if digits[0] != '+1':
            digits.insert(0,'+1')
        phone_update = (''.join(digits[0]) + ' ' +''.join(digits[1:4]) + ' ' 
                        + ''.join(digits[4:7]) + ' ' + ''.join(digits[7:11]))
        # Verify that phone_update is valid. If it isn't, phone is returned unchanged
        if PHONE_FORMAT.match(phone_update) is not None:
            return phone_update

    return phone

def update_postcode(postcode):
    m = POSTCODE_FORMAT.match(postcode)
    if m:
        return m.group()
    return postcode

def update_housenumber(housenumber):
    m = HOUSENUMBER_FORMAT.match(housenumber)
    if m:
        return m.group()
    return housenumber

