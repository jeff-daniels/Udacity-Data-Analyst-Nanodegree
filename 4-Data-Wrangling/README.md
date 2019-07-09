
README

This folder contains the files used for the OpenStreetMap Data Project.  The files include a sample of the map region wrangled in the project and files used to process this sample as well as the full region.

README.md : The eponymous file  

audit.py : Audits the osm data to find and list common issues in the data set.

dc_sample.osm : A sample of the map region used.

map_position.txt : Countains a link to the map region used in this project, Washington, DC.

process_csv.py : creates a database file and inserts tables nodes, nodes_tags, ways, ways_tags, and ways_nodes.

process_osm.py : converts osm data into a dictionay format for insertion into .csv files.  Fields addressed by the update.py script are converted here.

report.pdf : This reports documents the data wrangling process and answers rubric questions

sample.py : Gets a 1% sample from the map region to create dc_sample.osm.  Used to prototype scripts and queries.

update.py : Programatically cleans the data
