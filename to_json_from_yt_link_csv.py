import csv
import json
from pathlib import Path

csvfile = open('output100.csv', 'r')
jsonfile = open(Path(r'C:\Users\suhas\git\tf_frontend\src\meta_yt_links.json'), 'w')

fieldnames = ("name","link")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write(',\n')