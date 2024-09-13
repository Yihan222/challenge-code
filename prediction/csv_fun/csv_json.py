import csv
import json

csv_file = open('CO2_raw.csv', 'r')
csv_data = csv.DictReader(csv_file)
rows = []
for row in csv_data:
    row['CO2'] = None
    #print(row)
    rows.append(row)
json_data = json.dumps(rows)

json_file = open('test_dev.json', 'w')
json_file.write(json_data)
