import csv

data_path='bank-full.csv'
with open(data_path,'r') as content:
    data=csv.reader(content,delimiter=';',quotechar='"')
    rows=[r for r in data]

output_path='cleanup.csv'
with open(output_path,'wb') as content:
    data=csv.writer(content,quoting=csv.QUOTE_MINIMAL)
    for r in rows:
        print(r)
        data.writerow(r)

