import csv
import os
pth = os.path.dirname(os.path.abspath(__file__))

def FileAdder(data, fl):
    with open(pth+fl, 'a',newline='') as price_data:
        price_writer = csv.writer(price_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        price_writer.writerow(data)

def FileGetter():
    ls = []
    with open(pth+'/pricedata.csv', 'r',newline='') as price_data:
        price_reader = csv.reader(price_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in price_reader:
            ls.append(row)
    return ls