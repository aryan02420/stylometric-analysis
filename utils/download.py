import os
import csv
import re
# import urllib

def id_to_url(id):
  path = id.split()[:-1]
  path.extend(id+'.txt')
  return 'http://aleph.gutenberg.org/' + path.join('/')

def author_to_path(author):
  path = re.sub("\W", "_", author.lower())
  path = re.sub("_+", "_", path)
  return os.path.join('.', 'data_sources', path)

if __name__ == "__main__":
  with open('data.csv') as csvfile:
      csvreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
      next(csvreader)
      for row in csvreader:
        print(row)
        # url = id_to_url(row(2))
        # path = author_to_path(row(0))
        # filename = os.path.join(path, row(2))
        # response = urllib.urlopen(url)
        # print(url, path)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # with open(filename, "wb") as file:
        #     file.write(response.read())
