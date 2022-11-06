import csv
import numpy
import random

def make_data_binary(data):
    if data == 'Male' or data == 'No':
        return float(0)
    elif data == 'Female' or data == 'Yes':
        return float(1)
    else:
        return float(data)

def get_headers(file_name):
    file = open(file_name)
    file_reader = csv.reader(file)
    headers = next(file_reader)
    file.close()
    return headers

def retrieve_data(file_name, shuffle):
    file = open(file_name)
    file_reader = csv.reader(file)
    headers = next(file_reader)

    mapped_data = [] 
    for row in file_reader:
        mapped = map(lambda x: make_data_binary(x), row)
        to_add = list(mapped)
        mapped_data.append(to_add)

    if shuffle:
        random.shuffle(mapped_data)


    file.close()
    return numpy.asarray(mapped_data)