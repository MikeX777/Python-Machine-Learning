import csv
import numpy
import random

def make_data_floats(data):
    if data == 'African':
        return float(0)
    elif data == 'European':
        return float(1)
    elif data == 'EastAsian':
        return float(2)
    elif data == 'Oceanian':
        return float(3)
    elif data == 'NativeAmerican':
        return float(4)
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
        mapped = map(lambda x: make_data_floats(x), row)
        to_add = list(mapped)
        mapped_data.append(to_add)

    if shuffle:
        random.shuffle(mapped_data)


    file.close()
    return numpy.asarray(mapped_data)

def retrieve_test_data(file_name):
    file = open(file_name)
    file_reader = csv.reader(file)
    headers = next(file_reader)

    test_data = []
    for row in file_reader:
        test_data.append(row)

    file.close()
    return numpy.asarray(test_data)