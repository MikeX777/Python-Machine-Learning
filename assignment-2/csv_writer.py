import csv
from functools import reduce
import numpy

def map_headers(headers):
    return reduce(lambda previousHeaders, next: previousHeaders + f'{next}', headers, '')

def map_inputs(inputs):
    lines = []
    for observation in inputs:
        lines.append(reduce(lambda prevInputs, next: prevInputs + f'{next},', observation, ''))
    return lines

def write_weights(path, weights):
    file = open(path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['{:f}'.format(weight) for weight in weights[0]])
    file.close()

def write_output_for_assignment(path, deliverable1, deliverable2, deliverable4, headers):
    file = open(path, 'w', newline='')
    writer = csv.writer(file)

    writer.writerow(['Deliverable 1'])
    for trained_alpha in deliverable1:
        writer.writerows(trained_alpha.print(headers))
        writer.writerow(['\n'])
        writer.writerow(['\n'])
        writer.writerow(['\n'])
        writer.writerow(['\n'])
        writer.writerow(['\n'])
        writer.writerow(['\n'])

    writer.writerow(['Deliverable 2'])
    for trained_alpha in deliverable2:
        writer.writerows(trained_alpha.print(headers))
    
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])

    writer.writerow(['Deliverable 4'])
    for trained_alpha in deliverable4:
        writer.writerows(trained_alpha.print(headers))

    file.close()


def write_trained_weights(path, trained_weights, headers):
    file = open(path, 'w', newline='')
    writer = csv.writer(file)

    for trained_weight in trained_weights:
        writer.writerow([f'Alpha: {trained_weight.alpha}', f'Tuning Lambda: {trained_weight.tuning_lambda}', f'Fold number: {trained_weight.fold_number}'])
        writer.writerow(['Weights'])
        writer.writerow(headers)
        writer.writerow('{:f}'.format(weight) for weight in trained_weight.weights)
        writer.writerows('\n\n\n')

    file.close
    

def write_folds(path, folds, headers):

    file = open(path, 'w', newline='')
    writer = csv.writer(file)

    for x in range(len(folds)):
        writer.writerow([f'Fold number: {x}'])

        writer.writerow(['Training'])
        writer.writerow(headers)
        writer.writerows(numpy.hstack((folds[x].training.inputs, numpy.expand_dims(folds[x].training.outputs, axis=1))))

        writer.writerows('\n\n\n')

        writer.writerow(['Validation'])
        writer.writerow(headers)
        writer.writerows(folds[x].validation.inputs)

        writer.writerow('\n\n\n')
    
    file.close()

