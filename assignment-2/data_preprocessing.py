import numpy

import Classes.DataGrouping as dg
import Classes.Fold as f

def get_center(values):
    return float(sum(values) / len(values))

def get_center_and_standard_deviation(values):
    return (get_center(values), numpy.std(values))

def scale_inputs(inputs, center, std):
    return list(map(lambda i: (i - center) / std, inputs))

def center_outputs(outputs, center):
    return numpy.asarray(list(map(lambda o: o - center, outputs)))

def separate_outputs(data):
    if data.any():
        return dg.DataGrouping(data[:, :-1], data[:, -1])
    return dg.DataGrouping(numpy.array([]), numpy.array([]))


def make_folds(data, validationPercentage):

    if validationPercentage < 0:
        raise ValueError("Validation percentage cannot be less than 0.")

    if validationPercentage == 0:
        return [f.UnprocessedFold(data, numpy.array([]))]

    numOfObservations = data.shape[0]
    if numOfObservations / (100 / validationPercentage) % 1 != 0.0:
        raise ValueError("Validation percentage does not evenly divide observations.")

    numOfFolds = int((100 / validationPercentage))
    numOfValidationObservations = int(numOfObservations / numOfFolds)

    folds = []
    for x in range(numOfFolds):
        if x == 0:
            splitData = numpy.split(data.copy(), [numOfObservations - numOfValidationObservations])
            folds.append(f.UnprocessedFold(splitData[0], splitData[1]))
        elif x == numOfFolds - 1:
            splitData = numpy.split(data.copy(), [numOfValidationObservations])
            folds.append(f.UnprocessedFold(splitData[1], splitData[0]))
        else:
            splitData = numpy.split(data.copy(), [numOfObservations - (numOfValidationObservations * (x + 1)),
                numOfObservations - (numOfValidationObservations * x)])
            folds.append(f.UnprocessedFold(numpy.concatenate((splitData[0], splitData[2])), splitData[1]))
    return folds

def separate_outputs_from_fold(fold):
    return f.Fold(separate_outputs(fold.training), separate_outputs(fold.validation))

def separate_outputs_from_fold_list(folds):
    return list(map(lambda f: separate_outputs_from_fold(f), folds))

def scale_data_for_fold(fold):
    outputCenter = get_center(fold.training.outputs)
    transposed = fold.training.inputs.transpose()
    centers_and_std = []

    for x in range(len(transposed)):
        center_and_std = get_center_and_standard_deviation(transposed[x])
        transposed[x] = scale_inputs(transposed[x], center_and_std[0], center_and_std[1])
        centers_and_std.append(center_and_std)

    fold.training.inputs = transposed.transpose()
    fold.training.outputs = center_outputs(fold.training.outputs, outputCenter)   


    if fold.validation.inputs.any():
        transposedValidation = fold.validation.inputs.transpose()
        for x in range(len(transposedValidation)):
            transposedValidation[x] = scale_inputs(transposedValidation[x],
            centers_and_std[x][0],
            centers_and_std[x][1])
        fold.validation.inputs = transposedValidation.transpose()
        fold.validation.outputs = center_outputs(fold.validation.outputs, outputCenter)
    
    return fold

def scale_data_for_fold_list(folds):
    return list(map(lambda f: scale_data_for_fold(f), folds))
