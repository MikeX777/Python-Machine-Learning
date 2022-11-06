from logging.config import valid_ident
import numpy
import random

import Classes.DataGrouping as dg
import Classes.Fold as f
import Classes.Trained as t

def sign(val):
    if val < 0:
        return -1
    else:
        return 1

def return_if_is_positive(val):
    if val < 0:
        return 0
    else:
        return val

def get_squared_input_summed(data):
    b = []
    for column in data.T:
        b.append(sum(list(map(lambda x: x ** 2, column))))
    
    return b

def generate_starting_weights(data):
    betas = []
    for column in data.T:
        betas.append(random.uniform(-1, 1))

    return betas

def compute_a(inputs, outputs, betas, input_k, beta_k):
    return numpy.matmul(input_k.T, (numpy.add(numpy.subtract(outputs, numpy.matmul(inputs, betas)), (input_k * beta_k))))

def compute_beta(alpha, tuning_lambda, a_k, b_k):
    return (sign(a_k) * return_if_is_positive((abs(a_k) - ((tuning_lambda * (1 - alpha)) / 2)))) / (b_k + (tuning_lambda * alpha))

def compute_mse(weights, inputs, outputs):
    summed_difference = 0
    for i, o in zip(inputs, outputs):
        summed_difference += pow(o - (numpy.sum(numpy.multiply(weights, i))), 2)
    return (summed_difference / len(inputs))

def elastic_net_fit_iteration(alpha, tuning_lambda, inputs, outputs, betas, b_ks):
    columns = inputs.T
    copied_betas = betas.copy()
    for x in range(len(columns)):
        a_k = compute_a(inputs, outputs, copied_betas, columns[x], betas[x])
        copied_betas[x] = compute_beta(alpha, tuning_lambda, a_k, b_ks[x])
    return copied_betas

def train_elastic_net_fit(alpha, tuning_lambda, iterations, folds):
    trained_folds = []

    for i in range(len(folds)):
        training_iteration = 0
        betas = generate_starting_weights(folds[i].training.inputs)
        b_ks = get_squared_input_summed(folds[i].training.inputs)

        while training_iteration < iterations:
            new_betas = elastic_net_fit_iteration(alpha, tuning_lambda, folds[i].training.inputs, folds[i].training.outputs, betas, b_ks)
            if numpy.allclose(betas, new_betas):
                betas = new_betas
                break
            else:
                betas = new_betas
                training_iteration +=1

        tuned_fold = t.LambdaTunedFold(betas, tuning_lambda, i)

        if len(folds) > 1:
            mse = compute_mse(tuned_fold.weights, folds[i].validation.inputs, folds[i].validation.outputs)
            tuned_fold.set_mse(mse)

        trained_folds.append(tuned_fold)
    
    return trained_folds

