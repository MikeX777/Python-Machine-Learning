from logging.config import valid_ident
import numpy
import random

import Classes.DataGrouping as dg
import Classes.Fold as f
import Classes.Trained as t
import Classes.PredictedTestData as p

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


# Multinominal Ridge Regression

def create_design_matrix(inputs):
    num_of_observations = inputs.shape[0]
    zeroth_column = numpy.ones((num_of_observations, 1))
    return numpy.hstack((zeroth_column, inputs))

def create_response_matrix(outputs):
    num_of_observations = outputs.shape[0]
    respone_matrix = numpy.zeros((num_of_observations, 5))
    for i in range(num_of_observations):
        respone_matrix[i][int(outputs[i])] = 1
    return respone_matrix

def create_normalized_probability_matrix(input_matrix, parameter_matrix):
    unnormalized_probability_matrix = numpy.exp(numpy.matmul(input_matrix, parameter_matrix))
    normalized_probability_matrix = numpy.zeros(unnormalized_probability_matrix.shape)
    summed_rows = numpy.sum(unnormalized_probability_matrix, axis=1)
    for i in range(normalized_probability_matrix.shape[0]):
        for j in range(normalized_probability_matrix.shape[1]):
            normalized_probability_matrix[i][j] = unnormalized_probability_matrix[i][j] / summed_rows[i]
    return normalized_probability_matrix


def compute_multinominal_ridge_regression_train(alpha, 
                                                tuning_lambda,
                                                design_matrix,
                                                response_matrix,
                                                parameter_matrix,
                                                normalized_probability_matrix,
                                                intercept_matrix):

    return numpy.add(parameter_matrix, (alpha * 
            (numpy.subtract(numpy.matmul(design_matrix.T, numpy.subtract(response_matrix, normalized_probability_matrix)),
            (2 * tuning_lambda * (numpy.subtract(parameter_matrix, intercept_matrix)))))))

def multinominal_ridge_regression_iteration(alpha, tuning_lambda, design_matrix, response_matrix, parameter_matrix):
    normalized_probability_matrix = create_normalized_probability_matrix(design_matrix, parameter_matrix)

    intercept_matrix = numpy.zeros(parameter_matrix.shape)
    for i in range(len(parameter_matrix[0])):
        intercept_matrix[0][i] = parameter_matrix[0][i]

    return compute_multinominal_ridge_regression_train( alpha,
                                                        tuning_lambda,
                                                        design_matrix.copy(),
                                                        response_matrix.copy(),
                                                        parameter_matrix.copy(),
                                                        normalized_probability_matrix,
                                                        intercept_matrix)

def compute_cce_for_multinominal_ridge_regression(response_matrix, probability_matrix):
    
    num_of_observations = probability_matrix.shape[0]
    log_probability_matrix = numpy.log10(probability_matrix)
    element_wise_multiply = numpy.multiply(response_matrix, log_probability_matrix)
    summed_matricies = numpy.sum(element_wise_multiply)
    return - float(1 / num_of_observations) * float(summed_matricies)

    
def train_multinominal_ridge_regression(alpha, tuning_lambda, iterations, folds):
    trained_folds = []

    for i in range(len(folds)):
        training_iteration = 0
        design_matrix = create_design_matrix(folds[i].training.inputs.copy())
        response_matrix = create_response_matrix(folds[i].training.outputs.copy())
        parameter_matrix = numpy.zeros((11, 5))

        while training_iteration < iterations:
            parameter_matrix = multinominal_ridge_regression_iteration(alpha, tuning_lambda, design_matrix, response_matrix, parameter_matrix)
            training_iteration += 1

        tuned_fold = t.LambdaTunedFold(parameter_matrix, tuning_lambda, i)

        if (len(folds) > 1):
            validation_design_matrix = create_design_matrix(folds[i].validation.inputs.copy())
            validation_response_matrix = create_response_matrix(folds[i].validation.outputs.copy())
            normalized_validation_probability_matrix = create_normalized_probability_matrix(validation_design_matrix, parameter_matrix)
            cce = compute_cce_for_multinominal_ridge_regression(validation_response_matrix, normalized_validation_probability_matrix)
            tuned_fold.set_cce(cce)

        trained_folds.append(tuned_fold)

    return trained_folds

def predict_test_data(output_class_names, input_data, given_labels, parameter_matrix):
    design_matrix = create_design_matrix(input_data)
    test_normalized_probability_matrix = create_normalized_probability_matrix(design_matrix, parameter_matrix)
    prediction = p.PredictedTestData(output_class_names, test_normalized_probability_matrix, given_labels)
    prediction.set_predicted_labels()
    return prediction



