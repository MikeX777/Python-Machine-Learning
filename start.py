from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import csv_processing
import data_preprocessing
import machine_learning
import csv_writer
import Classes.Trained as trained


alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
tuning_lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
optimal_alpha_training = [0.0, 0.2, 1.0]
optimal_lambda = 1
iterations = 1000

headers = csv_processing.get_headers('Data/Credit_N400_p9.csv')
data = csv_processing.retrieve_data('Data/Credit_N400_p9.csv', shuffle=True)

# Data preprocessing for Deliverable 1
unfolded = data_preprocessing.make_folds(data.copy(), 0)
separated_unfolded = data_preprocessing.separate_outputs_from_fold_list(unfolded)
scaled_unfolded = data_preprocessing.scale_data_for_fold_list(separated_unfolded)

# Data preprocessing for Deliverable 2 and 3
folded = data_preprocessing.make_folds(data.copy(), 20)
separated_folded = data_preprocessing.separate_outputs_from_fold_list(folded)
scaled_folded = data_preprocessing.scale_data_for_fold_list(separated_folded)

# Data preprocessing for Deliverable 4
optimal = data_preprocessing.make_folds(data.copy(), 0)
separated_optimal = data_preprocessing.separate_outputs_from_fold_list(optimal)
scaled_optimal = data_preprocessing.scale_data_for_fold_list(separated_optimal)

# Training for Deliverable 1
print("Training for Deliverable 1!")
trained_unfolded_alphas = []
for alpha in alphas:
    alpha_to_train = trained.AlphaTrainedWeights(alpha)

    for tuning_lambda in tuning_lambdas:
        print(f'Training Alpha: {alpha} Tuning Lambda: {tuning_lambda}')
        trained_folds = machine_learning.train_elastic_net_fit(alpha, tuning_lambda, iterations, scaled_unfolded)
        alpha_to_train.add_tuned_folds(tuning_lambda, trained_folds)
    alpha_to_train.set_cv()
    trained_unfolded_alphas.append(alpha_to_train)

# Training for Deliverable 2 and 3
print("Training for Deliverable 2")
trained_folded_alphas = []
for alpha in alphas:
    alpha_to_train = trained.AlphaTrainedWeights(alpha)

    for tuning_lambda in tuning_lambdas:
        print(f'Training Alpha: {alpha} Tuning Lambda: {tuning_lambda}')
        trained_folds = machine_learning.train_elastic_net_fit(alpha, tuning_lambda, iterations, scaled_folded)
        alpha_to_train.add_tuned_folds(tuning_lambda, trained_folds)
    alpha_to_train.set_cv()
    trained_folded_alphas.append(alpha_to_train)


# Training for Deliverable 4
print("Training for Deliverable 4")
trained_optimal_alphas = []
for alpha in optimal_alpha_training:
    alpha_to_train = trained.AlphaTrainedWeights(alpha)

    print(f'Training Alpha: {alpha} Tuning Lambda: {optimal_lambda}')
    trained_folds = machine_learning.train_elastic_net_fit(alpha, optimal_lambda, iterations, scaled_optimal)
    alpha_to_train.add_tuned_folds(optimal_lambda, trained_folds)
    alpha_to_train.set_cv()
    trained_optimal_alphas.append(alpha_to_train)
    

csv_writer.write_output_for_assignment('Data/output.csv', trained_unfolded_alphas, trained_folded_alphas, trained_optimal_alphas, headers)
