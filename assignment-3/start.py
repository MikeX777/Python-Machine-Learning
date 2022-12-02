from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import numpy

import csv_processing
import data_preprocessing
import machine_learning
import csv_writer
import Classes.Trained as trained
import Classes.CvAcrossClass as cvData


alpha = 0.00002
tuning_lambdas = [0.0001, 0.001, .01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
classHeaders = ['African', 'European', 'EastAsian', 'Oceanian', 'NativeAmerican']
inputHeaders = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
iterations = 10000

headers = csv_processing.get_headers('Data/TrainingData_N183_p10.csv')
data = csv_processing.retrieve_data('Data/TrainingData_N183_p10.csv', shuffle=True)
unfolded = data_preprocessing.make_folds(data.copy(), 1)
separated_unfolded = data_preprocessing.separate_outputs_from_fold_list(unfolded)
scaled_unfolded = data_preprocessing.scale_data_for_fold_list(separated_unfolded, center_outputs=False)

folded = data_preprocessing.make_folds(data.copy(), 5)
separated_folded = data_preprocessing.separate_outputs_from_fold_list(folded)
scaled_folded = data_preprocessing.scale_data_for_fold_list(separated_folded, center_outputs=False)

print('Training for deliverable 1.')
deliverable1 = trained.AlphaTrainedWeights(alpha)
for tuning_lambda in tuning_lambdas:
    print(f'Training Alpha: {alpha} Tuning Lambda: {tuning_lambda}')
    trained_folds = machine_learning.train_multinominal_ridge_regression(alpha, tuning_lambda, iterations, scaled_unfolded)
    deliverable1.add_tuned_folds(tuning_lambda, trained_folds)
    deliverable1.set_cv_from_cce()

print('Training for deliverable 2.')
deliverable2 = trained.AlphaTrainedWeights(alpha)
for tuning_lambda in tuning_lambdas:
    print(f'Training Alpha: {alpha} Tuning Lambda: {tuning_lambda}')
    trained_folds = machine_learning.train_multinominal_ridge_regression(alpha, tuning_lambda, iterations, scaled_folded)
    deliverable2.add_tuned_folds(tuning_lambda, trained_folds)
    deliverable2.set_cv_from_cce()

lambda_with_min_cv = min(deliverable2.cvs, key=deliverable2.cvs.get)
unfolded_optimal = data_preprocessing.make_folds(data.copy(), 1)
separated_unfolded_optimal = data_preprocessing.separate_outputs_from_fold_list(unfolded_optimal)
scaled_unfolded_optimal = data_preprocessing.scale_data_for_fold_list(separated_unfolded_optimal, center_outputs=False)

print('Training for deliverable 4.')
print(f'Training Alpha: {alpha}, Tuning Lambda: {lambda_with_min_cv}')
trained_optimal = machine_learning.train_multinominal_ridge_regression(alpha, lambda_with_min_cv, iterations, scaled_unfolded_optimal)[0]


test_data = csv_processing.retrieve_test_data('Data/TestData_N111_p10.csv')
unfolded_test = data_preprocessing.make_folds(test_data.copy(), 1)[0]
separated_unfolded_test = data_preprocessing.separate_test_outputs_from_fold(unfolded_test)
separated_unfolded_test.training.inputs = data_preprocessing.map_to_float(separated_unfolded_test.training.inputs)
deliverable4 = machine_learning.predict_test_data(classHeaders,
                                                        separated_unfolded_test.training.inputs,
                                                        separated_unfolded_test.training.outputs, 
                                                        trained_optimal.weights)


print("Starting deliverable 7.")
combined_lambda_and_alpha = list(map(lambda l: l * alpha, tuning_lambdas))
unfolded_models = []
folded_models = []

for tuning in combined_lambda_and_alpha:
    unfolded_model = RidgeClassifier(tuning, max_iter=iterations).fit(data.copy()[:, :-1], data.copy()[:, -1])
    folded_model = RidgeClassifierCV([tuning], store_cv_values=True).fit(data.copy()[:, :-1], data.copy()[:, -1])
    unfolded_models.append(unfolded_model)
    folded_models.append(folded_model)

alpha_cv_across_class = []

for folded_model in folded_models:
    africanCvSum = float(0)
    europeanCvSum = float(0)
    eastAsianCvSum = float(0)
    oceanianCvSum = float(0)
    nativeAmericanCvSum = float(0)
    for cv in folded_model.cv_values_:
        africanCvSum += cv[0][0]
        europeanCvSum += cv[1][0]
        eastAsianCvSum += cv[2][0]
        oceanianCvSum += cv[3][0]
        nativeAmericanCvSum += cv[4][0]

    compiled = cvData.CvAcrossClass(folded_model.alpha_,
                                    numpy.average(africanCvSum),
                                    numpy.average(europeanCvSum),
                                    numpy.average(eastAsianCvSum),
                                    numpy.average(oceanianCvSum),
                                    numpy.average(nativeAmericanCvSum))
    alpha_cv_across_class.append(compiled)

min_alpha_from_training = min(alpha_cv_across_class,
    key=lambda trained_cvs: trained_cvs.africanCv + trained_cvs.europeanCv + trained_cvs.eastAsianCv + trained_cvs.oceanianCv + trained_cvs.nativeAmericanCv)

optimal_model = RidgeClassifier(min_alpha_from_training.alpha, max_iter=iterations).fit(data.copy()[:, :-1], data.copy()[:, -1])
predictions = optimal_model.predict(separated_unfolded_test.training.inputs)


csv_writer.write_output_for_assignment('Data/output.csv', classHeaders, inputHeaders, tuning_lambdas, deliverable1, deliverable2, deliverable4)
csv_writer.write_library_output_for_assignment('Data/library_output.csv', classHeaders, unfolded_models, alpha_cv_across_class, predictions)
    

    


# unfolded_model = RidgeClassifier(max_iter=iterations)
# grid_unfolded = GridSearchCV(estimator=unfolded_model, param_grid=dict(alpha=combined_lambda_and_alpha))
# grid_unfolded.fit(data.copy()[:, :-1], data.copy()[:, -1])
# print(grid_unfolded.)
# print(grid_unfolded.best_estimator_.alpha)

#model = RidgeClassifierCV([alpha], store_cv_values=True, cv=5).fit(data.copy()[:, :-1], data.copy()[:, -1])
#print(unfolded_model.score(data.copy()[:, :-1], data.copy()[:, -1]))
#print(unfolded_model.coef_)
#prediction = unfolded_model.predict(separated_unfolded_test.training.inputs)
#print(prediction)



#

#trained_folds = machine_learning.train_multinominal_ridge_regression(alpha, tuning_lambdas[0], iterations, scaled_folded)

# cv = 0.0

# for fold in trained_folds:
#     print("new fold:")
#     print("CCE: ", fold.cce)
#     cv += fold.cce

# cv = cv / float(5)

# print("CV: ", cv)

# tuned_training = []
# for tuning_lambda in tuning_lambdas:
#     trained_folds = machine_learning.train_multinumoial_ridge_regression(alpha, tuning_lambda, iterations, scaled_folded)


# for index, fold in enumerate(folded):
#     print(f'Fold number: {index}')
#     print(f'Training shape: {fold.training.shape}')
#     print(f'validation shape: {fold.validation.shape}')

# # Data preprocessing for Deliverable 1
# unfolded = data_preprocessing.make_folds(data.copy(), 0)
# separated_unfolded = data_preprocessing.separate_outputs_from_fold_list(unfolded)
# scaled_unfolded = data_preprocessing.scale_data_for_fold_list(separated_unfolded)

# # Data preprocessing for Deliverable 2 and 3
# folded = data_preprocessing.make_folds(data.copy(), 20)
# separated_folded = data_preprocessing.separate_outputs_from_fold_list(folded)
# scaled_folded = data_preprocessing.scale_data_for_fold_list(separated_folded)

# # Data preprocessing for Deliverable 4
# optimal = data_preprocessing.make_folds(data.copy(), 0)
# separated_optimal = data_preprocessing.separate_outputs_from_fold_list(optimal)
# scaled_optimal = data_preprocessing.scale_data_for_fold_list(separated_optimal)

# # Training for Deliverable 1
# print("Training for Deliverable 1!")
# trained_unfolded_alphas = []
# for alpha in alphas:
#     alpha_to_train = trained.AlphaTrainedWeights(alpha)

#     for tuning_lambda in tuning_lambdas:
#         print(f'Training Alpha: {alpha} Tuning Lambda: {tuning_lambda}')
#         trained_folds = machine_learning.train_elastic_net_fit(alpha, tuning_lambda, iterations, scaled_unfolded)
#         alpha_to_train.add_tuned_folds(tuning_lambda, trained_folds)
#     alpha_to_train.set_cv()
#     trained_unfolded_alphas.append(alpha_to_train)

# # Training for Deliverable 2 and 3
# print("Training for Deliverable 2")
# trained_folded_alphas = []
# for alpha in alphas:
#     alpha_to_train = trained.AlphaTrainedWeights(alpha)

#     for tuning_lambda in tuning_lambdas:
#         print(f'Training Alpha: {alpha} Tuning Lambda: {tuning_lambda}')
#         trained_folds = machine_learning.train_elastic_net_fit(alpha, tuning_lambda, iterations, scaled_folded)
#         alpha_to_train.add_tuned_folds(tuning_lambda, trained_folds)
#     alpha_to_train.set_cv()
#     trained_folded_alphas.append(alpha_to_train)


# # Training for Deliverable 4
# print("Training for Deliverable 4")
# trained_optimal_alphas = []
# for alpha in optimal_alpha_training:
#     alpha_to_train = trained.AlphaTrainedWeights(alpha)

#     print(f'Training Alpha: {alpha} Tuning Lambda: {optimal_lambda}')
#     trained_folds = machine_learning.train_elastic_net_fit(alpha, optimal_lambda, iterations, scaled_optimal)
#     alpha_to_train.add_tuned_folds(optimal_lambda, trained_folds)
#     alpha_to_train.set_cv()
#     trained_optimal_alphas.append(alpha_to_train)
    

# csv_writer.write_output_for_assignment('Data/output.csv', trained_unfolded_alphas, trained_folded_alphas, trained_optimal_alphas, headers)
