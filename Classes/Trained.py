class AlphaTrainedWeights:
    def __init__(self, alpha):
        self.alpha = alpha
        self.trained_tuned_folds = dict()
        self.cvs = dict()

    def add_tuned_folds(self, tuning_lambda, tuned_folds):
        self.trained_tuned_folds[tuning_lambda] = tuned_folds

    def set_cv(self):
        for key, value in self.trained_tuned_folds.items():
            self.cvs[key] = sum(fold.mse for fold in value)

    def print(self, headers):
        to_write = []
        to_write.append([f'Alpha: {self.alpha}'])
        to_write.append(['Lambda'] + headers[:-1] + ['Fold Number', 'MSE'])
        for key, value in self.trained_tuned_folds.items():
            for fold in value:
                to_write.append([f'{key}'] + fold.weights + [fold.fold_number, fold.mse])
            if self.cvs[key] > 0:
                to_write.append(['CV: ', self.cvs[key], 'Lambda: ', f'{key}', 'Alpha: ', f'{self.alpha}'])
        to_write.append(['\n'])
        return to_write



class LambdaTunedFold:
    def __init__(self, weights, tuning_lambda, fold_number):
        self.weights = weights
        self.tuning_lambda = tuning_lambda
        self.fold_number = fold_number
        self.mse = 0

    def set_mse(self, mse):
        self.mse = mse