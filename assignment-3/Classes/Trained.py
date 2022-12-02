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

    def set_cv_from_cce(self):
        for key, value in self.trained_tuned_folds.items():
            self.cvs[key] = float(sum(fold.cce for fold in value)) / float(len(value))

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

    def print_weights_deliverable1_assignment3(self, classHeaders, inputHeaders, tuning_lambdas):
        to_write = []
        for k in range(0, 5):
            to_write.append([f'Alpha: {self.alpha}', f'Class: {classHeaders[k]}'])
            to_write.append(['lambda'] + [str(x) for x in inputHeaders[1:]])
            for key, value in self.trained_tuned_folds.items():
                to_write.append([key] + [str(x) for x in value[0].weights.T[k][1:]])
            to_write.append(['\n'])
            to_write.append(['\n'])
        return to_write

    def print_assignment3(self, classHeaders, inputHeaders):
        to_write = []
        to_write.append([f'Alpha: {self.alpha}'])
        for key, value in self.trained_tuned_folds.items():
            to_write.append(['Lambda', key])
            to_write.append(classHeaders)
            for fold in value:
                for row, inputHeader in zip(fold.weights, inputHeaders):
                    to_write.append([f'{inputHeader}'] + [str(x) for x in row])
            if self.cvs[key] > 0:
                to_write.append(['CV: ', self.cvs[key], 'Lambda: ', f'{key}'])
        to_write.append(['\n'])
        return to_write

    def print_cv_deliverable2_assignment3(self):
        to_write = []
        to_write.append([f'Alpha: {self.alpha}'])
        to_write.append(['Lambda', 'CV_5'])
        for key, value in self.trained_tuned_folds.items():
            to_write.append([f'{key}', f'{self.cvs[key]}'])
        to_write.append(['\n'])
        return to_write
                    



class LambdaTunedFold:
    def __init__(self, weights, tuning_lambda, fold_number):
        self.weights = weights
        self.tuning_lambda = tuning_lambda
        self.fold_number = fold_number
        self.mse = 0
        self.cce = 0

    def set_mse(self, mse):
        self.mse = mse

    def set_cce(self, cce):
        self.cce = cce