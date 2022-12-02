import numpy as numpy

class PredictedTestData:
    def __init__(self, class_list, probability_matrix, given_labels):
        self.class_list = class_list
        self.probability_matrix = probability_matrix
        self.given_labels = given_labels
        self.predicted_labels = []

    def set_predicted_labels(self):
        for row in self.probability_matrix:
            self.predicted_labels.append(self.class_list[numpy.argmax(row)])

    def print(self):
        to_write = []

        to_write.append(['Probability Matrix for test Data'])
        to_write.append(self.class_list + ['predicted label', 'given label'])
        for row, prediction, given in zip(self.probability_matrix, self.predicted_labels, self.given_labels):
            to_write.append([str(x) for x in row] + [prediction, given])
        to_write.append(['\n'])
        return to_write

        