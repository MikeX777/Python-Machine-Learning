class UnprocessedFold:
    def __init__(self, training, validation):
        self.training = training
        self.validation = validation


class Fold:
    def __init__(self, training, validation):
        self.training = training
        self.validation = validation