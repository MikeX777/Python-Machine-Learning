class CvAcrossClass:

    def __init__(self, alpha, africanCv, europeanCv, eastAsianCv, oceanianCv, nativeAmericanCv):
        self.alpha = alpha
        self.africanCv = africanCv
        self.europeanCv = europeanCv
        self.eastAsianCv = eastAsianCv
        self.oceanianCv = oceanianCv
        self.nativeAmericanCv = nativeAmericanCv

    def print(self):
        to_write = []

        to_write.append(['Alpha','AfricanCV','EuropeanCV','EastAsianCV','OceanianCV','NativeAmericanCV'])
        to_write.append([f'{self.alpha}', f'{self.africanCv}', f'{self.europeanCv}', f'{self.eastAsianCv}', f'{self.oceanianCv}', f'{self.nativeAmericanCv}'])
        return to_write

    