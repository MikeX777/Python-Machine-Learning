import csv
from functools import reduce
import numpy


def write_output_for_assignment(path, classHeaders, inputHeaders, tuning_lambdas, deliverable1, deliverable2, deliverable4):
    file = open(path, 'w', newline='')
    writer = csv.writer(file)

    writer.writerow(['Deliverable 1'])
    writer.writerows(deliverable1.print_weights_deliverable1_assignment3(classHeaders, inputHeaders, tuning_lambdas))
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])

    writer.writerow(['Deliverable 2'])
    writer.writerows(deliverable2.print_cv_deliverable2_assignment3())
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])

    writer.writerow(['Deliverable 4'])
    writer.writerows(deliverable4.print())
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    writer.writerow(['\n'])
    file.close()


def write_library_output_for_assignment(path, classHeaders, unfolded_models, alpha_cv_across_class, predictions):
    file = open(path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Deliverable 1'])
    for unfolded_model in unfolded_models:
        writer.writerows(unfolded_model.coef_)
        writer.writerow(['\n'])
        writer.writerow(['\n'])
        writer.writerow(['\n'])
        writer.writerow(['\n'])
        writer.writerow(['\n'])

    writer.writerow(['Deliverable 2'])
    for alpha in alpha_cv_across_class:
        writer.writerows(alpha.print())
        writer.writerow(['\n'])

    writer.writerow(['Deliverable 5, predictions'])
    for prediction in predictions:
        writer.writerow([classHeaders[int(prediction)]])

    file.close

