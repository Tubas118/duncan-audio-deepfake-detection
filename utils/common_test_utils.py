import sys

def CALCULATE_EXPECTED_SCORE(expected_cm, num_classes: int, num_predictions: int):
    count_expected_match = 0

    for idx in range(0, num_classes):
        count_expected_match += expected_cm[idx][idx]

    return float(count_expected_match) / num_predictions


def CONFUSION_MATRIX_CHART_TITLE():
    return f'Confusion Matrix ({sys._getframe(1).f_code.co_name})'