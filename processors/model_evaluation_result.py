import copy
import numpy as np
import pytz
from datetime import datetime
import pprint
from prettytable import PrettyTable

__nextModelEvaluationResultIndex__ = 0

# =============================================================================
def nextModelEvaluationResultIndex() -> int:
    global __nextModelEvaluationResultIndex__
    __nextModelEvaluationResultIndex__ = __nextModelEvaluationResultIndex__ + 1
    return __nextModelEvaluationResultIndex__

# =============================================================================
def resetNextModelEvaluationResultIndex():
    global __nextModelEvaluationResultIndex__
    __nextModelEvaluationResultIndex__ = 0

# =============================================================================
class ModelEvaluationResult:

    # -------------------------------------------------------------------------
    def __init__(self, testAry: np.array, predAry: np.array, cross_validation_scores = None):
        if (len(testAry) != len(predAry)):
            raise ValueError("Arrays must be same length")
        
        self.batchId = nextModelEvaluationResultIndex()
        self.timestamp_utc = datetime.now(pytz.utc)

        self.cross_validation_scores = cross_validation_scores

        self.batchSize = len(testAry)
        self.testAry: np.array = testAry
        self.predAry: np.array = predAry

        self.accuracy_score: float = None
        self.confusion_matrix = None
        self.f1_score = None
        self.root_mean_squared_error = None

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
        #   - fpr = Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
        #   - tpr = Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        self.roc_fpr = None
        self.roc_tpr = None

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
        #   - auc = Area Under the Curve
        self.roc_auc = None

        self.precision_recall_curve = None


    # -------------------------------------------------------------------------
    def reportSnaphot(self) -> str:
        indent = "  "
        IGNORE_KEYS = ["testAry", "predAry"]


        t = PrettyTable(['Key', 'Value'])
        t.align = 'l'

        report = f"{indent}--- Results (start) ---\n"
        values: dict[str, any] = copy.deepcopy(self.__dict__)
        for ignoreKey in IGNORE_KEYS:
            del values[ignoreKey]

        for key in values:
            value = values.get(key)
            # if self.__report_raw_value__(value) == False:
            #     value = pprint.pformat(value, indent=2)
            value = self.__check_value_formatting__(key, value)
            t.add_row([key, value])

        report = report + f"{t}\n"

        report = report + f"{indent}--- Results (end) ---"

        return report


    # -------------------------------------------------------------------------
    def __check_value_formatting__(self, key, value):
        workingValue = value

        match key:
            case 'precision_recall_curve':
                workingValue = str(workingValue).replace("),", "),\n")
            case 'cross_validation_scores':
                workingValue = pprint.pformat(workingValue, indent=2)
            case _:
                # no action by design
                ignore = workingValue

        return workingValue