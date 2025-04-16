import numpy as np
import pytz
from datetime import datetime
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
    def __init__(self, testAry: np.array, predAry: np.array):
        if (len(testAry) != len(predAry)):
            raise ValueError("Arrays must be same length")
        
        self.batchId = nextModelEvaluationResultIndex()
        self.timestamp_utc = datetime.now(pytz.utc)

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

        for key in self.__dict__:
            if key not in IGNORE_KEYS:
                value = self.__dict__.get(key)
                value = self.__check_formatting__(key, value)
                t.add_row([key, value])

        report = report + f"{t}\n"

        report = report + f"{indent}--- Results (end) ---"

        return report


    # -------------------------------------------------------------------------
    def __check_formatting__(self, key, value):
        workingValue = value

        if (key == 'precision_recall_curve'):
            workingValue = str(workingValue).replace("),", "),\n")

        return workingValue