from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_recall_curve, roc_curve, root_mean_squared_error

from processors.model_evaluation_result import ModelEvaluationResult


class Metrics:

    # -------------------------------------------------------------------------
    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    def evaluateResults(self, results: ModelEvaluationResult):
        if results == None:
            raise ValueError("Parameter 'results' must be initialized")
        
        # (Chan, 2021, pg. 94)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        results.accuracy_score = accuracy_score(results.testAry, results.predAry)

        # (Chan, 2021, pg. 95-96)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        results.confusion_matrix = confusion_matrix(results.testAry, results.predAry)
        
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        results.f1_score = f1_score(results.testAry, results.predAry)

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html
        results.root_mean_squared_error = root_mean_squared_error(results.testAry, results.predAry)

        
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
        #   - fpr = Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
        #   - tpr = Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        fpr, tpr, _ = roc_curve(results.testAry, results.predAry)
        results.roc_fpr = fpr
        results.roc_tpr = tpr

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
        #   - auc = Area Under the Curve
        results.roc_auc = auc(fpr, tpr)

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
        results.precision_recall_curve = precision_recall_curve(results.testAry, results.predAry)
