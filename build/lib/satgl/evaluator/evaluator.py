import torch

from sklearn.metrics import roc_auc_score

def accuracy(pred: torch.Tensor, label: torch.Tensor) -> float:
    r"""
    Calculate the accuracy of the model.

    Parameters
    ----------
    pred : torch.Tensor
        The model's prediction.
    label : torch.Tensor
        The true label.
    """
    if pred.dim() == 2:
        pred = torch.argmax(pred, dim=1)
    else:
        pred = pred > 0.5
    correct = pred.eq(label).sum().item()
    return correct / len(label)

def micro_f1(pred: torch.Tensor, label: torch.Tensor, return_detail=True) -> float:
    r"""
    Calculate the F1 score of the model. If return_detail is True, also return the tp, fp, tn, fp

    Parameters
    ----------
    pred : torch.Tensor
        The model's prediction.
    label : torch.Tensor
        The true label.
    """
    if pred.dim() == 2:
        pred = torch.argmax(pred, dim=1)
    else:
        pred = pred > 0.5

    label = label.bool()
    tp = (pred & label).sum().item()
    fp = (pred & ~label).sum().item()
    fn = (~pred & label).sum().item()
    tn = (~pred & ~label).sum().item()

    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0

    tp = tp / len(label)
    fp = fp / len(label)
    tn = tn / len(label)
    fn = fn / len(label)

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    if return_detail:
        return f1, tp, fp, tn, fn
    else:
        return f1




class Evaluator(object):
    def __init__(
            self,
            eval_fn: callable,
            eval_fn_name: str,
            compare_type: str,
        ) -> None:
        super(Evaluator, self).__init__()
        self.eval_fn = eval_fn
        self.eval_fn_name = eval_fn_name

        self.compare_type = compare_type

        self.reset()
    
    def better(self, other_evaluator):
        if self.compare_type == ">":
            return self.eval_status[self.eval_fn_name] > other_evaluator.eval_status[self.eval_fn_name]
        else:
            return self.eval_status[self.eval_fn_name] < other_evaluator.eval_status[self.eval_fn_name]

    def reset(self):
        raise NotImplementedError("reset method not implemented")

    def update(self, other_evaluator):
        raise NotImplementedError("update method not implemented")

    def __call__(self, pred, label):
        return self.eval_fn(pred, label)
    

class Accuracy(Evaluator):
    def __init__(self):
        super(Accuracy, self).__init__(
            eval_fn=accuracy,
            eval_fn_name="accuracy",
            compare_type=">",
        )
    
    def reset(self):
        self.eval_status = {
            "data_size": 0,
            "accuracy": 0
        }
    
    def __call__(self, pred, label):
        acc = self.eval_fn(pred, label)
        data_size = len(label)

        cur_acc_sum = self.eval_status["accuracy"] * self.eval_status["data_size"]
        
        new_acc_sum = acc * data_size
        
        self.eval_status["data_size"] += data_size
        self.eval_status["accuracy"] = (cur_acc_sum + new_acc_sum) / self.eval_status["data_size"]


class MicroF1(Evaluator):
    def __init__(self):
        super(MicroF1, self).__init__(
            eval_fn=micro_f1,
            eval_fn_name="micro_f1",
            compare_type=">",
        )

    def reset(self):
        self.eval_status = {
            "data_size": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "micro_f1": 0
        }

    def __call__(self, pred, label):
        f1, tp, fp, tn, fn = self.eval_fn(pred, label)
        data_size = len(label)

        cur_tp_sum = self.eval_status["tp"] * self.eval_status["data_size"]
        cur_fp_sum = self.eval_status["fp"] * self.eval_status["data_size"]
        cur_tn_sum = self.eval_status["tn"] * self.eval_status["data_size"]
        cur_fn_sum = self.eval_status["fn"] * self.eval_status["data_size"]
        cur_f1_sum = self.eval_status["micro_f1"] * self.eval_status["data_size"]

        new_tp_sum = tp * data_size
        new_fp_sum = fp * data_size
        new_tn_sum = tn * data_size
        new_fn_sum = fn * data_size
        new_f1_sum = f1 * data_size

        self.eval_status["data_size"] += data_size
        self.eval_status["tp"] = (cur_tp_sum + new_tp_sum) / self.eval_status["data_size"]
        self.eval_status["fp"] = (cur_fp_sum + new_fp_sum) / self.eval_status["data_size"]
        self.eval_status["tn"] = (cur_tn_sum + new_tn_sum) / self.eval_status["data_size"]
        self.eval_status["fn"] = (cur_fn_sum + new_fn_sum) / self.eval_status["data_size"]
        self.eval_status["micro_f1"] = (cur_f1_sum + new_f1_sum) / self.eval_status["data_size"]


class AUC(Evaluator):
    def __init__(self):
        super(AUC, self).__init__(
            eval_fn=auc,
            eval_fn_name="auc",
            compare_type=">",
        )

    def reset(self):
        self.eval_status = {
            "data_size": 0,
            "auc": 0
        }
    
    def __call__(self, pred, label):
        auc = self.eval_fn(pred, label)
        data_size = len(label)

        cur_auc_sum = self.eval_status["auc"] * self.eval_status["data_size"]
        
        new_auc_sum = auc * data_size
        
        self.eval_status["data_size"] += data_size
        self.eval_status["auc"] = (cur_auc_sum + new_auc_sum) / self.eval_status["data_size"]

""" 
dict to store the evaluator for each metric

format: 
    "metric_name": Evaluator
"""
evaluator_dict = {
    "accuracy": Accuracy,
    "micro_f1": MicroF1
}

# def find_evaluator(eval_metric: "str") -> Evaluator:
#     r"""
#     Find the evaluator for the given metric.

#     Parameters
#     ----------
#     eval_metric : str
#         The evaluation metric.
#     """
#     if eval_metric not in evaluator_dict:
#         raise ValueError(f"evaluator for {eval_metric} not found")
#     return evaluator_dict[eval_metric]


# def get_all_evaluator(eval_metrics: list) -> dict:
#     r"""
#     Get evaluator for each metric.

#     Parameters
#     ----------
#     eval_metrics : list
#         The list of evaluation metrics.
#     """
#     evaluator_dict = {}
#     for metric in eval_metrics:
#         evaluator_dict[metric] = evaluator_dict[metric]
#     return evaluator_dict
    
class EvaluatorManager(object):
    def __init__(self, eval_metrics: list):
        self.eval_metrics = eval_metrics
        self.evaluators = {}
        for metric in self.eval_metrics:
            self.add_evaluator(metric)

    
    def add_evaluator(self, eval_metric: str):
        r"""
        Add evaluator for the given metric.

        Parameters
        ----------
        eval_metric : str
            The evaluation metric.
        """
        if eval_metric not in evaluator_dict:
            raise ValueError(f"evaluator for {eval_metric} not found")
        self.evaluators[eval_metric] = evaluator_dict[eval_metric]()
    
    def update_evaluators(self, pred, label):
        r"""
        Update the evaluators with the given prediction and label.
        
        Parameters
        ----------
        pred : torch.Tensor
            The model's prediction.
        label : torch.Tensor
            The true label.
        """
        for evaluator in self.evaluators.values():
            evaluator(pred, label)
    
    def get_eval_results(self):
        r"""
        Get the evaluation results.

        Returns
        -------
        dict
            The evaluation results.
        """
        results = {}
        for metric, evaluator in self.evaluators.items():
            results.update(**evaluator.eval_status)
        return results

    def reset(self):
        r"""
        Reset the evaluators.
        """
        for evaluator in self.evaluators.values():
            evaluator.reset()
    
    def __getitem__(self, key):
        return self.evaluators[key]
    
    def __setitem__(self, key, value):
        self.evaluators[key] = value
    
    def __iter__(self):
        return iter(self.evaluators)
    
    def __len__(self):
        return len(self.evaluators)