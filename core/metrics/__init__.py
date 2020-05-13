from .loss import *
from .boundary import boundary_evaluation
from .mask import mask_IoU, class_evaluation
from .metric_logger import *


def metric_wrapper(metric_fn, name, **kwargs):
    def wrapper(output, target):
        prediction = torch.argmax(output, dim=1)
        results = metric_fn(prediction, target, **kwargs)
        named_results = {f"{name}_{k}": v for k,v in results.items()}
        return named_results
    return wrapper