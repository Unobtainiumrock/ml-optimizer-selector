"""
ML Optimizer Selector

Automatically select the optimal solver for machine learning optimization problems.
"""

from optimizer_selector import (
    select_optimizer,
    get_sklearn_model,
    OptimizerRecommendation,
    RegularizationType
)

__version__ = "1.0.0"
__all__ = [
    "select_optimizer",
    "get_sklearn_model",
    "OptimizerRecommendation",
    "RegularizationType"
]

