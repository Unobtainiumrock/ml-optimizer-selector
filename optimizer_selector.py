"""
Optimizer/Solver Selection Utility

This module provides a function to automatically select the best optimization
method based on problem characteristics (dataset size, regularization, etc.)
"""

from typing import Literal, Optional, Dict, Any
from enum import Enum


class RegularizationType(Enum):
    """Types of regularization"""
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elasticnet"


class OptimizerRecommendation:
    """Container for optimizer recommendation with explanation"""
    
    def __init__(
        self, 
        solver: str, 
        reason: str, 
        sklearn_params: Optional[Dict[str, Any]] = None
    ):
        self.solver = solver
        self.reason = reason
        self.sklearn_params = sklearn_params or {}
    
    def __repr__(self) -> str:
        return (
            f"OptimizerRecommendation(\n"
            f"  solver='{self.solver}',\n"
            f"  reason='{self.reason}',\n"
            f"  sklearn_params={self.sklearn_params}\n"
            f")"
        )
    
    def __str__(self) -> str:
        params_str = ", ".join(
            f"{k}={repr(v)}" for k, v in self.sklearn_params.items()
        )
        return (
            f"Recommended Solver: {self.solver}\n"
            f"Reason: {self.reason}\n"
            f"Sklearn Parameters: {params_str if params_str else 'default'}"
        )


def select_optimizer(
    n_parameters: int,
    n_samples: int,
    regularization: Literal["none", "l1", "l2", "elasticnet"] = "none",
    dataset_in_memory: bool = True,
    is_multiclass: bool = False,
    max_memory_gb: Optional[float] = None
) -> OptimizerRecommendation:
    """
    Automatically select the best optimization method based on problem characteristics.
    
    This function implements a decision tree that considers:
    - Number of parameters (model complexity)
    - Dataset size and memory constraints
    - Regularization type (none, L1, L2, elastic net)
    - Whether the problem is multiclass
    
    Parameters
    ----------
    n_parameters : int
        Number of model parameters/features
    n_samples : int
        Number of training samples
    regularization : {"none", "l1", "l2", "elasticnet"}, default="none"
        Type of regularization to apply
    dataset_in_memory : bool, default=True
        Whether the entire dataset fits in memory
    is_multiclass : bool, default=False
        Whether this is a multiclass classification problem
    max_memory_gb : float, optional
        Maximum memory budget in GB. If specified, will check if L-BFGS
        memory requirements exceed this limit
    
    Returns
    -------
    OptimizerRecommendation
        Contains the recommended solver, explanation, and sklearn-compatible parameters
    
    Examples
    --------
    >>> # Small logistic regression with L2 regularization
    >>> rec = select_optimizer(n_parameters=100, n_samples=1000, regularization="l2")
    >>> print(rec)
    Recommended Solver: lbfgs
    Reason: Small-medium dataset with L2/no regularization. L-BFGS offers fast convergence.
    Sklearn Parameters: solver='lbfgs', penalty='l2'
    
    >>> # Large dataset with many features and L1 regularization
    >>> rec = select_optimizer(n_parameters=50000, n_samples=1000000, regularization="l1")
    >>> print(rec)
    Recommended Solver: saga
    Reason: Large number of parameters (>100k) requires memory-efficient method. SAGA supports L1.
    Sklearn Parameters: solver='saga', penalty='l1'
    
    >>> # Medium dataset that doesn't fit in memory
    >>> rec = select_optimizer(n_parameters=5000, n_samples=100000, 
    ...                        regularization="none", dataset_in_memory=False)
    >>> print(rec)
    Recommended Solver: sgd
    Reason: Dataset doesn't fit in memory. Use mini-batch SGD for out-of-core learning.
    Sklearn Parameters: None (use SGDClassifier/SGDRegressor)
    """
    
    # Convert regularization to enum for type safety
    reg_map = {
        "none": RegularizationType.NONE,
        "l1": RegularizationType.L1,
        "l2": RegularizationType.L2,
        "elasticnet": RegularizationType.ELASTIC_NET
    }
    reg_type = reg_map[regularization.lower()]
    
    # Check memory constraints if specified
    if max_memory_gb is not None:
        # L-BFGS needs approximately 2*m*n*8 bytes (m typically 10-20)
        m = 10  # typical L-BFGS memory parameter
        lbfgs_memory_gb = (2 * m * n_parameters * 8) / (1024**3)
        
        if lbfgs_memory_gb > max_memory_gb:
            if reg_type == RegularizationType.L1:
                return OptimizerRecommendation(
                    solver="saga",
                    reason=f"Memory constraint ({max_memory_gb}GB) exceeded. SAGA is memory-efficient and supports L1.",
                    sklearn_params={"solver": "saga", "penalty": "l1"}
                )
            else:
                return OptimizerRecommendation(
                    solver="sgd",
                    reason=f"Memory constraint ({max_memory_gb}GB) exceeded. Use mini-batch SGD.",
                    sklearn_params=None  # Use SGDClassifier instead
                )
    
    # Decision Tree Implementation
    
    # Branch 1: Large number of parameters (>100k)
    if n_parameters > 100_000:
        if reg_type == RegularizationType.L1:
            return OptimizerRecommendation(
                solver="saga",
                reason="Large number of parameters (>100k) requires memory-efficient method. SAGA supports L1.",
                sklearn_params={"solver": "saga", "penalty": "l1"}
            )
        elif reg_type == RegularizationType.ELASTIC_NET:
            return OptimizerRecommendation(
                solver="saga",
                reason="Large number of parameters (>100k). SAGA supports elastic net regularization.",
                sklearn_params={"solver": "saga", "penalty": "elasticnet"}
            )
        else:
            return OptimizerRecommendation(
                solver="sgd",
                reason="Large number of parameters (>100k). Mini-batch SGD is the only scalable option.",
                sklearn_params=None  # Use SGDClassifier/SGDRegressor
            )
    
    # Branch 2: Small-medium parameters (<100k)
    else:
        # Check if L1 regularization is needed
        if reg_type == RegularizationType.L1:
            return OptimizerRecommendation(
                solver="saga",
                reason="L1 regularization requires proximal methods. SAGA handles L1 efficiently.",
                sklearn_params={"solver": "saga", "penalty": "l1"}
            )
        
        elif reg_type == RegularizationType.ELASTIC_NET:
            return OptimizerRecommendation(
                solver="saga",
                reason="Elastic net regularization. SAGA supports both L1 and L2 components.",
                sklearn_params={"solver": "saga", "penalty": "elasticnet"}
            )
        
        # No L1 regularization - check if dataset fits in memory
        else:
            if not dataset_in_memory:
                return OptimizerRecommendation(
                    solver="sgd",
                    reason="Dataset doesn't fit in memory. Use mini-batch SGD for out-of-core learning.",
                    sklearn_params=None  # Use SGDClassifier/SGDRegressor
                )
            
            # Dataset fits in memory - L-BFGS is optimal!
            else:
                penalty = "l2" if reg_type == RegularizationType.L2 else None
                
                # Additional check: for very large datasets, SAG might be better
                if n_samples > 1_000_000:
                    return OptimizerRecommendation(
                        solver="sag",
                        reason="Very large dataset (>1M samples) in memory. SAG is optimized for this case.",
                        sklearn_params={"solver": "sag", "penalty": penalty}
                    )
                
                return OptimizerRecommendation(
                    solver="lbfgs",
                    reason="Small-medium dataset with L2/no regularization. L-BFGS offers fast convergence.",
                    sklearn_params={"solver": "lbfgs", "penalty": penalty}
                )


def get_sklearn_model(
    recommendation: OptimizerRecommendation,
    task: Literal["classification", "regression"] = "classification",
    **extra_kwargs: Any
):
    """
    Get the appropriate sklearn model based on the recommendation.
    
    Parameters
    ----------
    recommendation : OptimizerRecommendation
        The recommendation from select_optimizer()
    task : {"classification", "regression"}, default="classification"
        Type of machine learning task
    **extra_kwargs : Any
        Additional parameters to pass to the sklearn model
    
    Returns
    -------
    sklearn estimator
        Instantiated sklearn model with recommended parameters
    
    Examples
    --------
    >>> rec = select_optimizer(n_parameters=100, n_samples=1000, regularization="l2")
    >>> model = get_sklearn_model(rec, task="classification", max_iter=1000)
    >>> # Returns: LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000)
    """
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    
    # Merge recommendation params with extra kwargs
    params = {**recommendation.sklearn_params, **extra_kwargs}
    
    # Handle SGD-based solvers (they use different classes)
    if recommendation.solver == "sgd":
        if task == "classification":
            return SGDClassifier(**extra_kwargs)
        else:
            return SGDRegressor(**extra_kwargs)
    
    # Handle standard solvers
    if task == "classification":
        return LogisticRegression(**params)
    else:
        # For regression, map penalty to appropriate model
        penalty = params.get("penalty", "l2")
        solver = params.get("solver", "lbfgs")
        
        if penalty == "l1":
            # Lasso doesn't use solver parameter the same way
            return Lasso(**{k: v for k, v in extra_kwargs.items() if k != "solver"})
        elif penalty == "elasticnet":
            return ElasticNet(**{k: v for k, v in extra_kwargs.items() if k != "solver"})
        else:
            # L2 or none -> Ridge
            return Ridge(solver=solver, **extra_kwargs)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Optimizer Selection Examples")
    print("=" * 70)
    
    # Example 1: Small dataset, L2 regularization
    print("\n1. Small dataset with L2 regularization:")
    rec1 = select_optimizer(
        n_parameters=100,
        n_samples=1000,
        regularization="l2",
        dataset_in_memory=True
    )
    print(rec1)
    
    # Example 2: Large dataset with L1
    print("\n" + "=" * 70)
    print("2. Large parameter space with L1 regularization:")
    rec2 = select_optimizer(
        n_parameters=150_000,
        n_samples=50_000,
        regularization="l1",
        dataset_in_memory=True
    )
    print(rec2)
    
    # Example 3: Out-of-memory dataset
    print("\n" + "=" * 70)
    print("3. Dataset doesn't fit in memory:")
    rec3 = select_optimizer(
        n_parameters=5_000,
        n_samples=1_000_000,
        regularization="l2",
        dataset_in_memory=False
    )
    print(rec3)
    
    # Example 4: Very large in-memory dataset
    print("\n" + "=" * 70)
    print("4. Very large dataset that fits in memory:")
    rec4 = select_optimizer(
        n_parameters=10_000,
        n_samples=2_000_000,
        regularization="none",
        dataset_in_memory=True
    )
    print(rec4)
    
    # Example 5: Memory-constrained
    print("\n" + "=" * 70)
    print("5. Memory-constrained environment (max 1GB):")
    rec5 = select_optimizer(
        n_parameters=50_000,
        n_samples=100_000,
        regularization="l2",
        dataset_in_memory=True,
        max_memory_gb=1.0
    )
    print(rec5)
    
    print("\n" + "=" * 70)
    print("Getting actual sklearn models:")
    print("=" * 70)
    
    # Example: Get actual sklearn model
    rec = select_optimizer(n_parameters=500, n_samples=5000, regularization="l2")
    model = get_sklearn_model(rec, task="classification", max_iter=1000, random_state=42)
    print(f"\nModel: {model}")

