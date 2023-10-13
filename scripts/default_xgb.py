from xgboost import XGBRegressor


def default_xgb(**user_params):
    """
    Constructs an XGBoost regressor model equipped with default hyperparameters.

    Parameters
    ----------
    **user_params : dictionary
        Optional parameters to override the default settings.
        See the 'Notes' subsection for the default settings.

    Returns
    -------
    model : XGBRegressor
        Configured XGBoost regressor, where default hyperparameters are replaced by the ones from user_params if provided.

    Notes
    -----
    The default hyperparameters consist of:

    - n_estimators : integer, default=100
        The total number of boosting iterations.
    - learning_rate : float, default=0.3
        Step size for weight updates.
    - early_stopping_rounds : integer or None, default=None
        Trigger for early stopping. The evaluation metric must improve at least once in every 'early_stopping_rounds' iteration(s) for training to continue.
    - max_depth : integer, default=6
        Deepest level of any tree in the model.
    - min_child_weight : integer, default=1
        Minimum instance weight sum required for a child node.
    - subsample : float, default=1
        Fraction of total training data to be used in each boosting round.
    - colsample_bytree : float, default=1
        Fraction of features to choose for each boosting round.
    - gamma : float, default=0
        Minimum loss reduction needed for further leaf node splits.
    - reg_alpha : float, default=0
        L1 regularization on model weights.
    - reg_lambda : float, default=0
        L2 regularization on model weights.
    - tree_method : string, default='hist'
        Algorithm used for tree construction. 'hist' is recommended for better performance.
    - eval_metric : string, default='rmse'
        Metric utilized for assessing model quality on validation set.
    - booster : string, default='gbtree'
        Type of boosting model to use. 'gbtree' implies tree-based models.
    - random_state : integer or None, default=None
        Seed for random number generator.
    - n_jobs : integer, default=-1
        Count of CPU threads to use for running the model.

    Examples
    --------
    >>> model = standard_xgb_reg(n_estimators=200, max_depth=8)
    """
    default_params = {
        "n_estimators": 100,
        # Paso 1: Ajustar el learning rate.
        "learning_rate": 0.3,
        # Paso 2: Ajustar la profundidad máxima y el peso mínimo de las hojas.
        "max_depth": 6,
        "min_child_weight": 25,
        # Paso 3: Ajustar los parámetros de estocasticidad.
        "subsample": 1,
        "colsample_bytree": 1,
        # Paso 4: Ajustar los parámetros de regularización.
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 0,
        # General parameters
        "tree_method": "hist",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "random_state": None,
        "n_jobs": -1,
    }
    # Incorporate user-specified parameters into the default settings.
    default_params.update(user_params)
    model = XGBRegressor(**default_params)
    return model