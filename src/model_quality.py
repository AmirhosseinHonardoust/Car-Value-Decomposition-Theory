from __future__ import annotations


def describe_r2(r2: float) -> str:
    """Return a short, honest, plain-language interpretation of an R^2 value.

    Centralizes this so train_model, evaluate_model, and the Streamlit app
    all surface the same warning automatically -- rather than relying on
    someone reading the README to realize a low or negative R^2 means the
    model has little or no real predictive signal.
    """
    if r2 >= 0.5:
        return (
            f"R^2 = {r2:.3f} -- the model explains a meaningful share of price "
            "variance. Decomposition output reflects a real, if imperfect, signal."
        )
    if r2 >= 0.0:
        return (
            f"R^2 = {r2:.3f} -- the model explains very little price variance. "
            "Treat predictions and decomposition output cautiously."
        )
    return (
        f"R^2 = {r2:.3f} -- the model performs WORSE than simply predicting the "
        "mean price for every car. This indicates little to no real "
        "relationship between price and the available features in this "
        "dataset. Treat all predictions and decomposition output as a "
        "demonstration of the method, not a real pricing signal."
    )
