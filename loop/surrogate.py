"""Surrogate model: predicts IPC for an unseen config from the runs so far.

PLACEHOLDER model choice (team may swap for a GP / random forest):
an additive "main effects" model. Predicted IPC = overall mean + one
correction per knob value. It assumes knobs act independently, which is
exactly the textbook assumption - so when it is wrong, the interaction
it missed is what a good rule should capture.

Uncertainty = spread of the residuals on the data it was fit on, inflated
for knob values it has never seen.
"""

import math

UNSEEN_VALUE_PENALTY = 0.05   # extra IPC std for a knob value with no data
MIN_STD = 0.01                # never pretend to be more certain than this


def fit(history, search_space, objective, priors=None):
    """history: list of {"knobs", "metrics"}. Returns the fitted model dict.
    priors (rule virtual points) are ignored by this simple model."""
    ipcs = []
    for entry in history:
        ipcs.append(entry["metrics"][objective])
    overall_mean = sum(ipcs) / len(ipcs)

    # effects[knob][value] = mean IPC of runs with that value, minus overall mean
    effects = {}
    for entry in history:
        for knob in entry["knobs"]:
            value = str(entry["knobs"][knob])
            effects.setdefault(knob, {}).setdefault(value, []).append(entry["metrics"][objective])
    for knob in effects:
        for value in effects[knob]:
            value_mean = sum(effects[knob][value]) / len(effects[knob][value])
            effects[knob][value] = value_mean - overall_mean

    model = {"mean": overall_mean, "effects": effects, "std": MIN_STD}

    squared_errors = []
    for entry in history:
        prediction, _ = predict(model, entry["knobs"])
        squared_errors.append((prediction - entry["metrics"][objective]) ** 2)
    residual_std = math.sqrt(sum(squared_errors) / len(squared_errors))
    model["std"] = max(residual_std, MIN_STD)
    return model


def predict(model, knobs):
    """Return (predicted ipc, std). Unseen knob values add uncertainty."""
    prediction = model["mean"]
    std = model["std"]
    for knob in knobs:
        value = str(knobs[knob])
        if value in model["effects"].get(knob, {}):
            prediction += model["effects"][knob][value]
        else:
            std += UNSEEN_VALUE_PENALTY
    return prediction, std


def probability_at_least(model, knobs, threshold):
    """P(ipc >= threshold) under a normal around the prediction: the surrogate's bet."""
    prediction, std = predict(model, knobs)
    z = (threshold - prediction) / std
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
