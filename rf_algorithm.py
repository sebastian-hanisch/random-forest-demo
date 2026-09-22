"""Random Forest: wie Bagging (bootstrap-demo) B volle Bäume auf Bootstrap-Stichproben, aber jeder Schnitt sieht nur `mtry` zufällig gezogene Merkmale statt aller. Das entkoppelt die Bäume auch dann, wenn ein
Merkmal alle anderen dominiert (siehe bagging-demo: dort blieb die Korrelation hoch). Der Baumkern (mit `mtry`) steht in `rf_tree.py`. `mtry = alle Merkmale` verhält sich exakt wie Bagging."""

from dataclasses import dataclass

import numpy as np

import rf_tree as T


@dataclass(frozen=True)
class Forest:
    trees: tuple                 # Tupel von T.Tree, ein Baum je Bootstrap-Stichprobe
    in_bag: np.ndarray            # (B, n) bool: True, wo die Zeile in dieser Stichprobe vorkam
    task: str
    n_train: int
    mtry: int
    seed: int
    n_features: int


def bootstrap_indices(n, seed):
    """n Zeilennummern mit Zurücklegen (eine Bootstrap-Stichprobe)."""
    return np.random.default_rng(seed).integers(0, n, n)


def _bootstrap_seed(seed, b):
    return seed * 1_000_003 + b


def _feature_seed(seed, b):
    """Eigener Zufalls-Strang für die Merkmalswahl je Baum, unabhängig vom Bootstrap-Seed desselben Baums."""
    return seed * 1_000_003 + b + 500_000_000


def fit(X, y, task, criterion=None, min_leaf=1, n_trees=30, mtry=None, seed=0):
    """Wächst n_trees volle Bäume auf unabhängigen Bootstrap-Stichproben; jeder Baum wählt an jedem Knoten `mtry` zufällige Merkmale (eigener Merkmals-Seed je Baum, unabhängig vom Bootstrap-Seed)."""
    n, d = len(y), X.shape[1]
    mtry_eff = d if mtry is None else min(int(mtry), d)
    trees, in_bag = [], np.zeros((n_trees, n), dtype=bool)
    for b in range(n_trees):
        idx = bootstrap_indices(n, _bootstrap_seed(seed, b))
        trees.append(T.grow(X[idx], y[idx], task, criterion, None, min_leaf, mtry=mtry_eff, seed=_feature_seed(seed, b)))
        in_bag[b, idx] = True
    return Forest(tuple(trees), in_bag, task, n, mtry_eff, seed, d)


def root_candidates(forest, b):
    """Welche `mtry` Merkmale der Baum b an der Wurzel zur Wahl hatte (der erste Zufallszug seines Merkmals-Strangs - die Wurzel ist immer der erste Knoten, den `grow` bearbeitet)."""
    if forest.mtry >= forest.n_features:
        return np.arange(forest.n_features)
    return np.sort(np.random.default_rng(_feature_seed(forest.seed, b)).choice(forest.n_features, forest.mtry, replace=False))


def _tree_values(forest, X, upto=None):
    trees = forest.trees[:upto] if upto else forest.trees
    return np.array([T.predict_value(t, X) for t in trees])


def predict_value(forest, X, upto=None):
    return _tree_values(forest, X, upto).mean(axis=0)


def predict(forest, X, upto=None):
    v = predict_value(forest, X, upto)
    return (v > 0.5).astype(int) if forest.task == "class" else v


def oob_predict(forest, X):
    """Für jede Trainingszeile der Mittelwert nur der Bäume, die sie nicht gesehen haben. NaN, wenn nie out-of-bag."""
    vals = _tree_values(forest, X)
    oob = ~forest.in_bag
    count = oob.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, (vals * oob).sum(axis=0) / np.maximum(count, 1), np.nan)
    return np.where(count > 0, mean, np.nan)


def scores_from_values(task, y, v):
    """Gütemaße aus Blattwerten `v` gegen `y`, wie in cart-demo/bagging-demo. NaN-Zeilen (nie OOB) werden ausgeschlossen."""
    ok = ~np.isnan(v)
    y, v = np.asarray(y)[ok], v[ok]
    if task == "class":
        pred = (v > 0.5).astype(int)
        return {"error": 1.0 - T.accuracy(y, pred), "accuracy": T.accuracy(y, pred), "auc": T.auc(y, v), "logloss": T.log_loss(y, v), "n": int(ok.sum())}
    return {"error": T.rmse(y, v), "rmse": T.rmse(y, v), "mae": T.mae(y, v), "r2": T.r2(y, v), "n": int(ok.sum())}


# --- Wald-Kennzahlen (Korrelation, Wurzeln) -------------------------------------------------------------------------------------------------------

def root_shares(forest):
    """Häufigkeit jedes Wurzelmerkmals über die Bäume, absteigend: [(Merkmal, Anzahl), ...]."""
    feats = [int(t.feature[0]) for t in forest.trees]
    vals, counts = np.unique(feats, return_counts=True)
    order = np.argsort(-counts)
    return [(int(vals[i]), int(counts[i])) for i in order]


def tree_correlation(forest, X):
    """Mittlere paarweise Ähnlichkeit der Einzelbaum-Vorhersagen auf X: Korrelationskoeffizient (Regression) bzw. Anteil gleicher Vorhersagen (Klassifikation). None bei nur einem Baum."""
    vals = _tree_values(forest, X)
    B = len(vals)
    if B < 2:
        return None
    if forest.task == "reg":
        c = np.corrcoef(vals)
        iu = np.triu_indices(B, k=1)
        return float(np.nanmean(c[iu]))
    preds = (vals > 0.5).astype(int)
    iu = np.triu_indices(B, k=1)
    agree = np.array([np.mean(preds[i] == preds[j]) for i, j in zip(*iu)])
    return float(agree.mean())


def importances_mean(forest):
    """Mittlere Gini-Wichtigkeit je Merkmal über alle Bäume (jeder Baum einzeln normiert)."""
    imps = np.array([T.importances(t) for t in forest.trees])
    return imps.mean(axis=0)


# --- Permutationswichtigkeit (Breiman, auf den OOB-Zeilen) ------------------------------------------------------------------------------------------

def _point_error(task, y, v):
    if task == "class":
        return 1.0 - T.accuracy(y, (v > 0.5).astype(int))
    return T.rmse(y, v)


def permutation_importance(forest, X, y, seed=0, min_oob=10):
    """Für jeden Baum: Fehler auf seinen OOB-Zeilen, dann je Merkmal diese Spalte unter den OOB-Zeilen mischen und den Fehler erneut messen - der Anstieg ist die Wichtigkeit dieses Merkmals für diesen Baum
    (Breiman 2001). Gemittelt über alle Bäume mit mindestens `min_oob` OOB-Zeilen. Rückgabe in denselben Einheiten wie der Fehler (Fehlerquote bzw. RMSE in Minuten) - anders als die Gini-Wichtigkeit NICHT auf Summe 1 normiert."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    task = forest.task
    per_tree = []
    for b, tree in enumerate(forest.trees):
        oob_idx = np.nonzero(~forest.in_bag[b])[0]
        if len(oob_idx) < min_oob:
            continue
        Xo, yo = X[oob_idx], y[oob_idx]
        base_err = _point_error(task, yo, T.predict_value(tree, Xo))
        deltas = np.empty(d)
        for j in range(d):
            Xp = Xo.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas[j] = _point_error(task, yo, T.predict_value(tree, Xp)) - base_err
        per_tree.append(deltas)
    return np.mean(per_tree, axis=0) if per_tree else np.zeros(d)


def normalize_importance(imp):
    """Für den direkten Vergleich mit der Gini-Wichtigkeit: negative Werte (Merkmal half rein zufällig) auf 0 gekappt, dann auf Summe 1 skaliert (0, wenn alles negativ war)."""
    clipped = np.clip(imp, 0.0, None)
    s = clipped.sum()
    return clipped / s if s > 0 else clipped
