"""Messungen an Random Forest: Testfehler und Baumkorrelation gegen mtry, Gini- gegen Permutationswichtigkeit."""

import numpy as np
from dataclasses import dataclass

import rf_algorithm as rf
import rf_constants as C
import rf_scenario as S
import rf_tree as T

SIX = (C.DEFAULT_SEED,) + C.SWEEP_SEEDS
THREE = SIX[:3]                          # für den mtry-Sweep: voll gewachsene Regressionsbäume sind teuer (30 Bäume × 11 mtry × 6 Datensätze wäre > 1 Minute) - drei Datensätze und 20 Bäume genügen für den Trend
SWEEP_TREES = 20


def error_of(forest, X, y, upto=None):
    return rf.scores_from_values(forest.task, y, rf.predict_value(forest, X, upto))["error"]


def baseline_error(ds, task):
    _, ytr, _, yte = S.split(ds, task)
    if task == "class":
        return float(np.mean(yte != int(ytr.mean() > 0.5)))
    return float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))


def default_mtry(task, d):
    """Klassische Faustregel: Wurzel(d) bei Klassifikation, d/3 bei Regression (mindestens 1)."""
    return max(1, int(round(np.sqrt(d)))) if task == "class" else max(1, d // 3)


@dataclass
class Analysis:
    ds: object
    task: str
    criterion: str
    leaf: int
    n_trees: int
    forest: object
    single: object                # der erste Baum des Walds allein - wie in bagging-demo der Vergleichsmaßstab
    bagged: object                 # derselbe Wald mit mtry = alle Merkmale (= Bagging) - der zweite Vergleichsmaßstab
    train: dict
    test: dict
    single_test: dict
    bagged_test: dict
    oob: dict
    baseline: float
    verdict: str
    gini: np.ndarray
    perm: np.ndarray
    root_shares: list
    correlation: float
    bagged_correlation: float


def analyse(task, criterion, leaf, n_trees, mtry, n, n_noise, label_noise, seed):
    ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, seed)
    criterion = criterion if criterion in C.CRITERIA[task] else C.DEFAULT_CRITERION[task]
    Xtr, ytr, Xte, yte = S.split(ds, task)
    d = Xtr.shape[1]
    mtry = min(int(mtry), d)
    forest = rf.fit(Xtr, ytr, task, criterion, leaf, n_trees, mtry, seed=0)
    bagged = rf.fit(Xtr, ytr, task, criterion, leaf, n_trees, mtry=d, seed=0)                        # derselbe Bootstrap, aber ohne mtry-Einschränkung = Bagging
    single = forest.trees[0]
    train = rf.scores_from_values(task, ytr, rf.predict_value(forest, Xtr))
    test = rf.scores_from_values(task, yte, rf.predict_value(forest, Xte))
    single_test = rf.scores_from_values(task, yte, T.predict_value(single, Xte))
    bagged_test = rf.scores_from_values(task, yte, rf.predict_value(bagged, Xte))
    oob = rf.scores_from_values(task, ytr, rf.oob_predict(forest, Xtr))
    baseline = baseline_error(ds, task)
    gini = rf.importances_mean(forest)
    perm = rf.normalize_importance(rf.permutation_importance(forest, Xtr, ytr, seed=1))
    root_shares = rf.root_shares(forest)
    correlation = rf.tree_correlation(forest, Xte)
    bagged_correlation = rf.tree_correlation(bagged, Xte)
    a = Analysis(ds, task, criterion, leaf, n_trees, forest, single, bagged, train, test, single_test, bagged_test, oob, baseline, "", gini, perm, root_shares, correlation, bagged_correlation)
    a.verdict = verdict(a)
    return a


def verdict(a):
    """'stump' (ein Baum), 'worse' (schlechter als ein Einzelbaum - mtry zu klein, zu wenige Bäume), 'better' (spürbar besser als mtry = alle Merkmale, also besser als Bagging), 'decorrelated_no_gain'
    (die Bäume sind weniger korreliert, aber der Testfehler ist nicht besser als bei Bagging - z. B. mtry sehr klein), sonst 'similar' (mtry nahe an "alle Merkmale", praktisch Bagging)."""
    if a.n_trees <= 1:
        return "stump"
    if a.test["error"] > a.single_test["error"] * 1.01:
        return "worse"
    if a.test["error"] < a.bagged_test["error"] * 0.99:
        return "better"
    if a.correlation is not None and a.bagged_correlation is not None and a.correlation < a.bagged_correlation - 0.05:
        return "decorrelated_no_gain"
    return "similar"


# --- Testfehler und Baumkorrelation gegen mtry --------------------------------------------------------------------------------------------------------

def mtry_rows(task, criterion, leaf, n, n_noise, seeds=THREE, n_trees=SWEEP_TREES):
    """Mittel über drei Datensätze (je 20 Bäume - voll gewachsene Regressionsbäume sind teuer): je mtry der Testfehler und die Baumkorrelation."""
    ds0 = S.generate_dataset(n, n_noise, 0, seeds[0])
    d = ds0.X.shape[1]
    errs = {m: [] for m in range(1, d + 1)}
    corrs = {m: [] for m in range(1, d + 1)}
    oobs = {m: [] for m in range(1, d + 1)}
    for sd in seeds:
        ds = S.generate_dataset(n, n_noise, 0, sd)
        Xtr, ytr, Xte, yte = S.split(ds, task)
        for m in range(1, d + 1):
            forest = rf.fit(Xtr, ytr, task, criterion, leaf, n_trees, mtry=m, seed=0)
            errs[m].append(error_of(forest, Xte, yte))
            oobs[m].append(rf.scores_from_values(task, ytr, rf.oob_predict(forest, Xtr))["error"])
            c = rf.tree_correlation(forest, Xte)
            if c is not None:
                corrs[m].append(c)
    return [{"mtry": m, "test": float(np.mean(errs[m])), "oob": float(np.mean(oobs[m])), "correlation": float(np.mean(corrs[m])) if corrs[m] else None} for m in range(1, d + 1)]


# --- Gini gegen Permutationswichtigkeit ----------------------------------------------------------------------------------------------------------------

def importance_rows(task, criterion, leaf, n_trees, mtry, n, n_noise, seeds=SIX):
    """Mittel über sechs Datensätze: Anteil der Gini- bzw. Permutationswichtigkeit, der auf Rauschmerkmale entfällt."""
    ds0 = S.generate_dataset(n, n_noise, 0, seeds[0])
    n_real = C.N_BASE
    gini_share, perm_share = [], []
    gini_last, perm_last = None, None
    for sd in seeds:
        ds = S.generate_dataset(n, n_noise, 0, sd)
        Xtr, ytr, Xte, yte = S.split(ds, task)
        forest = rf.fit(Xtr, ytr, task, criterion, leaf, n_trees, mtry, seed=0)
        gini = rf.importances_mean(forest)
        perm = rf.normalize_importance(rf.permutation_importance(forest, Xtr, ytr, seed=1))
        gini_share.append(float(gini[n_real:].sum()) if n_noise else 0.0)
        perm_share.append(float(perm[n_real:].sum()) if n_noise else 0.0)
        gini_last, perm_last = gini, perm
    return {"gini_noise_share": float(np.mean(gini_share)), "perm_noise_share": float(np.mean(perm_share)), "gini_example": gini_last, "perm_example": perm_last, "names": ds0.names}
