"""Unabhängige Orakel für den Baumkern und die Wald-Kennzahlen (Regressionstest der Orakelprüfung).

1. Brute Force: jeder Split des gewachsenen Baums (auch mit `mtry`-Kandidaten) hat denselben Gain wie das Maximum einer Schleife über alle Schwellen direkt aus den Zeilen. Gleichstände zwischen
   Merkmalen lösen Bibliotheken verschieden auf - deshalb wird der Gain verglichen, nicht die Wahl des Merkmals.
2. Gini-Wichtigkeit aus den Zeilen von Hand; Beschneidungspfad und Gütemaße gegen scikit-learn (Pfad nur, wo sklearn denselben Baum baut)."""

import numpy as np
import pytest
from sklearn import metrics as M
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import rf_algorithm as rf
import rf_tree as T


def _imp(y, crit):
    if crit == "variance":
        return float(np.mean((y - y.mean()) ** 2))
    p = y.mean()
    if crit == "gini":
        return 1 - p ** 2 - (1 - p) ** 2
    return -sum(q * np.log2(q) for q in (p, 1 - p) if q > 0)


def _brute_best_gain(X, y, crit, min_leaf, cand=None):
    m = len(y)
    best = None
    for f in (range(X.shape[1]) if cand is None else cand):
        for v in np.unique(X[:, f])[:-1]:
            left = X[:, f] <= v
            nl = int(left.sum())
            if nl < min_leaf or m - nl < min_leaf:
                continue
            g = _imp(y, crit) - (nl * _imp(y[left], crit) + (m - nl) * _imp(y[~left], crit)) / m
            best = g if best is None or g > best else best
    return best


def _data(rng, n, d, task):
    X = rng.integers(0, 6, (n, d)).astype(float) if rng.random() < 0.5 else rng.normal(size=(n, d))        # ganzzahlig = viele Gleichstände
    y = (rng.random(n) < 0.4).astype(float) if task == "class" else rng.normal(size=n) * 3 + X[:, 0]
    return X, y


def test_every_split_is_optimal_and_leaf_values_and_impurities_are_exact():
    rng = np.random.default_rng(5)
    for it in range(60):
        task = "class" if it % 3 else "reg"
        crit = str(rng.choice(["gini", "entropy"])) if task == "class" else "variance"
        n, d, ml = int(rng.integers(6, 50)), int(rng.integers(1, 5)), int(rng.choice([1, 1, 2, 5]))
        X, y = _data(rng, n, d, task)
        tree = T.grow(X, y, task, crit, None, ml)
        members = {0: np.arange(n)}
        for t in range(tree.n_nodes):
            idx = members[t]
            yy = y[idx]
            assert tree.n[t] == len(idx) and tree.value[t] == pytest.approx(yy.mean(), abs=1e-12)
            assert tree.impurity[t] == pytest.approx(_imp(yy, crit), abs=1e-9)
            if tree.feature[t] >= 0:
                left = X[idx, tree.feature[t]] <= tree.threshold[t]
                assert left.sum() >= ml and (~left).sum() >= ml
                members[int(tree.left[t])], members[int(tree.right[t])] = idx[left], idx[~left]
                gain = _imp(yy, crit) - (left.sum() * _imp(yy[left], crit) + (~left).sum() * _imp(yy[~left], crit)) / len(idx)
                assert gain == pytest.approx(_brute_best_gain(X[idx], yy, crit, ml), abs=1e-9)
            elif len(idx) >= 2 * ml and _imp(yy, crit) > 1e-12:
                assert _brute_best_gain(X[idx], yy, crit, ml) is None, "Blatt trotz zulässigem Split"


def test_best_split_with_candidates_only_looks_at_the_drawn_features_and_is_optimal_among_them():
    rng = np.random.default_rng(6)
    for it in range(120):
        task = "class" if it % 3 else "reg"
        crit = "gini" if task == "class" else "variance"
        n, d = int(rng.integers(6, 40)), int(rng.integers(2, 6))
        X, y = _data(rng, n, d, task)
        cand = np.sort(rng.choice(d, int(rng.integers(1, d + 1)), replace=False))
        ml = int(rng.choice([1, 2]))
        s = T.best_split(X, y, crit, ml, cand)
        b = _brute_best_gain(X, y, crit, ml, cand)
        if s is None:
            assert b is None or b <= 1e-12
            continue
        f, th, g = s
        left = X[:, f] <= th
        assert f in cand
        assert g == pytest.approx(b, abs=1e-9)
        assert g == pytest.approx(_imp(y, crit) - (left.sum() * _imp(y[left], crit) + (~left).sum() * _imp(y[~left], crit)) / n, abs=1e-9)


def test_gini_importance_matches_a_hand_computation_from_the_rows():
    rng = np.random.default_rng(7)
    for it in range(40):
        task = "class" if it % 2 else "reg"
        crit = "gini" if task == "class" else "variance"
        n, d = int(rng.integers(15, 80)), int(rng.integers(2, 5))
        X, y = _data(rng, n, d, task)
        tree = T.grow(X, y, task, crit, None, 1)
        members, imp = {0: np.arange(n)}, np.zeros(d)
        for t in range(tree.n_nodes):
            idx = members[t]
            if tree.feature[t] >= 0:
                lm = X[idx, tree.feature[t]] <= tree.threshold[t]
                l, r = idx[lm], idx[~lm]
                members[int(tree.left[t])], members[int(tree.right[t])] = l, r
                imp[tree.feature[t]] += (len(idx) * _imp(y[idx], crit) - len(l) * _imp(y[l], crit) - len(r) * _imp(y[r], crit)) / n
        ref = imp / imp.sum() if imp.sum() > 0 else imp
        assert np.allclose(T.importances(tree), ref, atol=1e-9)


def _same_structure(tree, sk, t=0, u=0):
    a, b = tree.feature[t], sk.tree_.feature[u]
    if (a < 0) != (b < 0):
        return False
    if a < 0:
        return True
    if a != b or abs(tree.threshold[t] - sk.tree_.threshold[u]) > 1e-5:
        return False
    return _same_structure(tree, sk, tree.left[t], sk.tree_.children_left[u]) and _same_structure(tree, sk, tree.right[t], sk.tree_.children_right[u])


def test_pruning_path_matches_scikit_learn_where_both_grow_the_same_tree():
    rng = np.random.default_rng(99)
    checked = 0
    for it in range(80):
        task = "class" if it % 2 else "reg"
        n, d, ml = int(rng.integers(15, 90)), int(rng.integers(2, 5)), int(rng.choice([1, 3, 5]))
        X = rng.normal(size=(n, d))
        sig = X[:, 0] + 0.5 * X[:, 1]
        y = (sig + rng.normal(size=n) > 0).astype(float) if task == "class" else sig * 3 + rng.normal(size=n)
        cls = DecisionTreeClassifier if task == "class" else DecisionTreeRegressor
        tree = T.grow(X, y, task, None, None, ml)
        sk = cls(min_samples_leaf=ml, random_state=0).fit(X, y)
        if not _same_structure(tree, sk):                                                              # Gleichstand zwischen Merkmalen: anderer, ebenso guter Baum
            continue
        checked += 1
        cp = sk.cost_complexity_pruning_path(X, y)
        assert sorted(set(round(a, 9) for a, _, _, _ in T.pruning_path(tree))) == sorted(set(round(float(a), 9) for a in cp.ccp_alphas))
        for a in cp.ccp_alphas[1:-1:2]:
            assert T.prune(tree, a + 1e-10).n_leaves == cls(min_samples_leaf=ml, random_state=0, ccp_alpha=a + 1e-10).fit(X, y).get_n_leaves()
    assert checked >= 15


def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(3)
    for it in range(150):
        n = int(rng.integers(3, 40))
        y = (rng.random(n) < 0.5).astype(int)
        s = np.round(rng.random(n), 1) if it % 2 else rng.random(n)                                    # gerundet = Gleichstände in der AUC
        if y.min() != y.max():
            assert T.auc(y, s) == pytest.approx(M.roc_auc_score(y, s), abs=1e-12)
            p = np.clip(s, 0.001, 0.999)
            assert T.log_loss(y, p) == pytest.approx(M.log_loss(y, p), abs=1e-12)
        yr, pr = rng.normal(size=n), rng.normal(size=n)
        assert T.rmse(yr, pr) == pytest.approx(np.sqrt(M.mean_squared_error(yr, pr)), abs=1e-12)
        assert T.mae(yr, pr) == pytest.approx(M.mean_absolute_error(yr, pr), abs=1e-12)
        assert T.r2(yr, pr) == pytest.approx(M.r2_score(yr, pr), abs=1e-12)


def test_tree_correlation_and_root_shares_match_a_naive_computation():
    rng = np.random.default_rng(8)
    for it in range(25):
        task = "class" if it % 2 else "reg"
        X, y = _data(rng, 120, 4, task)
        Xt = rng.normal(size=(60, 4))
        f = rf.fit(X, y, task, None, 3, int(rng.integers(2, 7)), int(rng.integers(1, 5)), seed=int(rng.integers(0, 50)))
        vals = [T.predict_value(t, Xt) for t in f.trees]
        pairs = []
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if task == "class":
                    pairs.append(np.mean((vals[i] > .5) == (vals[j] > .5)))
                elif vals[i].std() > 0 and vals[j].std() > 0:
                    pairs.append(np.corrcoef(vals[i], vals[j])[0, 1])
        got = rf.tree_correlation(f, Xt)
        assert (np.isnan(got) if not pairs else got == pytest.approx(np.mean(pairs), abs=1e-9))
        counts = {}
        for t in f.trees:
            counts[int(t.feature[0])] = counts.get(int(t.feature[0]), 0) + 1
        assert sorted(rf.root_shares(f)) == sorted(counts.items())
