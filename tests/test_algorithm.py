"""Random Forest gegen unabhängige Referenzen: mtry = alle Merkmale muss exakt Bagging ergeben (denselben Bootstrap-Formel, derselbe Baumkern ohne Einschränkung); scikit-learns RandomForestClassifier/Regressor
liefert (andere Zufallsquelle für Bootstrap und mtry) nur Rang-/Fehlerbänder, keine exakte Übereinstimmung. Der Baumkern selbst ist in cart-demo geprüft; hier geht es um `mtry`, Bootstrap-Orchestrierung, OOB und Wichtigkeit."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import rf_algorithm as rf
import rf_scenario as S
import rf_tree as T


def _continuous(n=400, d=6, seed=0, task="class", noise=1.6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    signal = X[:, 0] + 0.7 * np.sin(2 * X[:, 1]) + 0.5 * (X[:, 2] > 0.3) * X[:, 3]
    y = (signal + rng.normal(0, noise, n) > 0).astype(float) if task == "class" else signal * 3 + 10 + rng.normal(0, 1.0, n)
    return X, y


# --- mtry = alle Merkmale == Bagging ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["class", "reg"])
@pytest.mark.parametrize("mtry", [None, 6, 99])                          # None, genau d, und über d hinaus - alle drei müssen dasselbe ergeben
def test_mtry_covering_all_features_reproduces_the_unrestricted_tree(task, mtry):
    X, y = _continuous(300, 6, 1, task)
    full = T.grow(X, y, task, None, None, 5)
    restricted = T.grow(X, y, task, None, None, 5, mtry=mtry, seed=42)
    assert np.array_equal(full.feature, restricted.feature) and np.allclose(full.threshold[full.feature >= 0], restricted.threshold[restricted.feature >= 0])
    assert np.allclose(full.value, restricted.value)


def test_forest_with_mtry_none_equals_a_freshly_grown_forest_of_the_same_bootstrap_rows():
    """Kein Bagging-Import nötig: derselbe Bootstrap-Index (bootstrap_indices) und derselbe Baumkern (grow ohne Einschränkung) ergeben exakt dieselben Bäume wie ein Wald mit mtry = alle Merkmale."""
    X, y = _continuous(250, 5, 2, "class")
    forest = rf.fit(X, y, "class", None, 1, 8, mtry=None, seed=3)
    assert forest.mtry == 5
    for b, tree in enumerate(forest.trees):
        idx = rf.bootstrap_indices(len(y), 3 * 1_000_003 + b)
        ref = T.grow(X[idx], y[idx], "class", None, None, 1)                                        # kein mtry: die unveränderte cart-Regel
        assert np.array_equal(tree.feature, ref.feature) and np.allclose(tree.threshold[tree.feature >= 0], ref.threshold[ref.feature >= 0])


# --- mtry beschränkt die Split-Suche wirklich ---------------------------------------------------------------------------------------------------------

def test_mtry_restricts_the_search_to_the_drawn_candidates():
    X, y = _continuous(400, 8, 0, "class")
    gain, thr, _ = T.gain_matrix(X, y, "gini", 1)
    best_unrestricted = int(gain.max(axis=0).argmax())
    candidates = np.array([c for c in range(8) if c != best_unrestricted])[:3]                       # drei Merkmale ohne das insgesamt beste
    s = T.best_split(X, y, "gini", 1, candidates)
    assert s is not None and s[0] in candidates and s[0] != best_unrestricted


def test_mtry_one_still_grows_a_usable_tree_and_uses_many_features_over_many_nodes():
    """Mit mtry = 1 entscheidet an jedem Knoten der Zufall, WELCHES Merkmal überhaupt zur Wahl steht - über viele Knoten sollten trotzdem mehrere verschiedene Merkmale vorkommen."""
    X, y = _continuous(600, 6, 0, "class")
    tree = T.grow(X, y, "class", None, None, 5, mtry=1, seed=0)
    used = set(int(f) for f in tree.feature[tree.feature >= 0])
    assert len(used) >= 2 and tree.n_leaves > 1


def test_different_mtry_seeds_draw_different_candidate_sets():
    X, y = _continuous(300, 8, 0, "class")
    t1 = T.grow(X, y, "class", None, 2, 5, mtry=3, seed=1)
    t2 = T.grow(X, y, "class", None, 2, 5, mtry=3, seed=2)
    assert not np.array_equal(t1.feature, t2.feature) or not np.allclose(t1.threshold, t2.threshold, equal_nan=True)


# --- Kreuzprobe mit scikit-learn (Rang-/Fehlerbänder, andere Zufallsquelle) ----------------------------------------------------------------------------

@pytest.mark.parametrize("task,cls", [("class", RandomForestClassifier), ("reg", RandomForestRegressor)])
def test_forest_error_is_close_to_scikit_learns_random_forest(task, cls):
    ds = S.generate_dataset(800, 3, 0, 7)
    Xtr, ytr, Xte, yte = S.split(ds, task)
    d = Xtr.shape[1]
    mtry = int(round(np.sqrt(d))) if task == "class" else max(1, d // 3)
    forest = rf.fit(Xtr, ytr, task, None, 5, 60, mtry=mtry, seed=0)
    ref = cls(n_estimators=60, max_features=mtry, min_samples_leaf=5, random_state=0).fit(Xtr, ytr)
    v = rf.predict_value(forest, Xte)
    pred = (v > 0.5).astype(int) if task == "class" else v
    if task == "class":
        our_err = 1.0 - T.accuracy(yte, pred)
        ref_err = 1.0 - ref.score(Xte, yte)
    else:
        our_err = T.rmse(yte, pred)
        ref_err = float(np.sqrt(np.mean((ref.predict(Xte) - yte) ** 2)))
    assert our_err < 2.0 * ref_err + 0.05                                                            # grobes Band: dieselbe Größenordnung, keine exakte Übereinstimmung erwartet


# --- Bootstrap, OOB, Mittel (wie bagging-demo, hier nur die neuen Teile) --------------------------------------------------------------------------------

def test_oob_predict_matches_a_naive_loop():
    X, y = _continuous(150, 5, 4, "reg")
    forest = rf.fit(X, y, "reg", None, 1, 15, mtry=3, seed=2)
    mine = rf.oob_predict(forest, X)
    naive = np.full(len(y), np.nan)
    for i in range(len(y)):
        vals = [T.predict_value(t, X[i:i + 1])[0] for b, t in enumerate(forest.trees) if not forest.in_bag[b, i]]
        if vals:
            naive[i] = float(np.mean(vals))
    both = ~np.isnan(mine) & ~np.isnan(naive)
    assert both.sum() > 0.8 * len(y) and np.allclose(mine[both], naive[both])


def test_predict_upto_k_trees_matches_a_fresh_forest_of_k_trees():
    X, y = _continuous(200, 5, 0, "class")
    Xt, _ = _continuous(100, 5, 1, "class")
    full = rf.fit(X, y, "class", None, 1, 12, mtry=3, seed=5)
    for k in (1, 5, 12):
        small = rf.fit(X, y, "class", None, 1, k, mtry=3, seed=5)
        assert np.allclose(rf.predict_value(full, Xt, upto=k), rf.predict_value(small, Xt))


# --- Wichtigkeit: Gini gegen Permutation ---------------------------------------------------------------------------------------------------------------

def test_permutation_importance_is_near_zero_for_pure_noise_features():
    rng = np.random.default_rng(0)
    n, d_real, d_noise = 800, 3, 4
    Xr = rng.normal(size=(n, d_real))
    y = (Xr[:, 0] + 0.8 * Xr[:, 1] > 0).astype(float)
    Xn = rng.normal(size=(n, d_noise))
    X = np.column_stack([Xr, Xn])
    forest = rf.fit(X, y, "class", None, 5, 40, mtry=2, seed=0)
    imp = rf.permutation_importance(forest, X, y, seed=1)
    assert imp[:d_real].max() > imp[d_real:].max() + 0.02                                            # echte Merkmale klar wichtiger
    assert abs(imp[d_real:]).max() < 0.05                                                            # Rauschmerkmale: nahe 0 (positiv wie negativ)


def test_gini_importance_can_overrate_a_noise_feature_with_many_thresholds_relative_to_permutation():
    """Strobl et al. (2007): Gini-Wichtigkeit bevorzugt Merkmale mit vielen möglichen Schwellen. Ein stetiges Rauschmerkmal (viele Schwellen) bekommt bei kleinen Datensätzen messbare Gini-Wichtigkeit,
    obwohl die Permutationswichtigkeit (korrekt) nahe 0 bleibt."""
    rng = np.random.default_rng(3)
    n = 300
    real = rng.normal(size=n)
    y = (real > 0).astype(float)
    noise_continuous = rng.normal(size=n)                                                             # viele mögliche Schwellen
    noise_coarse = rng.integers(0, 2, n).astype(float)                                                 # nur eine mögliche Schwelle
    X = np.column_stack([real, noise_continuous, noise_coarse])
    forest = rf.fit(X, y, "class", None, 3, 40, mtry=1, seed=0)                                        # mtry klein: Rauschen kommt öfter zum Zug
    gini = rf.importances_mean(forest)
    perm = rf.permutation_importance(forest, X, y, seed=1)
    assert gini[1] > gini[2]                                                                           # stetiges Rauschen bekommt mehr Gini-Wichtigkeit als das grobe
    assert perm[1] < 0.03 and perm[2] < 0.03                                                           # beide Rauschmerkmale: Permutationswichtigkeit nahe 0


def test_normalize_importance_sums_to_one_or_zero():
    assert rf.normalize_importance(np.array([0.1, 0.2, -0.05])).sum() == pytest.approx(1.0)
    assert rf.normalize_importance(np.array([-0.1, -0.2])).sum() == pytest.approx(0.0)


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_mtry_larger_than_d_is_clamped():
    X, y = _continuous(100, 4, 0, "class")
    forest = rf.fit(X, y, "class", None, 1, 3, mtry=99, seed=0)
    assert forest.mtry == 4


def test_root_candidates_reconstructs_the_actual_draw_and_contains_the_chosen_root():
    X, y = _continuous(200, 6, 0, "class")
    forest = rf.fit(X, y, "class", None, 1, 8, mtry=3, seed=7)
    for b in range(8):
        cand = rf.root_candidates(forest, b)
        assert len(cand) == 3 and int(forest.trees[b].feature[0]) in cand
    forest_full = rf.fit(X, y, "class", None, 1, 3, mtry=None, seed=7)
    assert np.array_equal(rf.root_candidates(forest_full, 0), np.arange(6))                          # mtry = alle: jedes Merkmal ist Kandidat


def test_a_single_tree_forest_equals_that_tree():
    X, y = _continuous(120, 4, 0, "reg")
    forest = rf.fit(X, y, "reg", None, 1, 1, mtry=2, seed=0)
    assert np.allclose(rf.predict_value(forest, X), T.predict_value(forest.trees[0], X))
