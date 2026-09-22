"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-22, Toleranzen fangen Rundung ab). `analyse()` ist deterministisch (bag.fit intern mit seed=0); die mtry- und Wichtigkeits-Experimente
sind Mittel über mehrere feste Seeds - keine Zufallsstreuung zwischen Testläufen."""

import functools

import numpy as np
import pytest

import rf_algorithm as rf
import rf_constants as C
import rf_evaluation as ev
import rf_scenario as S

PRESET = {"one": "🎲 mtry = 1", "rule": "📐 Faustregel (√d)", "best": "🎯 Bestes mtry", "all": "🌲🌳 mtry = alle", "reg": "📈 Regression"}


@functools.lru_cache(maxsize=None)
def _preset(key):
    p = C.PRESETS[PRESET[key]]
    return ev.analyse(p["task"], p["criterion"], p["leaf"], p["n_trees"], p["mtry"], p["n"], p["n_noise"], p["label_noise"], p["seed"])


def _help(key, *needles):
    text = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in text, (key, n)


@functools.lru_cache(maxsize=None)
def _default(task, mtry=3):
    return ev.analyse(task, None, 1, C.DEFAULT_N_TREES, mtry, C.DEFAULT_N, C.DEFAULT_NOISE, 0, C.DEFAULT_SEED)


# --- Preset-Hilfen --------------------------------------------------------------------------------------------------------------------------------

def test_mtry_one_preset():
    a = _preset("one")
    assert a.forest.mtry == 1 and a.verdict == "decorrelated_no_gain"
    assert (a.test["error"], a.bagged_test["error"], a.correlation, a.bagged_correlation) == pytest.approx((0.1917, 0.1500, 0.5975, 0.7725), abs=0.0005)
    _help("one", "1 von 11", "0.60", "0.77", "19.2 %", "15.0 %")


def test_rule_of_thumb_preset():
    a = _preset("rule")
    assert a.forest.mtry == 3 and a.test["error"] == pytest.approx(0.1528, abs=0.0005)
    _help("rule", "Wurzel(11)", "15.3 %")


def test_best_mtry_preset_beats_bagging():
    a = _preset("best")
    assert a.forest.mtry == 4 and a.verdict == "better"
    assert (a.test["error"], a.bagged_test["error"]) == pytest.approx((0.1472, 0.1500), abs=0.0005)
    _help("best", "mtry = 4", "5 und 8", "14.7 %", "15.0 %")


def test_all_features_preset_equals_bagging_exactly():
    a = _preset("all")
    assert a.forest.mtry == 11 and a.verdict == "similar"
    assert a.test["error"] == pytest.approx(a.bagged_test["error"]) and a.correlation == pytest.approx(a.bagged_correlation)
    assert (a.test["error"], a.correlation) == pytest.approx((0.1500, 0.7725), abs=0.0005)                                   # dieselben Zahlen wie bagging-demos Standardfall
    _help("all", "11 = alle Merkmale", "0.77", "15.0 %")


def test_regression_preset_beats_bagging_slightly():
    a = _preset("reg")
    assert a.task == "reg" and a.forest.mtry == 6 and a.verdict == "better"
    assert (a.test["error"], a.bagged_test["error"], a.single_test["error"]) == pytest.approx((10.1039, 10.2796, 16.8344), abs=0.005)
    _help("reg", "mtry = 6", "10.1 min", "10.3 min", "17.0 min")


def test_every_preset_is_a_valid_setting():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS and p["criterion"] in C.CRITERIA[p["task"]] and C.LEAF_MIN <= p["leaf"] <= C.LEAF_MAX and C.N_TREES_MIN <= p["n_trees"] <= C.N_TREES_MAX
        d = C.N_BASE + p["n_noise"]
        assert 1 <= p["mtry"] <= d and C.N_MIN <= p["n"] <= C.N_MAX and 0 <= p["fx"] < d and 0 <= p["fy"] < d and name in C.PRESET_HELP


# --- Standardansicht (mtry = 3, entspricht der Faustregel) -----------------------------------------------------------------------------------------

def test_default_view_numbers():
    a = _default("class")
    assert (a.test["error"], a.oob["error"], a.baseline) == pytest.approx((0.1528, 0.1631, 0.4639), abs=0.0005)
    assert a.correlation == pytest.approx(0.6914, abs=0.0005) and a.bagged_correlation == pytest.approx(0.7725, abs=0.0005)
    ar = _default("reg")
    assert (ar.test["error"], ar.baseline) == pytest.approx((12.1046, 27.1543), abs=0.005)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------------

def test_mtry_sweep_numbers_class():
    rows = ev.mtry_rows("class", None, 1, C.DEFAULT_N, C.DEFAULT_NOISE)
    assert [r["mtry"] for r in rows] == list(range(1, 12))
    tests = [r["test"] for r in rows]
    corrs = [r["correlation"] for r in rows]
    assert tests == pytest.approx([0.1694, 0.1648, 0.1454, 0.1444, 0.1389, 0.1444, 0.1528, 0.1435, 0.1528, 0.1509, 0.1546], abs=0.001)
    assert corrs == pytest.approx([0.613, 0.661, 0.713, 0.728, 0.741, 0.761, 0.768, 0.775, 0.782, 0.782, 0.788], abs=0.002)
    best = min(rows, key=lambda r: r["test"])
    assert best["mtry"] == 5 and best["test"] < rows[0]["test"] and best["test"] < rows[-1]["test"]
    assert corrs[0] < corrs[-1]                                                                                                # Korrelation steigt monoton in der Tendenz mit mtry


def test_mtry_sweep_numbers_reg():
    rows = ev.mtry_rows("reg", None, 1, C.DEFAULT_N, C.DEFAULT_NOISE)
    tests = [r["test"] for r in rows]
    assert tests == pytest.approx([16.979, 13.806, 12.115, 10.728, 10.695, 10.135, 10.144, 9.914, 10.195, 10.146, 10.215], abs=0.03)
    best = min(rows, key=lambda r: r["test"])
    assert best["mtry"] == 8 and rows[-1]["correlation"] == pytest.approx(0.801, abs=0.003)


@pytest.mark.parametrize("task,nn,gini_share,perm_share", [("class", 3, 0.1236, 0.0091), ("class", 8, 0.2723, 0.0312), ("reg", 3, 0.0914, 0.0082), ("reg", 8, 0.2137, 0.0271)])
def test_gini_overrates_noise_features_relative_to_permutation(task, nn, gini_share, perm_share):
    r = ev.importance_rows(task, None, 1, C.DEFAULT_N_TREES, 3, C.DEFAULT_N, nn)
    assert (r["gini_noise_share"], r["perm_noise_share"]) == pytest.approx((gini_share, perm_share), abs=0.004)
    assert r["gini_noise_share"] > 3 * r["perm_noise_share"]                                                                   # der Kernbefund: Gini überschätzt Rauschen um ein Vielfaches
    assert r["perm_noise_share"] < 0.05                                                                                        # Permutation bleibt nahe 0


def test_more_noise_features_increase_the_gini_bias():
    r3 = ev.importance_rows("class", None, 1, C.DEFAULT_N_TREES, 3, C.DEFAULT_N, 3)
    r8 = ev.importance_rows("class", None, 1, C.DEFAULT_N_TREES, 3, C.DEFAULT_N, 8)
    assert r8["gini_noise_share"] > r3["gini_noise_share"]


# --- mtry = alle == Bagging (auf Auswertungsebene, nicht nur im Baum) --------------------------------------------------------------------------------

def test_forest_with_mtry_equal_to_feature_count_matches_bagging_for_every_mtry_setting():
    """analyse() rechnet 'bagged' immer separat mit mtry = alle - das muss unabhängig vom eingestellten mtry immer dieselben Zahlen liefern."""
    ds_bagged = [ev.analyse("class", None, 1, 30, m, 1200, 3, 0, 7).bagged_test["error"] for m in (1, 4, 7, 11)]
    assert all(v == pytest.approx(ds_bagged[0]) for v in ds_bagged)


# --- Erzeuger (geteilt mit cart-demo/bagging-demo) ---------------------------------------------------------------------------------------------------

def test_generator_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 11) and ds.names[:2] == ("Distanz", "Ladegewicht")
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150


def test_default_mtry_heuristic():
    assert ev.default_mtry("class", 11) == 3 and ev.default_mtry("reg", 11) == 3
    assert ev.default_mtry("class", 4) == 2 and ev.default_mtry("reg", 4) == 1
