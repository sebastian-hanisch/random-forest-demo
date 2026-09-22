"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Aufgabenwechsel, ausgeblendete Regler, mtry-Grenzen, Abspielen, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rf_constants as C
from rf_presets import KEPT, PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]
    at.session_state["criterion_select"] = p["criterion"] if p["criterion"] in ("gini", "entropy") else "gini"


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception_and_gives_one_verdict():
    at = _run()
    assert any("Random Forest in Aktion" in m.value for m in at.markdown) and not at.error
    assert any("Weniger korrelierte Bäume" in m.value for m in at.markdown)                          # Standard: mtry = 3, verdict decorrelated_no_gain


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert not at.error and not at.exception


def test_the_task_switch_hides_the_criterion_and_the_label_noise():
    at = _run()
    assert {"Split-Kriterium", "Falsche Etiketten im Training [%]"} <= _labels(at)
    at.session_state["task_select"] = "reg"
    at.run()
    assert not at.exception and not {"Split-Kriterium", "Falsche Etiketten im Training [%]"} & _labels(at)
    assert any("Varianz" in c.value for c in at.sidebar.caption) and any("nur bei der Klassifikation" in c.value for c in at.sidebar.caption)


def test_hidden_label_noise_comes_back_when_the_task_returns():
    # Die erste Sicht muss die Klassifikation sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run()
    at.session_state["label_noise_slider"] = 10
    at.run()
    at.session_state["task_select"] = "reg"
    at.run()
    at.session_state["task_select"] = "class"
    at.run()
    assert not at.exception and at.slider(key="label_noise_slider").value == 10


def test_mtry_slider_is_clamped_to_the_current_feature_count():
    at = _run()
    at.session_state["mtry_slider"] = 11
    at.run()
    assert at.slider(key="mtry_slider").value == 11
    at.session_state["n_noise_slider"] = 0
    at.run()
    assert not at.exception and at.slider(key="mtry_slider").value == 8                              # 8 Basismerkmale + 0 Rauschmerkmale = 8, mtry wird gekappt
    assert at.slider(key="mtry_slider").max == 8


def test_a_single_tree_shows_no_play_button_and_says_so():
    at = _run(lambda a: (a.session_state.__setitem__("n_trees_slider", 1)))
    assert not any(b.label == "▶️ Abspielen" for b in at.button)
    assert any("kein Mittel zu bilden" in i.value for i in at.info)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_slider"] = C.N_MIN
        at.session_state["leaf_slider"] = C.LEAF_MAX
        at.session_state["n_trees_slider"] = C.N_TREES_MIN
        at.session_state["n_noise_slider"] = 0
        at.session_state["mtry_slider"] = 1

    def big(at):
        at.session_state["n_slider"] = C.N_MAX
        at.session_state["leaf_slider"] = 1
        at.session_state["n_trees_slider"] = 50
        at.session_state["n_noise_slider"] = C.NOISE_MAX
        at.session_state["mtry_slider"] = C.N_BASE + C.NOISE_MAX
        at.session_state["label_noise_slider"] = C.LABEL_NOISE_MAX

    def reg(at):
        at.session_state["task_select"] = "reg"
        at.session_state["leaf_slider"] = 1
        at.session_state["mtry_slider"] = 1
    for setup in (small, big, reg):
        at = _run(setup)
        assert not at.exception


def test_map_features_beyond_the_columns_fall_back_after_fewer_noise_features():
    at = _run()
    at.session_state["map_x_select"] = 10
    at.run()
    at.session_state["n_noise_slider"] = 0
    at.run()
    assert not at.exception and at.selectbox(key="map_x_select").value == C.DEFAULT_MAP[0]


def test_the_test_delivery_slider_survives_a_smaller_data_set():
    at = _run()
    at.session_state["sample_slider"] = 300
    at.run()
    at.session_state["n_slider"] = C.N_MIN
    at.run()
    assert not at.exception and at.slider(key="sample_slider").value == 0


def test_step_slider_returns_to_the_full_forest_when_settings_change():
    at = _run()
    at.slider(key="rf_step").set_value(5)
    at.run()
    assert at.slider(key="rf_step").value == 5
    at.session_state["n_trees_slider"] = 50
    at.run()
    assert not at.exception and at.slider(key="rf_step").value == 50


def test_every_step_of_a_small_forest_renders():
    at = _run(lambda a: a.session_state.__setitem__("n_trees_slider", 6))
    for k in range(1, int(at.slider(key="rf_step").max) + 1):
        at.slider(key="rf_step").set_value(k)
        at.run()
        assert not at.exception, k


def test_play_renders_several_frames_without_duplicate_keys(monkeypatch):
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    at = _run(lambda a: a.session_state.__setitem__("n_trees_slider", 8))
    _play(at)
    assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_are_clamped_and_unknown_choices_fall_back():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["nt"] = "9999"
    at.query_params["leaf"] = "-4"
    at.query_params["task"] = "forest"
    at.query_params["crit"] = "variance"
    at.query_params["mtry"] = "-3"
    at.query_params["fx"] = "99"
    at.run()
    assert not at.exception
    assert at.slider(key="n_trees_slider").value == C.N_TREES_MAX and at.slider(key="leaf_slider").value == C.LEAF_MIN
    assert at.selectbox(key="task_select").value == C.DEFAULT_TASK and at.slider(key="mtry_slider").value == C.MTRY_MIN
    assert at.selectbox(key="criterion_select").value == "gini"                                     # "variance" gehört nicht zur Klassifikation


def test_the_address_bar_mirrors_the_settings():
    at = _run(lambda a: _apply(a, C.PRESETS["🎯 Bestes mtry"]))
    assert str(at.query_params["mtry"]) in ("4", "['4']") and str(at.query_params["task"]) in ("class", "['class']")


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Mittel über drei Datensätze" in c.value for c in at.caption)
    for key in ("mtry_start", "imp_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("Mittel über drei Datensätze", "Mittel über sechs Datensätze"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert sorted(keys) == sorted(["tree_chart", "map_chart", "importance_chart", "mtry_chart", "mtry_corr_chart", "imp_chart"]), keys
    looped = [c for c in calls if 'key=f"' in c]
    assert len(looped) == 2 and all('_{current}"' in c for c in looped)                              # die Bilder der Abspiel-Schleife tragen den Schritt
    viz = (ROOT / "rf_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("lock_axes(fig") >= 5


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_kept_values_cover_the_conditionally_hidden_controls():
    assert set(KEPT) == {"criterion_select", "label_noise_slider"}
