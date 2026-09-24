"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio, siehe bag_presets.py in bagging-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import rf_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


ALL_CRITERIA = ("gini", "entropy", "variance")
N_FEATURES_MAX = C.N_BASE + C.NOISE_MAX

SETTING_SPECS = {
    "task_select": SettingSpec("task", _choice(C.TASKS), C.DEFAULT_TASK),
    "criterion_select": SettingSpec("crit", _choice(ALL_CRITERIA), "gini"),
    "leaf_slider": SettingSpec("leaf", int, C.DEFAULT_LEAF, C.LEAF_MIN, C.LEAF_MAX),
    "n_trees_slider": SettingSpec("nt", int, C.DEFAULT_N_TREES, C.N_TREES_MIN, C.N_TREES_MAX),
    "mtry_slider": SettingSpec("mtry", int, 3, C.MTRY_MIN, N_FEATURES_MAX),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "n_noise_slider": SettingSpec("nn", int, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "label_noise_slider": SettingSpec("ln", int, C.DEFAULT_LABEL_NOISE, C.LABEL_NOISE_MIN, C.LABEL_NOISE_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "map_x_select": SettingSpec("fx", int, C.DEFAULT_MAP[0], 0, N_FEATURES_MAX - 1),
    "map_y_select": SettingSpec("fy", int, C.DEFAULT_MAP[1], 0, N_FEATURES_MAX - 1),
}
PRESET_KEYS = {"task": "task_select", "leaf": "leaf_slider", "n_trees": "n_trees_slider", "mtry": "mtry_slider", "n": "n_slider", "n_noise": "n_noise_slider",
               "label_noise": "label_noise_slider", "seed": "seed_input", "fx": "map_x_select", "fy": "map_y_select"}
# Regler, die je nach Einstellung ausgeblendet sind (Kriterium bei Regression, Etiketten-Rauschen bei Regression): der Wert bleibt hier erhalten
KEPT = {"criterion_select": "_criterion_kept", "label_noise_slider": "_label_noise_kept"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in KEPT and state_key not in st.session_state:       # ausblendbare Regler: siehe seed_widget
            st.session_state[state_key] = spec.default


def seed_widget(state_key):
    """Vor dem Zeichnen eines ausblendbaren Reglers: fehlt sein Zustand, kommt der zuletzt gewählte (oder der Standard-) Wert.
    Ein Wert, der in einem Lauf ohne den Regler in den Zustand des Reglers geschrieben wird, erscheint später als Mindestwert im Regler, während die App mit dem geschriebenen Wert rechnet."""
    if state_key not in st.session_state:
        st.session_state[state_key] = st.session_state.get(KEPT[state_key], SETTING_SPECS[state_key].default)


def stash_kept_widget_state():
    """Permalink und Preset legen den Wert eines ausblendbaren Reglers nur in KEPT ab (der Regler holt ihn sich mit `seed_widget`, sobald er gezeichnet wird)."""
    for state_key, kept in KEPT.items():
        if state_key in st.session_state:
            st.session_state[kept] = st.session_state.pop(state_key)


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
                if state_key in KEPT:
                    st.session_state[KEPT[state_key]] = value
            except (ValueError, TypeError):
                pass
    stash_kept_widget_state()
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = p[key]
    st.session_state["criterion_select"] = p["criterion"] if p["criterion"] in ("gini", "entropy") else st.session_state.get("_criterion_kept", "gini")
    for state_key, kept in KEPT.items():
        st.session_state[kept] = st.session_state[state_key]
    stash_kept_widget_state()


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
