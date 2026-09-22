"""Random Forest - Bagging mit zufälliger Merkmalsteilmenge je Split - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Random Forest - und lässt stattdessen das Beispiel wachsen.
Drittes Stück der Baumbasierten Linie der "Konzepte"-Reihe: der Nachfolger von Bagging (bagging-demo). Dort blieben die Bäume korreliert, wenn ein Merkmal jeden Baum dominierte - Random Forest
lässt an jedem Split nur eine zufällige Teilmenge der Merkmale zur Wahl (mtry) und entkoppelt die Bäume dadurch, ohne die Daten anzufassen.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import rf_algorithm as rf
import rf_constants as C
import rf_evaluation as ev
from rf_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from rf_visualization import (
    build_importance_pair,
    build_map,
    build_mtry_correlation,
    build_mtry_curve,
    build_tree,
    feature_label,
    value_text,
)

st.set_page_config(page_title="Random Forest – Sebastian Hanisch", layout="wide")

VERDICT_TEXT = {
    "stump": "ℹ️ Nur ein Baum: kein Wald, kein Vergleich möglich.",
    "worse": "⚠️ **Schlechter als ein Einzelbaum** - mtry oder die Zahl der Bäume sind hier zu klein.",
    "better": "✅ **Besser als mtry = alle Merkmale (= Bagging)** - die Entkopplung durch mtry lohnt sich hier.",
    "decorrelated_no_gain": "ℹ️ **Weniger korrelierte Bäume, aber kein besserer Testfehler als bei Bagging** - Dekorrelation allein genügt nicht, wenn die Splits dadurch zu schwach werden.",
    "similar": "➖ **Kaum Unterschied zu Bagging** - mtry ist hier nahe an der vollen Merkmalszahl.",
}


def _err(task, x):
    return f"{x:.1%}" if task == "class" else f"{x:.1f} min"


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(*params):
    return ev.analyse(*params)


@st.cache_data(show_spinner=False, max_entries=6)
def _mtry_rows(task, criterion, leaf, n, n_noise):
    return ev.mtry_rows(task, criterion, leaf, n, n_noise)


@st.cache_data(show_spinner=False, max_entries=6)
def _importance_rows(task, criterion, leaf, n_trees, mtry, n, n_noise):
    return ev.importance_rows(task, criterion, leaf, n_trees, mtry, n, n_noise)


st.title("🌳🎲 Random Forest – Bagging mit zufälliger Merkmalsteilmenge")
st.markdown(
    """
Bagging (bagging-demo) mittelt viele Bäume auf Bootstrap-Stichproben - das senkt die Varianz stark, **solange sich die Bäume unterscheiden**. Dominiert ein Merkmal jeden Baum, bleiben sie trotz
unterschiedlicher Stichproben korreliert, und Mitteln hilft weniger (gemessen in bagging-demo). **Random Forest** (Breiman 2001) behebt das an der Wurzel des Problems: an **jedem** Split jedes Baums
darf nur eine zufällige Teilmenge von **mtry** Merkmalen überhaupt zur Wahl stehen - selbst das insgesamt stärkste Merkmal fehlt an vielen Knoten einfach. Das entkoppelt die Bäume, ohne die Daten zu ändern.
Als Nebenprodukt gibt es eine zweite Wichtigkeit: **Permutationswichtigkeit** auf den Out-of-Bag-Zeilen, die - anders als die Gini-Wichtigkeit - Rauschmerkmale nicht systematisch überschätzt.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - drittes Stück der Baumbasierten Linie der \"Konzepte\"-Reihe, Nachfolger von Bagging (bagging-demo) - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Verfahren geht auf Breiman (2001) zurück; alle Lieferungen, Merkmale und Zahlen dieser Demo sind erzeugt und gemessen - keine echten Daten. Der Baumkern (`rf_tree.py`) ist aus cart-demo übernommen und um `mtry` erweitert; scikit-learn kommt nur in den Tests als Gegenprobe vor."
)
st.caption(
    "**Bezug zu OR:** die Permutationswichtigkeit zeigt, welche Größen eine Tourenplanung wirklich beeinflussen (z. B. Ladegewicht, Verkehr) statt nur zufällig hoch bewertet zu werden - ein Filter, bevor man ein Merkmal in ein Optimierungsmodell aufnimmt."
)

with st.expander("So funktioniert Random Forest", expanded=True):
    st.markdown(
        """
1. **Wie Bagging:** B Bootstrap-Stichproben (mit Zurücklegen), auf jeder ein voller CART-Baum.
2. **Der Unterschied:** an jedem Knoten wird zuerst eine zufällige Teilmenge von **mtry** der d Merkmale gezogen (ohne Zurücklegen, ein frischer Zug je Knoten) - nur unter diesen wird der beste Split gesucht. Ein Merkmal, das an diesem Knoten nicht gezogen wurde, kommt dort nicht in Frage, egal wie stark es wäre.
3. **mtry = alle Merkmale** macht daraus wieder exakt Bagging (Kreuzprobe, siehe Verifikation) - **mtry = 1** macht jeden Split fast zufällig.
4. **Mitteln und Out-of-Bag:** wie bei Bagging unverändert.
5. **Permutationswichtigkeit:** für jeden Baum den Fehler auf seinen Out-of-Bag-Zeilen messen, dann eine Spalte unter diesen Zeilen mischen und den Fehler erneut messen - der Anstieg ist die Wichtigkeit dieses Merkmals für diesen Baum, gemittelt über alle Bäume.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
if st.session_state["criterion_select"] not in ("gini", "entropy"):
    st.session_state["criterion_select"] = "gini"

with st.sidebar:
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASKS, key="task_select", format_func=lambda k: C.TASK_LABELS[k],
                        help="Klassifikation: gemittelte Wahrscheinlichkeit, dass die Lieferung zu spät kommt. Regression: gemittelte Dauer in Minuten.")
    if task == "class":
        crit = st.selectbox("Split-Kriterium", C.CRITERIA["class"], key="criterion_select", format_func=lambda k: C.CRITERION_LABELS[k])
        st.session_state[KEPT["criterion_select"]] = crit
    else:
        crit = "variance"
        st.caption("Split-Kriterium: Varianz - bei einem Zahlenziel gibt es keine Wahl.")
    leaf = st.slider("Mindestgröße eines Blatts", *bounds("leaf_slider"), key="leaf_slider", help="Wie in bagging-demo: 1 = Bäume wachsen voll (klassisch), kein Beschneiden.")
    n_trees = st.slider("Zahl der Bäume", *bounds("n_trees_slider"), key="n_trees_slider", help="Wie in bagging-demo: der Testfehler sättigt ab einigen Dutzend Bäumen.")
    n_noise = st.slider("Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", help="Zusätzliche Merkmale ohne Bezug zum Ziel (wie in cart-demo).")
    d_now = C.N_BASE + int(n_noise)
    mtry_hi = min(bounds("mtry_slider")[1], d_now)
    if st.session_state["mtry_slider"] > mtry_hi:
        st.session_state["mtry_slider"] = mtry_hi
    mtry = st.slider(f"mtry (Merkmale je Split, von {d_now})", 1, mtry_hi, key="mtry_slider",
                     help=f"Wie viele der {d_now} Merkmale an jedem Split überhaupt zur Wahl stehen. 1 = fast zufällig (schwach, aber entkoppelt), {d_now} = alle Merkmale = Bagging. "
                          "Auf dem Standarddatensatz liegt das gemessene Optimum zwischen 5 und 8 (Klassifikation) bzw. um 8 (Regression) - siehe Experiment.")
    if task == "class":
        label_noise = st.slider("Falsche Etiketten im Training [%]", *bounds("label_noise_slider"), key="label_noise_slider", help="Anteil vertauschter Trainingsetiketten; der Test bleibt sauber.")
        st.session_state[KEPT["label_noise_slider"]] = label_noise
    else:
        label_noise = int(st.session_state.get(KEPT["label_noise_slider"], C.DEFAULT_LABEL_NOISE))
        st.caption("Falsche Etiketten gibt es nur bei der Klassifikation.")
    st.markdown("**Daten**")
    n = st.slider("Lieferungen", *bounds("n_slider"), key="n_slider", step=100, help="Zahl der erzeugten Lieferungen; 70 % zum Lernen, 30 % zum Testen.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Erzeugt einen anderen Datensatz mit denselben Regeln.")
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed.")

base_params = (task, crit, int(leaf), int(n_trees), int(mtry))
data_params = (int(n), int(n_noise), int(label_noise), int(seed))
with st.spinner("Rechne ..."):
    a = _analysis(*base_params, *data_params)
ds = a.ds
Xtr, ytr, Xte, yte = ds.X[ds.train], ds.y(task)[ds.train], ds.X[ds.test], (ds.y_true if task == "class" else ds.y_reg)[ds.test]
names = ds.names
n_feat = len(names)
n_test = len(ds.test)

with st.sidebar:
    st.markdown("**Ansicht**")
    for key, default in (("map_x_select", C.DEFAULT_MAP[0]), ("map_y_select", C.DEFAULT_MAP[1])):
        if st.session_state[key] >= n_feat:
            st.session_state[key] = default
    fx = st.selectbox("Karte: waagerecht", range(n_feat), key="map_x_select", format_func=lambda f: feature_label(names, f))
    fy = st.selectbox("Karte: senkrecht", range(n_feat), key="map_y_select", format_func=lambda f: feature_label(names, f))
    if st.session_state.get("sample_slider", 0) > n_test - 1:
        st.session_state["sample_slider"] = 0
    sample_idx = st.slider("Testlieferung", 0, n_test - 1, 0, key="sample_slider")
sync_query_params({"task_select": task, "criterion_select": crit if task == "class" else st.session_state.get(KEPT["criterion_select"], "gini"), "leaf_slider": int(leaf), "n_trees_slider": int(n_trees),
                   "mtry_slider": int(mtry), "n_slider": int(n), "n_noise_slider": int(n_noise), "label_noise_slider": int(label_noise), "seed_input": int(seed), "map_x_select": int(fx), "map_y_select": int(fy)})

view_key = (base_params, data_params)
if st.session_state.get("rf_owner") != view_key:
    st.session_state["rf_owner"] = view_key
    st.session_state["rf_step"] = int(n_trees)

# --- Random Forest in Aktion ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Random Forest in Aktion")
st.caption("Der Wald wächst Baum für Baum; links der zuletzt hinzugekommene Einzelbaum mit seinem Wurzel-Merkmal, rechts das Wald-Mittel über zwei Merkmale.")
if n_trees > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Bäume im Mittel", 1, int(n_trees), key="rf_step")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 1, False
    st.info("ℹ️ Nur ein Baum eingestellt - kein Mittel zu bilden. Mehr Bäume in der Seitenleiste zeigen den Effekt.")
view_slot = st.empty()
sample_x = Xte[sample_idx]
sample_y = yte[sample_idx]


def _render(current):
    sub = rf.Forest(a.forest.trees[:current], a.forest.in_bag[:current], task, a.forest.n_train, a.forest.mtry, a.forest.seed, a.forest.n_features)
    latest = a.forest.trees[current - 1]
    cand = rf.root_candidates(a.forest, current - 1)
    with view_slot.container():
        c1, c2 = st.columns([2, 3])
        with c1:
            st.plotly_chart(build_tree(latest, task, (float(ytr.min()), float(ytr.max()))), width="stretch", key=f"tree_chart_{current}")
            if int(mtry) < n_feat:
                st.caption(f"Baum {current}: an der Wurzel standen **{', '.join(names[c] for c in cand)}** zur Wahl, gewählt **{names[latest.feature[0]]}** ({latest.n_leaves} Blätter).")
            else:
                st.caption(f"Baum {current}: Wurzel = **{names[latest.feature[0]]}** (alle Merkmale standen zur Wahl), {latest.n_leaves} Blätter.")
        with c2:
            st.plotly_chart(build_map(sub, ds, fx, fy, upto=None, sample=sample_x), width="stretch", key=f"map_chart_{current}")
        pred_here = rf.predict_value(sub, sample_x.reshape(1, -1))[0]
        truth = ("zu spät" if sample_y == 1 else "pünktlich") if task == "class" else f"{sample_y:.0f} min"
        st.markdown(f"**Testlieferung {sample_idx}:** Mittel der ersten {current} Bäume = **{value_text(task, pred_here)}**; tatsächlich: **{truth}**.")


if auto_play:
    frames = sorted(set(np.unique(np.round(np.linspace(1, int(n_trees), min(12, int(n_trees)))).astype(int))))
    for kk in frames:
        _render(kk)
        time.sleep(min(0.9, 6.0 / len(frames)))
    step = int(n_trees)
else:
    _render(step)

st.markdown("---")

# --- Was der Wald gelernt hat -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Was der Wald gelernt hat – und wie gut er auf neuen Lieferungen ist")
st.caption("Vergleichsmaßstäbe: **ein Einzelbaum** (der erste Baum dieses Walds) und **mtry = alle Merkmale** (derselbe Bootstrap, aber ohne mtry-Einschränkung = Bagging).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("mtry / Bäume", f"{int(mtry)} / {int(n_trees)}", help=f"Merkmale je Split von {n_feat} insgesamt, und Zahl der Bäume.")
m2.metric("Testfehler", _err(task, a.test["error"]), delta=f"1 Baum: {_err(task, a.single_test['error'])}", delta_color="off")
m3.metric("Out-of-Bag-Fehler", _err(task, a.oob["error"]), delta=f"Raten: {_err(task, a.baseline)}", delta_color="off")
gap = a.test["error"] - a.bagged_test["error"]
gap_text = f"{gap:+.1%} ggü. hier" if task == "class" else f"{gap:+.1f} min ggü. hier"
m4.metric("mtry = alle (Bagging)", _err(task, a.bagged_test["error"]), delta=gap_text, delta_color="inverse", help="Testfehler mit mtry = alle Merkmale auf demselben Bootstrap - der direkte Bagging-Vergleich.")
st.markdown(VERDICT_TEXT[a.verdict])
if a.correlation is not None:
    st.caption(f"Baumkorrelation: **{a.correlation:.2f}** (mit mtry = alle: {a.bagged_correlation:.2f}). {a.root_shares[0][1]} von {int(n_trees)} Bäumen beginnen mit **{names[a.root_shares[0][0]]}**" +
               (f", der Rest verteilt sich auf {len(a.root_shares) - 1} weitere Merkmale." if len(a.root_shares) > 1 else "."))

st.markdown("**Wichtigkeit: Gini gegen Permutation**")
st.plotly_chart(build_importance_pair(names, a.gini, a.perm), width="stretch", key="importance_chart")
st.caption("Gini-Wichtigkeit (blau) summiert die Unreinheitsabnahme aller Splits eines Merkmals - sie bevorzugt Merkmale mit vielen möglichen Schwellen, auch wenn sie nur Rauschen sind. Permutationswichtigkeit (grün) mischt "
           "eine Spalte unter den Out-of-Bag-Zeilen und misst den Anstieg des Fehlers - näher an der tatsächlichen Vorhersagekraft (Experiment unten). Orange umrandet: Rauschmerkmale.")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Testfehler und Baumkorrelation gegen mtry")
if st.button("mtry von 1 bis alle Merkmale durchprobieren (dauert einen Moment)", key="mtry_start"):
    st.session_state["mtry_on"] = True
if st.session_state.get("mtry_on"):
    with st.spinner("Wachse Wälder für jedes mtry auf drei Datensätzen ..."):
        mrows = _mtry_rows(task, crit, int(leaf), int(n), int(n_noise))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_mtry_curve(mrows, task, int(mtry)), width="stretch", key="mtry_chart")
    c1.plotly_chart(build_mtry_correlation(mrows), width="stretch", key="mtry_corr_chart")
    c2.table({"mtry": [r["mtry"] for r in mrows], "Testfehler": [_err(task, r["test"]) for r in mrows], "Korrelation": [f"{r['correlation']:.2f}" for r in mrows]})
    best = min(mrows, key=lambda r: r["test"])
    worst_corr, best_corr = mrows[0], mrows[-1]
    st.caption(f"Mittel über drei Datensätze, 20 Bäume je Wald (volle Regressionsbäume sind teuer - für den Trend genügt das). Die Korrelation steigt von {worst_corr['correlation']:.2f} (mtry 1) auf {best_corr['correlation']:.2f} (mtry {best_corr['mtry']}, = Bagging). "
               f"Der Testfehler hat sein Minimum bei **mtry {best['mtry']}** ({_err(task, best['test'])}) - klar besser als mtry 1 ({_err(task, mrows[0]['test'])}) und etwas besser als mtry {mrows[-1]['mtry']} ({_err(task, mrows[-1]['test'])}). "
               "Die Kurve ist flach in der Mitte: die genaue Wahl von mtry ist weniger wichtig als überhaupt zu dekorrelieren.")

st.markdown("---")

st.subheader("🔬 Gini gegen Permutationswichtigkeit: der Rauschmerkmal-Test")
if st.button("Anteil der Wichtigkeit auf Rauschmerkmalen messen (dauert einen Moment)", key="imp_start"):
    st.session_state["imp_on"] = True
if st.session_state.get("imp_on"):
    with st.spinner("Berechne Gini- und Permutationswichtigkeit auf sechs Datensätzen ..."):
        irows = _importance_rows(task, crit, int(leaf), int(n_trees), int(mtry), int(n), int(n_noise))
    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric("Gini-Anteil auf Rauschen", f"{irows['gini_noise_share']:.1%}", help=f"Mittel über sechs Datensätze mit {int(n_noise)} Rauschmerkmalen von {n_feat} insgesamt.")
        st.metric("Permutations-Anteil auf Rauschen", f"{irows['perm_noise_share']:.1%}", delta=f"{irows['perm_noise_share'] - irows['gini_noise_share']:+.1%}", delta_color="inverse")
    with c2:
        st.plotly_chart(build_importance_pair(irows["names"], irows["gini_example"], irows["perm_example"]), width="stretch", key="imp_chart")
    st.caption(f"Mittel über sechs Datensätze: bei {int(n_noise)} Rauschmerkmalen bekommt die Gini-Wichtigkeit **{irows['gini_noise_share']:.1%}** ihrer Summe auf reine Rauschmerkmale verteilt - obwohl die nichts mit dem Ziel zu tun haben. "
               f"Die Permutationswichtigkeit gibt ihnen nur **{irows['perm_noise_share']:.1%}**. Der rechte Ausschnitt zeigt ein Beispiel: Gini (blau) bewertet Rauschen sichtbar über Null, Permutation (grün) bleibt nahe Null. "
               "Grund: mehr mögliche Schwellen (stetige Merkmale) geben mehr Gelegenheit für einen zufällig guten Split - Gini zählt das als Wichtigkeit mit, Permutation nicht (Strobl u. a. 2007).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **mtry braucht einen Mittelweg** | Zu klein (mtry = 1): jeder Split ist fast zufällig, die Bäume sind entkoppelt, aber einzeln schwach - der Testfehler kann schlechter sein als bei Bagging (gemessen oben). Zu groß (mtry = alle): wieder Bagging, mit seiner Korrelation. | Experiment / Kreuzvalidierung für mtry |
| **Voraussetzung ist, dass mehrere Merkmale etwas taugen** | Gibt es nur ein wirklich informatives Merkmal und der Rest ist Rauschen, hilft auch ein kleines mtry nicht - die zufällige Teilmenge trifft es oft gar nicht. | mehr Bäume, mehr echte Merkmale |
| **Gini-Wichtigkeit bleibt verzerrt, auch gemittelt** | Sie bevorzugt Merkmale mit vielen möglichen Schwellen (stetige Merkmale, hohe Kardinalität) - das ist ein Artefakt der Split-Suche, kein Verdienst des Merkmals (gemessen oben). | Permutationswichtigkeit |
| **Permutationswichtigkeit ist teurer und selbst nicht perfekt** | Sie braucht für jeden Baum und jedes Merkmal eine zusätzliche Auswertung auf den OOB-Zeilen; bei stark korrelierten Merkmalen (zwei tragen dieselbe Information) teilen sie sich die Wichtigkeit und wirken beide weniger wichtig, als sie zusammen sind. | bedingte Wichtigkeit (Literatur, hier nicht gebaut) |
| **Immer noch Stufenfunktionen** | Wie CART und Bagging liefert jeder Einzelbaum eine Treppe; das Mitteln glättet sie, ersetzt aber keine echten Geraden oder schrägen Grenzen. | Boosting-Ast der Linie |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Random Forest.** Wie Bagging: $B$ Bäume $T_1,\dots,T_B$ auf Bootstrap-Stichproben, $\hat f(x)=\frac1B\sum_b T_b(x)$. Der einzige Unterschied: an jedem Knoten $t$ mit Merkmalsmenge $\{1,\dots,d\}$ wird vor der Split-Suche eine
Teilmenge $M_t\subset\{1,\dots,d\}$, $|M_t|=\texttt{mtry}$, ohne Zurücklegen gezogen; der Gain $\Delta(j,s)$ (wie in cart-demo) wird nur für $j\in M_t$ maximiert. Für $\texttt{mtry}=d$ ist $M_t=\{1,\dots,d\}$ immer - identisch mit Bagging.

**Warum das dekorreliert.** Sei $\rho$ die mittlere Korrelation zwischen zwei Bäumen, $\sigma^2$ die Varianz eines Baums. Für das Mittel von $B$ Bäumen gilt (Bagging-Formel, siehe bagging-demo)
$$\mathrm{Var}(\hat f) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.$$
Kleineres mtry senkt $\rho$ (jeder Baum sieht andere Merkmale an jedem Knoten), erhöht aber typischerweise $\sigma^2$ je Baum (schwächere Splits) - das Produkt hat ein Minimum, das gemessene "beste mtry".

**Permutationswichtigkeit** (Breiman 2001) für Merkmal $j$ und Baum $b$: mit $\mathrm{OOB}_b$ den Zeilen, die Baum $b$ nicht sah, $e_b=\mathrm{Fehler}(T_b,\mathrm{OOB}_b)$ und $e_b^{(j)}$ demselben Fehler, nachdem Spalte $j$ unter $\mathrm{OOB}_b$
zufällig gemischt wurde: $\mathrm{Imp}(j) = \frac1B\sum_b \left(e_b^{(j)} - e_b\right)$. Ein Merkmal ohne Vorhersagekraft hat $\mathbb E[\mathrm{Imp}(j)]\approx 0$ (Mischen ändert am Fehler im Mittel nichts); ein
Merkmal, dessen Wegfall den Baum schlechter macht, hat $\mathrm{Imp}(j)>0$.

Implementiert in `rf_tree.py` (Baumkern mit `mtry`), `rf_algorithm.py` (Bootstrap, Mitteln, OOB, Permutationswichtigkeit), `rf_evaluation.py` (Analyse, mtry-Sweep, Wichtigkeits-Experiment).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
