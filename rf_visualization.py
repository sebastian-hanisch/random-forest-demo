"""Plotly-Darstellungen: einzelner Baum (mit Wurzel-Kandidaten), Karte des Wald-Mittels, mtry- und Wichtigkeits-Kurven. Alle Achsen sind gesperrt (Touch-Scrollen)."""

import numpy as np
import plotly.graph_objects as go

import rf_algorithm as rf
import rf_constants as C

CLASS_SCALE = [[0.0, "#2ca02c"], [0.5, "#f2e394"], [1.0, "#d62728"]]        # pünktlich (grün) -> zu spät (rot)
REG_SCALE = "Viridis"
NOISE = "#ff7f0e"


def lock_axes(fig, height=None, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=height, dragmode=False, **layout)
    return fig


def feature_label(names, f):
    unit = dict(C.FEATURES).get(names[f], "")
    return f"{names[f]} [{unit}]" if unit else names[f]


def value_text(task, v):
    return f"{v:.0%} zu spät" if task == "class" else f"{v:.0f} min"


def _scale(task):
    return CLASS_SCALE if task == "class" else REG_SCALE


# --- Ein einzelner Baum (klein), Wurzel-Kandidaten hervorgehoben -------------------------------------------------------------------------------------

def tree_layout(tree):
    x = np.zeros(tree.n_nodes)
    counter = 0
    stack = [(0, False)]
    while stack:
        t, done = stack.pop()
        if tree.feature[t] < 0:
            x[t] = counter
            counter += 1
        elif done:
            x[t] = (x[tree.left[t]] + x[tree.right[t]]) / 2.0
        else:
            stack += [(t, True), (int(tree.right[t]), False), (int(tree.left[t]), False)]
    return x, -tree.depth.astype(float)


def build_tree(tree, task, y_range, height=280):
    """Kleines Baumdiagramm ohne Beschriftung (Übersicht, nicht zum Ablesen)."""
    x, y = tree_layout(tree)
    inner = tree.feature >= 0
    fig = go.Figure()
    ex, ey = [], []
    for t in np.nonzero(inner)[0]:
        for c in (tree.left[t], tree.right[t]):
            ex += [x[t], x[c], None]
            ey += [y[t], y[c], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#9aa0a6", width=1), hoverinfo="skip", showlegend=False))
    vmin, vmax = (0.0, 1.0) if task == "class" else y_range
    size = 6 + 10 * np.sqrt(tree.n / tree.n_total)
    color = np.where(inner, np.nan, tree.value)
    fig.add_trace(go.Scatter(x=x[~inner], y=y[~inner], mode="markers", marker=dict(size=size[~inner], color=color[~inner], colorscale=_scale(task), cmin=vmin, cmax=vmax, line=dict(color="#111111", width=1)),
                             hovertext=[f"Blatt: {value_text(task, tree.value[t])}, n={tree.n[t]}" for t in np.nonzero(~inner)[0]], hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=x[inner], y=y[inner], mode="markers", marker=dict(size=size[inner], color="#ffffff", line=dict(color="#555555", width=1)),
                             hovertext=[f"Split {tree.feature[t]}" for t in np.nonzero(inner)[0]], hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return lock_axes(fig, height, plot_bgcolor="rgba(0,0,0,0)")


# --- Karte des Wald-Mittels ---------------------------------------------------------------------------------------------------------------------------

def build_map(forest, ds, fx, fy, upto=None, sample=None, height=430):
    task = forest.task
    Xtr = ds.X[ds.train]
    ytr = ds.y(task)[ds.train]
    med = np.median(Xtr, axis=0)
    gx = np.round(np.linspace(Xtr[:, fx].min(), Xtr[:, fx].max(), 60), 4)
    gy = np.round(np.linspace(Xtr[:, fy].min(), Xtr[:, fy].max(), 60), 4)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.tile(med, (XX.size, 1))
    grid[:, fx], grid[:, fy] = XX.ravel(), YY.ravel()
    z = np.round(rf.predict_value(forest, grid, upto), 3).reshape(XX.shape)
    vmin, vmax = (0.0, 1.0) if task == "class" else (float(np.min(ytr)), float(np.max(ytr)))
    scale = _scale(task)
    fig = go.Figure(go.Heatmap(x=gx, y=gy, z=z, colorscale=scale, zmin=vmin, zmax=vmax, opacity=0.55, showscale=False, hovertemplate="%{z:.2f}<extra></extra>"))
    if task == "class":
        fig.add_trace(go.Contour(x=gx, y=gy, z=z, contours=dict(start=0.5, end=0.5, size=1, coloring="none"), line=dict(color="#111111", width=2), showscale=False, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=np.round(Xtr[:, fx], 4), y=np.round(Xtr[:, fy], 4), mode="markers", marker=dict(size=5, color=ytr, colorscale=scale, cmin=vmin, cmax=vmax, line=dict(color="#333333", width=0.5)),
                             hovertemplate="%{x:.3g} / %{y:.3g}<extra></extra>", showlegend=False))
    if sample is not None:
        fig.add_trace(go.Scatter(x=[sample[fx]], y=[sample[fy]], mode="markers", marker=dict(symbol="star", size=16, color="#ffffff", line=dict(color="#111111", width=2)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title=feature_label(ds.names, fx))
    fig.update_yaxes(title=feature_label(ds.names, fy))
    return lock_axes(fig, height)


# --- Wichtigkeit: Gini gegen Permutation ----------------------------------------------------------------------------------------------------------------

def build_importance_pair(names, gini, perm, height=360):
    """Gini- und Permutationswichtigkeit nebeneinander, sortiert nach Gini; Rauschmerkmale orange."""
    order = np.argsort(-gini, kind="stable")
    colors_line = [NOISE if f >= C.N_BASE else "#333333" for f in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=gini[order], y=[names[f] for f in order], orientation="h", marker=dict(color="#1f77b4", line=dict(color=colors_line, width=1.5)), name="Gini", text=[f"{v:.1%}" for v in gini[order]], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(x=perm[order], y=[names[f] for f in order], orientation="h", marker=dict(color="#2ca02c", line=dict(color=colors_line, width=1.5)), name="Permutation", text=[f"{v:.1%}" for v in perm[order]], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Wichtigkeit (auf Summe 1 normiert)", tickformat=".0%", rangemode="tozero")
    return lock_axes(fig, height, barmode="group", legend=dict(orientation="h", y=1.1)).update_layout(margin=dict(l=10, r=60, t=30, b=10))


# --- Testfehler, OOB und Korrelation gegen mtry ---------------------------------------------------------------------------------------------------------

def _error_axis(fig, task):
    fig.update_yaxes(title="Fehlerquote" if task == "class" else "RMSE [min]", rangemode="tozero", **({"tickformat": ".0%"} if task == "class" else {}))


def build_mtry_curve(rows, task, current_mtry, height=380):
    m = [r["mtry"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m, y=[r["test"] for r in rows], mode="lines+markers", name="Testfehler", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=m, y=[r["oob"] for r in rows], mode="lines+markers", name="Out-of-Bag-Fehler", line=dict(color="#ff7f0e")))
    fig.add_vline(x=current_mtry, line=dict(color="#111111", dash="dot"))
    fig.update_xaxes(title="mtry (Merkmale je Split)", dtick=1)
    _error_axis(fig, task)
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


def build_mtry_correlation(rows, height=280):
    m = [r["mtry"] for r in rows]
    fig = go.Figure(go.Scatter(x=m, y=[r["correlation"] for r in rows], mode="lines+markers", line=dict(color="#1f77b4")))
    fig.update_xaxes(title="mtry (Merkmale je Split)", dtick=1)
    fig.update_yaxes(title="Baumkorrelation", range=[0, 1.03])
    return lock_axes(fig, height, showlegend=False)
