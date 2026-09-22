"""Der CART-Baumkern (aus cart-demo übernommen, dort ausführlich gegen Brute-Force und scikit-learn geprüft) - für Random Forest um `mtry` erweitert: `best_split`/`grow` können die Schnittsuche an jedem Knoten auf eine
zufällige Teilmenge der Merkmale beschränken (`candidates`/`mtry`), sonst unverändert. Mit `mtry=None` verhält sich `grow` exakt wie in cart-demo/bagging-demo (Kreuzprobe: mtry = alle Merkmale == Bagging).

CART von Grund auf in numpy: gieriges Wachsen mit Schnittsuche über alle Schwellen aller Merkmale (Sortieren + kumulative Summen), Gini | Entropie | Varianz, Beschneiden nach Kosten-Komplexität,
Auswertung von Hand (Genauigkeit, AUC, Log-Loss, RMSE, MAE, R²). Bibliotheken kommen nur in den Tests als Gegenprobe vor.

Der Baum liegt in parallelen Feldern; die Knoten sind in Breitenreihenfolge nummeriert (Wurzel = 0, dann Ebene für Ebene). Deshalb ist "der Baum nach den ersten k Schnitten" einfach der Baum, in dem alle Knoten
ab dem k-ten inneren Knoten wieder Blätter sind. Ein Punkt geht nach links, wenn x[Merkmal] <= Schwelle (Schwelle = Mitte zwischen zwei benachbarten Werten)."""

from dataclasses import dataclass, replace

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Tree:
    feature: np.ndarray          # -1 = Blatt
    threshold: np.ndarray
    left: np.ndarray             # -1 bei Blättern
    right: np.ndarray
    value: np.ndarray            # Klassifikation: Anteil "1" im Knoten; Regression: Mittelwert
    impurity: np.ndarray
    n: np.ndarray                # Trainingsbeispiele im Knoten
    depth: np.ndarray
    task: str                    # "class" | "reg"
    criterion: str               # "gini" | "entropy" | "variance"
    n_total: int                 # Beispiele an der Wurzel (Gewichtung der Unreinheiten)
    n_features: int

    @property
    def n_nodes(self):
        return len(self.feature)

    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    @property
    def max_depth(self):
        return int(self.depth.max())

    def internal_nodes(self):
        return np.nonzero(self.feature >= 0)[0]


# --- Unreinheit und Schnittsuche ---------------------------------------------------------------------------------------------------------------------

def _impurity_from_p(p, criterion):
    if criterion == "gini":
        return 2.0 * p * (1.0 - p)
    q = np.clip(p, 1e-300, 1.0)
    r = np.clip(1.0 - p, 1e-300, 1.0)
    return -(p * np.log2(q) + (1.0 - p) * np.log2(r))


def node_impurity(y, criterion):
    """Unreinheit einer Beispielmenge: Gini 2p(1-p), Entropie in bit, Varianz um den Mittelwert."""
    if criterion == "variance":
        return float(np.mean((y - y.mean()) ** 2))
    return float(_impurity_from_p(np.asarray(y.mean()), criterion))


def gain_matrix(X, y, criterion, min_leaf):
    """Gewinn (Unreinheit des Knotens minus gewichtete Unreinheit der Kinder) für jede Schwelle jedes Merkmals.
    Rückgabe: (gain, thr, xs): Matrizen (m-1, d); gain = -inf, wo die Schwelle unzulässig ist (gleiche Werte, zu kleines Blatt)."""
    m, d = X.shape
    order = np.argsort(X, axis=0, kind="stable")
    xs = np.take_along_axis(X, order, axis=0)
    ys = y[order]
    nl = np.arange(1, m, dtype=float)[:, None]
    nr = m - nl
    if criterion == "variance":
        yc = ys - y.mean()                                                    # zentrieren gegen Auslöschung
        sl = np.cumsum(yc, axis=0)[:-1]
        tot = yc[:, 0].sum()
        sq = float((yc[:, 0] ** 2).sum())
        parent = sq / m - (tot / m) ** 2
        weighted = (sq - sl ** 2 / nl - (tot - sl) ** 2 / nr) / m
    else:
        cl = np.cumsum(ys, axis=0)[:-1]
        tot = ys[:, 0].sum()
        parent = float(_impurity_from_p(np.asarray(tot / m), criterion))
        weighted = (nl * _impurity_from_p(cl / nl, criterion) + nr * _impurity_from_p((tot - cl) / nr, criterion)) / m
    gain = parent - weighted
    ok = (xs[:-1] < xs[1:]) & (nl >= min_leaf) & (nr >= min_leaf)
    thr = (xs[:-1] + xs[1:]) / 2.0
    thr = np.where(thr >= xs[1:], xs[:-1], thr)                               # Rundung: die Schwelle bleibt links vom oberen Wert
    return np.where(ok, gain, -np.inf), thr, xs


def best_split(X, y, criterion, min_leaf, candidates=None):
    """(Merkmal, Schwelle, Gewinn) des besten Schnitts oder None. Bei Gleichstand gewinnt das kleinste Merkmal, dann die kleinste Schwelle.
    `candidates` (Random Forest): nur diese Spaltennummern werden geprüft - der Rest des Knotens sieht sie nicht. Ohne `candidates` wie in cart-demo: alle Merkmale."""
    if len(y) < 2 * min_leaf or len(y) < 2:
        return None
    Xc = X if candidates is None else X[:, candidates]
    gain, thr, _ = gain_matrix(Xc, y, criterion, min_leaf)
    per_feature = gain.argmax(axis=0)
    best_per = gain[per_feature, np.arange(gain.shape[1])]
    fc = int(best_per.argmax())
    if not np.isfinite(best_per[fc]):
        return None
    f = fc if candidates is None else int(candidates[fc])
    return f, float(thr[per_feature[fc], fc]), float(best_per[fc])


# --- Wachsen ---------------------------------------------------------------------------------------------------------------------------------

def grow(X, y, task="class", criterion=None, max_depth=None, min_leaf=1, mtry=None, seed=0):
    """Wächst den Baum Ebene für Ebene. Ein Knoten wird nicht geteilt, wenn er rein ist, die Tiefe erreicht ist oder kein zulässiger Schnitt existiert.
    `mtry` (Random Forest): an jedem Knoten werden nur `mtry` zufällig gezogene Merkmale geprüft (ohne Zurücklegen, ein frischer Zug je Knoten); `None` oder `mtry >= d` prüft wie in cart-demo/bagging-demo alle - dann ist das Ergebnis exakt dasselbe wie ohne `mtry`."""
    criterion = criterion or ("gini" if task == "class" else "variance")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    max_depth = 10 ** 6 if max_depth is None else int(max_depth)
    d = X.shape[1]
    use_mtry = mtry is not None and mtry < d
    rng = np.random.default_rng(seed) if use_mtry else None
    feature, threshold, left, right, value, imp, size, depth = [], [], [], [], [], [], [], []

    def new_node(idx, dep):
        yy = y[idx]
        feature.append(-1), threshold.append(np.nan), left.append(-1), right.append(-1)
        value.append(float(yy.mean())), imp.append(node_impurity(yy, criterion)), size.append(len(idx)), depth.append(dep)
        return len(feature) - 1

    members = {0: np.arange(len(y))}
    new_node(members[0], 0)
    queue = [0]
    while queue:
        t = queue.pop(0)
        idx = members.pop(t)
        if depth[t] >= max_depth or imp[t] <= EPS:
            continue
        candidates = np.sort(rng.choice(d, mtry, replace=False)) if use_mtry else None
        s = best_split(X[idx], y[idx], criterion, min_leaf, candidates)
        if s is None:
            continue
        f, thr, _ = s
        go_left = X[idx, f] <= thr
        lt, rt = new_node(idx[go_left], depth[t] + 1), new_node(idx[~go_left], depth[t] + 1)
        feature[t], threshold[t], left[t], right[t] = f, thr, lt, rt
        members[lt], members[rt] = idx[go_left], idx[~go_left]
        queue += [lt, rt]
    return Tree(np.array(feature), np.array(threshold), np.array(left), np.array(right), np.array(value), np.array(imp), np.array(size), np.array(depth), task, criterion, len(y), X.shape[1])


# --- Anwenden ---------------------------------------------------------------------------------------------------------------------------------

def apply(tree, X):
    """Blatt (Knotennummer) jedes Beispiels, vektorisiert Ebene für Ebene."""
    X = np.asarray(X, dtype=float)
    node = np.zeros(len(X), dtype=int)
    while True:
        idx = np.nonzero(tree.feature[node] >= 0)[0]
        if len(idx) == 0:
            return node
        cur = node[idx]
        go_left = X[idx, tree.feature[cur]] <= tree.threshold[cur]
        node[idx] = np.where(go_left, tree.left[cur], tree.right[cur])


def predict_value(tree, X):
    """Blattwert: Klassifikation = Anteil "1" (Wahrscheinlichkeit), Regression = Mittelwert."""
    return tree.value[apply(tree, X)]


def predict(tree, X):
    v = predict_value(tree, X)
    return (v > 0.5).astype(int) if tree.task == "class" else v


def decision_path(tree, x):
    """Knoten von der Wurzel bis zum Blatt für einen Punkt."""
    t, path = 0, [0]
    while tree.feature[t] >= 0:
        t = tree.left[t] if x[tree.feature[t]] <= tree.threshold[t] else tree.right[t]
        path.append(int(t))
    return path


def importances(tree):
    """Wichtigkeit je Merkmal: Summe der gewichteten Unreinheitsabnahmen aller Schnitte dieses Merkmals, auf Summe 1 normiert (alles 0 bei Wurzelblatt)."""
    imp = np.zeros(tree.n_features)
    for t in tree.internal_nodes():
        l, r = tree.left[t], tree.right[t]
        imp[tree.feature[t]] += (tree.n[t] * tree.impurity[t] - tree.n[l] * tree.impurity[l] - tree.n[r] * tree.impurity[r]) / tree.n_total
    s = imp.sum()
    return imp / s if s > 0 else imp


# --- Zusammenklappen: Schritt k des Wachsens und Beschneiden -------------------------------------------------------------------------------------

def collapse(tree, leaves):
    """Neuer Baum, in dem die Knoten in `leaves` zu Blättern werden (ihre Nachkommen entfallen); Knotennummern werden lückenlos neu vergeben, die Reihenfolge bleibt."""
    leaves = set(int(t) for t in leaves)
    keep, stack = [], [0]
    while stack:
        t = stack.pop()
        keep.append(t)
        if tree.feature[t] >= 0 and t not in leaves:
            stack += [int(tree.left[t]), int(tree.right[t])]
    keep = np.array(sorted(keep))
    new_id = {int(t): i for i, t in enumerate(keep)}
    is_leaf = np.array([tree.feature[t] < 0 or int(t) in leaves for t in keep])
    lf = np.where(is_leaf, -1, [new_id.get(int(tree.left[t]), -1) for t in keep])
    rt = np.where(is_leaf, -1, [new_id.get(int(tree.right[t]), -1) for t in keep])
    return replace(tree, feature=np.where(is_leaf, -1, tree.feature[keep]), threshold=np.where(is_leaf, np.nan, tree.threshold[keep]), left=lf, right=rt,
                   value=tree.value[keep], impurity=tree.impurity[keep], n=tree.n[keep], depth=tree.depth[keep])


def tree_after_splits(tree, k):
    """Der Baum nach den ersten k Schnitten (Breitenreihenfolge)."""
    inner = tree.internal_nodes()
    return collapse(tree, inner[int(k):])


def n_splits(tree):
    return int((tree.feature >= 0).sum())


def _subtree_stats(tree, is_leaf):
    """Für jeden Knoten: Blätter im Teilbaum und Summe der gewichteten Unreinheiten dieser Blätter (Knoten in `is_leaf` gelten als Blätter)."""
    m = tree.n_nodes
    leaves = np.ones(m)
    sub = (tree.n * tree.impurity / tree.n_total).copy()
    for t in range(m - 1, -1, -1):
        if not is_leaf[t] and tree.feature[t] >= 0:
            l, r = tree.left[t], tree.right[t]
            leaves[t] = leaves[l] + leaves[r]
            sub[t] = sub[l] + sub[r]
    return leaves, sub


def pruning_path(tree):
    """Beschneidungspfad nach Kosten-Komplexität. Immer wird der Teilbaum mit dem kleinsten effektiven alpha = (R(t) - R(T_t)) / (|Blätter(T_t)| - 1) zum Blatt gemacht,
    dabei ist R die mit n/N gewichtete Unreinheit. Rückgabe: Liste von (alpha, Blätter, Gesamt-Unreinheit der Blätter, geschnittener Knoten); Eintrag 0 ist der volle Baum mit alpha 0.
    Nach jedem Schnitt ändern sich nur die Vorfahren des Knotens; alpha wird nie kleiner als das vorige (wie in scikit-learn)."""
    m = tree.n_nodes
    is_leaf = tree.feature < 0
    cost = tree.n * tree.impurity / tree.n_total
    leaves, sub = _subtree_stats(tree, is_leaf)
    parent = _parents(tree)
    g = np.full(m, np.inf)
    inner = np.nonzero(~is_leaf)[0]
    g[inner] = (cost[inner] - sub[inner]) / (leaves[inner] - 1)
    path = [(0.0, int(leaves[0]), float(sub[0]), -1)]
    prev = 0.0
    while leaves[0] > 1:
        t = int(g.argmin())
        prev = max(prev, float(g[t]))
        g[t] = np.inf
        stack = [int(tree.left[t]), int(tree.right[t])]                        # der Teilbaum unter t verschwindet
        while stack:
            u = stack.pop()
            g[u] = np.inf
            if not is_leaf[u]:
                stack += [int(tree.left[u]), int(tree.right[u])]
        d_leaves, d_sub = leaves[t] - 1, sub[t] - cost[t]
        is_leaf[t] = True
        leaves[t], sub[t] = 1, cost[t]
        u = parent[t]
        while u >= 0:
            leaves[u] -= d_leaves
            sub[u] -= d_sub
            g[u] = (cost[u] - sub[u]) / (leaves[u] - 1)
            u = parent[u]
        path.append((prev, int(leaves[0]), float(sub[0]), t))
    return path


def _parents(tree):
    p = np.full(tree.n_nodes, -1)
    for t in tree.internal_nodes():
        p[tree.left[t]] = p[tree.right[t]] = t
    return p


def prune(tree, alpha, path=None):
    """Beschneidet, solange das kleinste effektive alpha <= alpha ist (wie ccp_alpha in scikit-learn). `path` = schon berechneter Pfad, spart die Neuberechnung."""
    path = path or pruning_path(tree)
    cut = [t for a, _, _, t in path[1:] if a <= alpha]
    return collapse(tree, cut) if cut else tree


def prune_to_leaves(tree, max_leaves, path=None):
    """Der Baum auf dem Beschneidungspfad mit den meisten Blättern, die höchstens `max_leaves` sind (mindestens ein Blatt)."""
    path = path or pruning_path(tree)
    k = next(i for i, (_, leaves, _, _) in enumerate(path) if leaves <= max_leaves)
    cut = [t for _, _, _, t in path[1:k + 1]]
    return collapse(tree, cut) if cut else tree


# --- Gütemaße von Hand -----------------------------------------------------------------------------------------------------------------------------

def accuracy(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def auc(y, score):
    """Fläche unter der ROC-Kurve über die Rangsumme (Gleichstände zählen halb)."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def log_loss(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def r2(y, pred):
    y = np.asarray(y, dtype=float)
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def scores(tree, X, y):
    """Alle Gütemaße der Aufgabe: Klassifikation (Fehlerquote, AUC, Log-Loss), Regression (RMSE, MAE, R²). `error` ist das eine Maß, das die Kurven zeigen."""
    v = predict_value(tree, X)
    if tree.task == "class":
        pred = (v > 0.5).astype(int)
        return {"error": 1.0 - accuracy(y, pred), "accuracy": accuracy(y, pred), "auc": auc(y, v), "logloss": log_loss(y, v)}
    return {"error": rmse(y, v), "rmse": rmse(y, v), "mae": mae(y, v), "r2": r2(y, v)}
