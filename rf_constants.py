"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

# Merkmale der Lieferungen: (Name, Einheit) - wortgleich aus cart-demo
FEATURES = [("Distanz", "km"), ("Ladegewicht", "kg"), ("Stopps", ""), ("Verkehr", "0-1"), ("Wetter", "0-1"), ("Wochentag", "0 = Mo"), ("Zeitfenster-Enge", "0-1"), ("Fahrerjahre", "Jahre")]
N_BASE = len(FEATURES)

TASKS = ("class", "reg")
TASK_LABELS = {"class": "Klassifikation: kommt die Lieferung zu spät?", "reg": "Regression: wie lange dauert die Lieferung?"}
DEFAULT_TASK = "class"
CRITERIA = {"class": ("gini", "entropy"), "reg": ("variance",)}
CRITERION_LABELS = {"gini": "Gini-Unreinheit", "entropy": "Entropie", "variance": "Varianz (Fehlerquadrate)"}
DEFAULT_CRITERION = {"class": "gini", "reg": "variance"}

N_MIN, N_MAX, DEFAULT_N = 400, 3000, 1200
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3               # Rauschmerkmale (zufällig, ohne Bezug zum Ziel)
LABEL_NOISE_MIN, LABEL_NOISE_MAX, DEFAULT_LABEL_NOISE = 0, 20, 0     # Prozent falsche Etiketten (nur Klassifikation)
TEST_SHARE = 0.3
DEFAULT_SEED = 7

SWEEP_SEEDS = tuple(range(100000, 100005))

# Random-Forest-eigene Regler
N_TREES_MIN, N_TREES_MAX, DEFAULT_N_TREES = 1, 150, 30
LEAF_MIN, LEAF_MAX, DEFAULT_LEAF = 1, 50, 1                 # wie Bagging: Bäume wachsen voll, kein Beschneiden
BOOTSTRAPS = 30

DEFAULT_MAP = (0, 3)          # Kartenausschnitt: Distanz x Verkehr

COLORS = {"train": "#1f77b4", "test": "#d62728", "oob": "#ff7f0e", "gini": "#1f77b4", "perm": "#2ca02c"}

MTRY_MIN = 1                    # Grenzen des mtry-Reglers werden zur Laufzeit mit der aktuellen Merkmalszahl geclippt (siehe app.py)

PRESETS = {
    "🎲 mtry = 1": dict(task="class", criterion="gini", leaf=1, n_trees=DEFAULT_N_TREES, mtry=1, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "📐 Faustregel (√d)": dict(task="class", criterion="gini", leaf=1, n_trees=DEFAULT_N_TREES, mtry=3, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🎯 Bestes mtry": dict(task="class", criterion="gini", leaf=1, n_trees=DEFAULT_N_TREES, mtry=4, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🌲🌳 mtry = alle": dict(task="class", criterion="gini", leaf=1, n_trees=DEFAULT_N_TREES, mtry=11, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "📈 Regression": dict(task="reg", criterion="variance", leaf=1, n_trees=DEFAULT_N_TREES, mtry=6, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
}
PRESET_HELP = {
    "🎲 mtry = 1": "Jeder Schnitt sieht nur 1 von 11 Merkmalen - reiner Zufall, welches. Die Bäume sind kaum noch korreliert (0.60 statt 0.77 bei mtry = alle), aber jeder Baum ist für sich schwächer: Testfehler 19.2 % - schlechter als mtry = alle (15.0 %). Dekorrelation allein genügt nicht, die Schnitte müssen noch etwas taugen.",
    "📐 Faustregel (√d)": "Die klassische Faustregel (Wurzel der Merkmalszahl, hier Wurzel(11) ≈ 3): Testfehler 15.3 % - besser als mtry = 1, aber noch nicht am gemessenen Optimum dieses Datensatzes.",
    "🎯 Bestes mtry": "mtry = 4 liegt nahe am gemessenen Optimum (im Mittel über mehrere Datensätze zwischen 5 und 8, siehe Experiment): Testfehler 14.7 % gegen 15.0 % bei mtry = alle (= Bagging) - eine kleine, aber echte Verbesserung ohne jeden Mehraufwand.",
    "🌲🌳 mtry = alle": "mtry = 11 = alle Merkmale: Random Forest verhält sich exakt wie Bagging (Korrelation 0.77, Testfehler 15.0 % - dieselben Zahlen wie im Standardfall von bagging-demo). Kein Widerspruch, sondern der Beweis: mtry ist der einzige Unterschied zu Bagging.",
    "📈 Regression": "Regression, mtry = 6: Testfehler 10.1 min gegen 10.3 min bei mtry = alle (= Bagging) und 17.0 min beim Einzelbaum. Der Gewinn gegenüber Bagging ist bei Regression insgesamt kleiner als bei Klassifikation (siehe Experiment) - hier auf diesem Datensatz sichtbar, im Mittel eher bescheiden.",
}
