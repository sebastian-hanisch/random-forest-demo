# Random Forest – Bagging mit zufälliger Merkmalsteilmenge – Streamlit-Demo

Drittes Stück der **Baumbasierten Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Nachfolger von [Bagging](../bagging-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Random Forest** (Breiman 2001) – an einem wachsenden Beispiel.
Vehikel: dieselben **Lieferungen** wie in cart-demo/bagging-demo. Der Baumkern ist aus cart-demo übernommen und um `mtry` erweitert (`rf_tree.py`); die Bootstrap-/Mittel-Logik ist wortgleich zu bagging-demo, nur um `mtry` und Permutationswichtigkeit ergänzt (`rf_algorithm.py`).
Alle Daten sind erzeugt, alle Zahlen gemessen und in `tests/test_claims.py` festgehalten – keine echten Daten, scikit-learn nur in den Tests als Gegenprobe.

**Bezug zu OR:** die Permutationswichtigkeit zeigt, welche Größen eine Tourenplanung wirklich beeinflussen (z. B. Ladegewicht, Verkehr) statt nur zufällig hoch bewertet zu werden – ein Filter, bevor man ein Merkmal in ein Optimierungsmodell aufnimmt.

**Einordnung in die Reihe:** Bagging mittelt Bootstrap-Bäume, bleibt aber korreliert, wenn ein Merkmal jeden Baum dominiert (gemessen in bagging-demo). Random Forest setzt genau dort an: an **jedem** Split jedes Baums darf nur eine zufällige Teilmenge von **mtry** Merkmalen überhaupt zur Wahl stehen. `mtry = alle Merkmale` macht daraus wieder exakt Bagging – die Kreuzprobe dieses Stücks.

```
CART → Bagging → Random Forest (dieses Stück) → Extra Trees
```

| Frage | Ergebnis (1200 Lieferungen, 3 Rauschmerkmale [d = 11], 70 % Training / 30 % Test, Seed 7; Klassifikation "zu spät", 30 Bäume) |
|---|---|
| **mtry = alle Merkmale == Bagging** | ✅ Exakt dieselben Bäume, dieselbe Vorhersage (Kreuzprobe: dieselben Bootstrap-Indizes + kein `mtry` ergeben bitweise identische Bäume). Testfehler 15,0 %, Korrelation 0,77 – dieselben Zahlen wie der Standardfall in bagging-demo. |
| mtry = 1 | ❌ Baumkorrelation fällt auf 0,60, aber jeder Split ist fast zufällig: Testfehler 19,2 % – **schlechter** als mtry = alle. Dekorrelation allein genügt nicht. |
| Faustregel (√d ≈ 3) | ➖ Testfehler 15,3 % – besser als mtry = 1, aber nicht am gemessenen Optimum. |
| **Bestes mtry** (dieser Datensatz) | ✅ mtry = 4: Testfehler 14,7 % gegen 15,0 % bei mtry = alle – eine kleine, aber echte Verbesserung ohne Mehraufwand. |
| **mtry-Sweep** (Mittel über drei Datensätze, 20 Bäume) | ✅ Klassifikation: Minimum bei **mtry 5** (13,9 %), klar besser als mtry 1 (16,9 %) und etwas besser als mtry 11 (15,5 %). Regression: Minimum bei **mtry 8** (9,9 min) gegen 17,0 min (mtry 1) und 10,2 min (mtry 11). Die Kurve ist in der Mitte flach – die genaue Wahl ist weniger wichtig als überhaupt zu dekorrelieren. |
| **Gini- gegen Permutationswichtigkeit** (Mittel über sechs Datensätze) | ❌ Bei 3 Rauschmerkmalen bekommt die Gini-Wichtigkeit **12,4 %** ihrer Summe auf reines Rauschen verteilt, die Permutationswichtigkeit nur **0,9 %**. Bei 8 Rauschmerkmalen wächst die Verzerrung: **27,2 %** gegen **3,1 %**. Gini bevorzugt Merkmale mit vielen möglichen Schwellen (Strobl u. a. 2007) – auch bei Regression sichtbar (9,1 % gegen 0,8 % bei 3 Rauschmerkmalen). |
| Regression, bestes mtry | ✅ mtry = 6: Testfehler 10,1 min gegen 10,3 min bei mtry = alle und 17,0 min beim Einzelbaum – der Gain gegenüber Bagging ist bei Regression insgesamt kleiner als bei Klassifikation. |

## Was die Demo zeigt

- **Random Forest in Aktion:** der Wald wächst Baum für Baum mit Schritt-Regler und Abspielen: links der zuletzt hinzugekommene Einzelbaum, dessen **Wurzel-Kandidaten** (die zufällig gezogenen mtry Merkmale) genannt werden, rechts das Wald-Mittel über zwei wählbare Merkmale.
- **Was der Wald gelernt hat:** Testfehler, Out-of-Bag-Fehler gegen einen Einzelbaum **und** gegen mtry = alle (Bagging), ein Urteil (besser / kein Gain trotz Dekorrelation / kaum Unterschied / schlechter), Baumkorrelation, Wurzel-Anteile, **Gini- gegen Permutationswichtigkeit** nebeneinander.
- **Regler:** Aufgabe, Kriterium, Mindestblattgröße, Zahl der Bäume, **mtry** (Grenze passt sich der aktuellen Merkmalszahl an), Rauschmerkmale, Lieferungen, falsche Etiketten, Seed.
- **Experimente auf Knopfdruck:** Testfehler und Baumkorrelation gegen mtry (voller Sweep), Anteil der Gini- bzw. Permutationswichtigkeit auf Rauschmerkmalen.

## Modell und Verfahren

- **Wie Bagging:** B Bootstrap-Stichproben, volle Bäume, gemittelt; Out-of-Bag wie gehabt.
- **mtry:** an jedem Knoten wird vor der Split-Suche eine zufällige Teilmenge von mtry Merkmalen gezogen (eigener, vom Bootstrap unabhängiger Zufalls-Strang je Baum); nur unter diesen wird der beste Split gesucht. `mtry >= d` (Merkmalszahl) verhält sich exakt wie ohne Einschränkung.
- **Permutationswichtigkeit** (Breiman 2001): je Baum der Fehler auf seinen Out-of-Bag-Zeilen, dann eine Spalte unter diesen Zeilen gemischt und der Fehler erneut gemessen – der Anstieg ist die Wichtigkeit, gemittelt über alle Bäume mit genug OOB-Zeilen. Für den direkten Vergleich mit der (auf Summe 1 normierten) Gini-Wichtigkeit werden negative Werte auf 0 gekappt und ebenfalls normiert.

## Was nicht funktioniert hat / Grenzen

- **Der Bagging-Vergleich läuft in jeder Analyse mit:** `analyse()` fitted immer zusätzlich einen Wald mit `mtry = alle Merkmale` auf denselben Bootstrap-Stichproben – so ist der Bagging-Vergleich nie ein anderer Lauf mit anderem Zufall, sondern derselbe Bootstrap, nur ohne die mtry-Einschränkung. Ohne das wäre der Vorher-Nachher-Vergleich selbst verrauscht gewesen.
- **mtry als "Stärke"-Regler funktioniert, ohne die Daten anzufassen** – anders als beim Dominanz-Experiment in bagging-demo (das die Daten künstlich verrauschen musste, weil Stauchen/Skalieren wirkungslos ist), ändert mtry direkt die Split-Suche selbst.
- **Der mtry-Sweep braucht abgespeckte Einstellungen:** ein voller Sweep (11 mtry-Werte × 30 Bäume × sechs Datensätze) hätte bei Regression über 90 Sekunden gedauert (volle Regressionsbäume sind teuer, siehe cart-demo). Der Sweep läuft deshalb mit 20 Bäumen auf drei Datensätzen – für den Trend reicht das, die genauen Zahlen weichen leicht vom Hauptdatensatz (30 Bäume) ab.
- **Gini-Verzerrung ist bei Regression etwas schwächer als bei Klassifikation** (9,1 % gegen 12,4 % bei 3 Rauschmerkmalen) – der Effekt ist real in beiden Aufgaben, aber nicht gleich stark; wurde gemessen statt angenommen.

## Verifikation

`tests/test_algorithm.py` (20 Tests): **mtry = alle Merkmale reproduziert den unveränderten cart-Baum exakt** (drei Varianten: `None`, genau `d`, über `d` hinaus); ein frisch gewachsener Wald mit `mtry = None` stimmt mit dem Baumkern ohne Einschränkung überein; `mtry` schränkt die Split-Suche nachweislich auf die gezogenen Kandidaten ein; verschiedene Merkmals-Seeds ziehen verschiedene Kandidaten; **`root_candidates` rekonstruiert den tatsächlichen Zug** und enthält immer das gewählte Wurzelmerkmal; Kreuzprobe gegen `RandomForestClassifier`/`RandomForestRegressor` über ein Fehlerband (andere Zufallsquelle für Bootstrap und mtry, keine exakte Übereinstimmung erwartet); Out-of-Bag gegen eine naive Schleife; Permutationswichtigkeit nahe 0 für reines Rauschen und die Gini-Verzerrung bei einem stetigen gegen ein grobes Rauschmerkmal; Grenzfälle.
`tests/test_claims.py` hält **jede Zahl** aus App und README fest. `tests/test_app.py` prüft die Oberfläche per AppTest (jedes Preset, Aufgabenwechsel, mtry-Grenzen bei wechselnder Merkmalszahl, Abspielen mit mehreren Bildern und schrittspezifischen Diagramm-Schlüsseln, Permalink, Experimente).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `rf_tree.py` | Baumkern (aus cart-demo, um `mtry` erweitert) |
| `rf_algorithm.py` | Bootstrap, Mitteln, Out-of-Bag, Baumkorrelation, Permutationswichtigkeit |
| `rf_scenario.py` | Lieferdaten (wie cart-demo/bagging-demo) |
| `rf_evaluation.py` | Analyse, mtry-Sweep, Wichtigkeits-Experiment |
| `rf_visualization.py` | Baumdiagramm, Karte, mtry- und Wichtigkeits-Kurven |
| `rf_presets.py`, `rf_constants.py` | Regler, Permalink, Schnellstart-Beispiele, Grenzen |
| `tests/` | Algorithmus-, Claims- und App-Tests |

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -q
```

---

Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
