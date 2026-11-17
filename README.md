# inferlab

`inferlab` ist eine Python-Bibliothek für schließende Statistik, die vollständig
auf Basis von NumPy und den Verteilungsfunktionen aus SciPy implementiert ist.
Sie umfasst Schätzverfahren, Konfidenzintervalle, Hypothesentests,
Varianzanalyse und lineare Regression.

Das Projekt entstand begleitend zur Vorlesung Statistik im Wintersemester
2025/26 am Karlsruher Institut für Technologie (KIT). Ziel war es, die
Verfahren der Vorlesung nicht nur anzuwenden, sondern selbst umzusetzen.
Entsprechend greift `inferlab` nicht auf fertige Testroutinen zurück, sondern
berechnet jede Größe selbst. SciPy und statsmodels dienen ausschließlich als
Referenz in der Testsuite.

<img width="1388" height="428" alt="image" src="https://github.com/user-attachments/assets/32918a81-51fb-4022-b557-5f1d38815790" />


*Ergebnis aus dem Simulationslabor: Bei nicht normalverteilten Daten verwirft
der χ²-Varianztest eine zutreffende Nullhypothese in knapp 30 % der Fälle
statt in den vorgesehenen 5 %. Der t-Test bleibt unter denselben Bedingungen
nahe am Nominalniveau.*

## Inhalt

Der Aufbau orientiert sich an den Kapiteln der Vorlesung.

| Modul | Inhalt |
|---|---|
| `datasets` | Beispieldatensätze aus dem Vorlesungsskript, u. a. Rutherfords Zerfallsmessungen und die PTCA-Studie |
| `estimation` | Maximum-Likelihood-, Momenten- und Bayes-Schätzer |
| `intervals` | Konfidenzintervalle auf Basis von Normal-, t- und χ²-Verteilung sowie per Bootstrap |
| `hypothesis_tests` | Tests für eine und zwei Stichproben zu Lage und Streuung, Likelihood-Quotienten-Test |
| `anova` | Einfaktorielle Varianzanalyse und paarweise Vergleiche nach Bonferroni und Tukey |
| `regression` | Lineare Regression mit Koeffiziententests, Bestimmtheitsmaß und Vorhersageintervallen |
| `categorical` | Analyse von Kontingenztafeln mit χ²-Test und Fishers exaktem Test |
| `nonparametric` | Verteilungsfreie Verfahren und Anpassungstests |
| `simulate` | Monte-Carlo-Werkzeuge zur Überprüfung statistischer Eigenschaften |

### Designentscheidungen

- **Generischer ML-Schätzer.** Die Log-Dichten der unterstützten Verteilungen
  sind in einer Tabelle hinterlegt; die Maximierung übernimmt ein numerischer
  Optimierer. Weitere Verteilungen lassen sich durch einen zusätzlichen
  Tabelleneintrag ergänzen.
- **Zentrale p-Wert-Berechnung.** Alle Tests bestimmen lediglich ihre
  Teststatistik und die Verteilung unter H₀. Die Berechnung des p-Werts für
  ein- und zweiseitige Alternativen erfolgt an einer gemeinsamen Stelle.
- **Reproduzierbarkeit.** Funktionen, die Zufallszahlen benötigen, erhalten
  den Zufallsgenerator als Parameter. Alle Ergebnisse sind damit
  reproduzierbar.

## Validierung

Jedes Verfahren wird auf drei Ebenen geprüft:

1. **Abgleich mit Referenzimplementierungen.** Teststatistiken, p-Werte und
   Intervallgrenzen werden mit den Ergebnissen von SciPy und statsmodels
   verglichen, in der Regel mit einer relativen Toleranz von 10⁻⁹.
2. **Abgleich mit dem Vorlesungsskript.** Die Ergebnisse müssen die dort
   angegebenen Werte reproduzieren, etwa T = 2.63 und p = 0.0108 beim
   Vergleich der Reaktionszeiten oder X² = 8.1767 für die PTCA-Tafel.
3. **Strukturelle Eigenschaften.** Zusätzlich werden Zusammenhänge geprüft,
   die unabhängig von konkreten Daten gelten müssen: die Dualität von Test und
   Konfidenzintervall, die Beziehung F = t² zwischen Varianzanalyse und
   t-Test im Zwei-Gruppen-Fall oder die Zerlegung der Quadratsummen.

## Simulationslabor

Das Notebook `simulation_lab.ipynb` ergänzt die Bibliothek um eine
empirische Perspektive und ist in zwei Teile gegliedert.

**Teil I** behandelt die Module der Bibliothek. Jeder Abschnitt erläutert das
jeweilige Verfahren, validiert es gegen Referenzwerte und veranschaulicht es
grafisch.

**Teil II** untersucht Eigenschaften, die sich analytisch nur eingeschränkt
beurteilen lassen. Dazu werden Daten mit bekanntem wahren Parameter
wiederholt simuliert und die Ergebnisse ausgewertet. Untersucht werden unter
anderem:

- die tatsächliche Überdeckungswahrscheinlichkeit von Konfidenzintervallen,
- die Güte von Tests in Abhängigkeit von Effektstärke und Stichprobenumfang,
- die Robustheit von Tests gegenüber verletzten Verteilungsannahmen,
- der Vergleich von Bootstrap- und klassischen Intervallen bei kleinen
  Stichproben,
- die Inflation der Fehlerrate bei multiplen Paarvergleichen.

Ein abschließendes Prüfprotokoll fasst alle Kontrollen des Notebooks
zusammen.

## Installation

Das Projekt verwendet [uv](https://docs.astral.sh/uv/) zur Verwaltung von
Abhängigkeiten.

```bash
git clone https://github.com/ShawnSoek/inferlab
cd inferlab
uv sync --extra dev
```

Testsuite ausführen:

```bash
uv run pytest
```

Notebook starten:

```bash
uv run jupyter lab simulation_lab.ipynb
```

Die vollständige Ausführung des Notebooks dauert etwa zwei Minuten.

## Anwendungsbeispiel

Das folgende Beispiel aus der Vorlesung vergleicht die Reaktionszeiten von
Autofahrern, die während der Fahrt telefonieren, mit denen von Fahrern, die
Radio hören.

```python
from inferlab.datasets import load_reactiontime
from inferlab.hypothesis_tests import two_sample_t_test

phone, radio = load_reactiontime()
t, p = two_sample_t_test(phone.mu, phone.sigma, phone.group_size,
                         radio.mu, radio.sigma, radio.group_size)

print(f"T = {t:.2f}, p = {p:.4f}")   # T = 2.63, p = 0.0108
```

Bei einem Signifikanzniveau von 5 % wird die Hypothese gleicher mittlerer
Reaktionszeiten verworfen.

## Projektstruktur

```
inferlab/
├── inferlab/              # Quellcode der Bibliothek
├── tests/                 # Testsuite, eine Datei pro Modul
├── demos/                 # Beispielskripte
├── docs/                  # Abbildungen für die Dokumentation
├── simulation_lab.ipynb   # Simulationslabor
└── pyproject.toml
```

## Einschränkungen

Der Funktionsumfang deckt den Großteil, aber nicht die Gesamtheit der
Vorlesung ab. Nicht umgesetzt sind derzeit die zweifaktorielle
Varianzanalyse, der Kruskal-Wallis-Test, der Kolmogorov-Smirnov-Test sowie
grafische Diagnoseverfahren wie Q-Q-Plots.

`inferlab` ist als Lernprojekt konzipiert. Für produktive Datenanalysen
empfehlen sich etablierte Bibliotheken wie SciPy oder statsmodels, die einen
größeren Funktionsumfang bieten und umfassender getestet sind.
