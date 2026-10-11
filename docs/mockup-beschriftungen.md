# Mockup-Beschriftungen

Referenz für alle UI-Beschriftungen in den statischen Produkt-Mockups der Website; dient Audits gegen die App und als Grundlage für einen Wächter-Test, der anschlägt, wenn ein Mockup etwas zeigt, das die App nicht so nennt.

Zahlen in den Mockups sind Beispielwerte und werden nicht geprüft — nur die Wörter müssen in der App genauso vorkommen.

**Nicht enthalten:** Fließtext, Features-Listen, Pain-Antworten, Pricing-Tabellen. Fließtext darf umschreiben, Mockups müssen 1:1 der App entsprechen.

**Legende:** `datei:zeile  "wörtliches Zitat" — [Kontext in Klammern]`

---

## 1. Fair-Value-Pro Live-Vorschau-Terminal

Datei: `fair-value-pro.html` — Section `#product`, Element `<div class="terminal-col fade"><div class="terminal">`

### Terminal-Topbar
```
fair-value-pro.html:674  "Fair Value Pro" — [Terminal-Titel]
```

### Sucheingabe / Prompt-Zeile
```
fair-value-pro.html:678-679  "$" "AAPL" — [Prompt-Zeichen + Ticker-Eingabe]
```

### Abschnittsüberschriften
```
fair-value-pro.html:680  "Unternehmensqualität"
fair-value-pro.html:690  "Fairer Wert"
```

### Qualitäts-Kennzahlen (Zeilen-Labels)
```
fair-value-pro.html:681  "Eigenkapitalrendite (Ø)"
fair-value-pro.html:682  "Gewinnwachstum 5J"
fair-value-pro.html:683  "Verschuldung"      — Wert-Beispiel "moderat"
fair-value-pro.html:684  "Score"
```

### Fairer-Wert-Zeilen (Labels)
```
fair-value-pro.html:691  "Aktueller Kurs"
fair-value-pro.html:692  "DCF Fair Value"
fair-value-pro.html:693  "KGV Fairer Wert"
fair-value-pro.html:694  "Margin of Safety"
```

### Verdict-Boxen (Labels)
```
fair-value-pro.html:696  "Qualität"
fair-value-pro.html:697  "Bewertung"       — Wert-Beispiel "Überbewertet"
```

**Prüfpunkte für App:**
- „Verschuldung" vs. „Verschuldungsgrad" (Website nennt beides, siehe Output-Section Z. 581)
- „Eigenkapitalrendite (Ø)" vs. „Eigenkapitalrendite (Ø 10 J.)" (Website nennt beides, Z. 579)
- „DCF Fair Value" vs. „DCF Fairer Wert" (gemischt in derselben Datei)
- „Verdict"-Werte: Website nutzt „Überbewertet", „Unterbewertet", „Fair bewertet" — App auch?

---

## 2. Pro-Mode Dashboard-Mockup

Datei: `pro-mode.html` — Section `#hero`, Element `<div class="mockup">` (Z. 382–584)

### Topbar
```
pro-mode.html:397   "Dokumentation"                     — [einziger sichtbarer Tab]
pro-mode.html:400   "Bewertung mit angepassten Annahmen" — [Status-Text]
pro-mode.html:400   "↺ Zurücksetzen"                    — [Button]
pro-mode.html:400   "Light Mode"                        — [Button]
pro-mode.html:400   "Pro Mode"                          — [Status-Pill/Button]
```

### Sidebar (Inputs)
```
pro-mode.html:409   "Wert suchen"            — [Suchfeld-Platzhalter oder Label]
pro-mode.html:417   "Währung"                — [Dropdown-Label]
pro-mode.html:424   "Branchen-Presets"       — [Dropdown-Label]
                    "Technologie"            — [Beispiel-Preset-Wert]
pro-mode.html:427   "✓ Bewertung starten"    — [Primär-Button]
pro-mode.html:429   "⚙ Erweiterte Einstellungen" — [Button]
```

### KPI-Karten (Labels)
```
pro-mode.html:449   "DCF Fairer Wert"
pro-mode.html:454   "Komposit Fairer Wert"
pro-mode.html:456   "50% DCF · 25% KGV · 25% EV/EBITDA"  — [Sub-Zeile]
pro-mode.html:461   "Qualitätsscore"         — Wert-Beispiel "Note C"
pro-mode.html:462   "4/12 Kriterien · 33 %"  — [Sub-Zeile]
```

### Chart-Legende
```
pro-mode.html:485   "DCF-Linie"
pro-mode.html:486   "Faire Zone"             — [Legenden-Item, Band im Chart]
```

### Komposit-Panel-Zeilen (rechte Mockup-Spalte)
```
pro-mode.html:520   "Komposit Fair Value"
pro-mode.html:523   "DCF Fairer Wert"
pro-mode.html:524   "KGV Fairer Wert"
pro-mode.html:525   "Hist. Median-KGV"
pro-mode.html:526   "DDM (Gordon Growth)"
pro-mode.html:527   "PEG-Ratio"
```

### Szenarien-Labels
```
pro-mode.html:539   "Pessimistisch (Bär)"
pro-mode.html:543   "Realistisch (Basis)"
pro-mode.html:548   "Optimistisch (Bulle)"
```

### Verdict-Pill
```
pro-mode.html:529   "Stark überbewertet"     — [Verdict-Wert]
```

### Scorecard-Kriterien (10 Zeilen, Z. 573–582)
```
pro-mode.html:573   "Umsatzwachstum 10J"
pro-mode.html:574   "Umsatzwachstum 3J"
pro-mode.html:575   "EBIT-Wachstum 10J"
pro-mode.html:576   "EBIT-Wachstum 3J"
pro-mode.html:577   "Nettoverschuldung"
pro-mode.html:578   "Gewinnkontinuität"
pro-mode.html:579   "EBIT-Drawdown"
pro-mode.html:580   "Eigenkapitalrendite"
pro-mode.html:581   "ROIC/ROCE"
pro-mode.html:582   "Renditeerwartung"
```

**Prüfpunkte für App:**
- Scorecard: Website nennt 12 Kriterien (Fließtext), zeigt aber 10 im Mockup — welche zwei fehlen in der Mockup-Liste?
- Scorecard-Badge: „4/12 Kriterien · 33 %" (KPI-Sub Z. 462) vs. „C 4/10" an anderer Stelle derselben Seite (Z. 571) — welche Form zeigt die App?
- „Fair Value" vs. „Fairer Wert" — im selben Mockup gemischt (Z. 520 vs. Z. 523)
- „Komposit Fair Value"/„Komposit Fairer Wert"/„Komposit-Fairer Wert" — welche Schreibweise ist App-Standard?
- Szenarien mit Klammerzusatz („Bär"/„Basis"/„Bulle") oder ohne?

---

## 3. Portfolio-Tracker-Mockup

Datei: `portfolio-tracker.html` — Section `#mockup-section`, Element `<div class="mockup">` (Z. 315–412)

### Topbar
```
portfolio-tracker.html:317  "Portfolio Tracker"                  — [Logo-Zeile]
portfolio-tracker.html:319  "Übersicht" (active)
portfolio-tracker.html:320  "Depots"
portfolio-tracker.html:321  "Vergleich"
portfolio-tracker.html:322  "Export"
portfolio-tracker.html:324  "1J · 2J · 3J · 5J · Alle"           — [Zeithorizont-Umschalter]
```

### KPI-Karten (Labels)
```
portfolio-tracker.html:331  "Gesamtwert"              — Sub-Beispiel "3 Depots"
portfolio-tracker.html:336  "Wachstum (inkl. CF)"     — Sub-Beispiel "seit Jan 2022"
portfolio-tracker.html:341  "Rendite p.a."            — Sub-Beispiel "über 3 Jahre"
portfolio-tracker.html:346  "Marktentwicklung"        — Sub-Beispiel "exkl. Cashflows"
```

### Chart
```
portfolio-tracker.html:355  "↗ Depotverlauf"          — [Chart-Titel]
portfolio-tracker.html:356  "1J · 2J · 3J · 5J · Alle" — [Zeithorizont-Buttons]
```

### Depot-Tabelle (Spalten-Headers)
```
portfolio-tracker.html:382  "Position"
portfolio-tracker.html:382  "Wert"
portfolio-tracker.html:382  "G/V"
portfolio-tracker.html:382  "G/V %"
portfolio-tracker.html:382  "FVP-Verdict"
```

### Verdict-Werte (in Zellen)
```
portfolio-tracker.html:388  "Überbewertet"
portfolio-tracker.html:395  "Unterbewertet"
portfolio-tracker.html:402  "Fair bewertet"
```

**Prüfpunkte für App:**
- Tab-Beschriftungen „Übersicht/Depots/Vergleich/Export" — exakt so in der App?
- Zeithorizont-Umschalter „1J · 2J · 3J · 5J · Alle" — exakt diese fünf?
- KPI-Namen (vier) — exakt so?
- Spaltenname „FVP-Verdict" — oder nur „Verdict"? Oder „Fair Value"?
- Verdict-Werte „Überbewertet/Unterbewertet/Fair bewertet" — exakt diese drei?

---

## 4. Investment-Planner-Mockup (drei Panels + 10-Modul-Sidebar)

Datei: `investment-planner.html` — Section `#hero`, Element `<div class="mockup" id="td-mockup">` (Z. 340–560)

### Topbar
```
investment-planner.html:343  "Investment Planner"                                       — [Logo]
investment-planner.html:344  "Die Module — Werkzeuge, die Kaufentscheidungen vorbereiten" — [Status-Text]
investment-planner.html:345  "MAX"                                                      — [Plan-Badge]
```

### Sidebar (10 Modul-Namen)
```
investment-planner.html:361  "Meine Richtwerte"
investment-planner.html:365  "Stresstest"
investment-planner.html:369  "Watchlist"
investment-planner.html:373  "Positionsgröße"
investment-planner.html:377  "Einstiegsplan"
investment-planner.html:381  "Verkaufsprüfung"
investment-planner.html:385  "Recovery Return"
investment-planner.html:389  "Zinsrechner"
investment-planner.html:393  "Rebalancing"
investment-planner.html:397  "Trade Journal"       — [KI-Badge daneben]
```

### Panel 1 — Recovery Return
```
investment-planner.html:410  "Recovery Return"                                             — [Panel-Titel]
investment-planner.html:411  "Wie viel Wachstum braucht es, um einen Verlust auszugleichen?" — [Panel-Sub]
investment-planner.html:415  "Zeithorizont:"                                               — [Slider-Label]
```

**Recovery-Spalten-Header:**
```
investment-planner.html:425  "#"
investment-planner.html:426  "Ticker"
investment-planner.html:427  "Akt. Kurs"
investment-planner.html:428  "Fairer Wert"
investment-planner.html:429  "Drawdown"
investment-planner.html:430  "Recovery"
investment-planner.html:431  "p.a."
```

**Zellen-Marker:**
```
investment-planner.html:437, 445, 453  "FVP"   — [Badge an jedem Ticker, markiert Quelle]
```

### Panel 2 — Watchlist
```
investment-planner.html:470  "Watchlist"                                                              — [Panel-Titel]
investment-planner.html:471  "Aktien beobachten — der Titel landet in der Kaufzone, sobald er die Sicherheitsmarge erreicht." — [Panel-Sub]
investment-planner.html:475  "Watchlist"                   — [Section-Label]
investment-planner.html:476  "1 IN KAUFZONE"               — [Status-Badge]
investment-planner.html:479  "Aktualisieren"               — [Button]
investment-planner.html:480  "+ Eintrag"                   — [Button]
```

**Position-Karten-Felder (3× wiederholt):**
```
investment-planner.html:487, 495, 503  "Aktueller Kurs"
investment-planner.html:488, 496, 504  "LIVE"             — [Live-Indikator-Badge]
investment-planner.html:488, 496, 504  "Fairer Wert"
investment-planner.html:489, 497, 505  "Status"
```

**Status-Werte:**
```
investment-planner.html:489  "58,2 % über FW"
investment-planner.html:497  "21,3 % über FW"
investment-planner.html:505  "29,0 % unter FW"
investment-planner.html:490, 498  "Beobachten"
investment-planner.html:506  "Kaufzone erreicht"           — [Status-Text]
```

### Panel 3 — Positionsgröße
```
investment-planner.html:516  "Positionsgröße"                     — [Panel-Titel]
investment-planner.html:517  "Wie viele Stück, bei welchem Risiko?" — [Panel-Sub]
```

**Eingaben-Spalte (Labels):**
```
investment-planner.html:521  "Eingaben"
investment-planner.html:522  "Depotwert"
investment-planner.html:523  "Risiko je Position"
investment-planner.html:524  "Aktie"
investment-planner.html:525  "Einstiegskurs"
investment-planner.html:526  "Falsifikationsniveau"        — [App-Begriff aus trading-desk.js:2122]
investment-planner.html:527  "Kursziel"
```

**Ergebnis-Spalte (Labels):**
```
investment-planner.html:531  "Ergebnis"
investment-planner.html:533  "Stückzahl"
investment-planner.html:537  "Positionsgröße"
investment-planner.html:541  "Tatsächliches Risiko"
investment-planner.html:545  "Chance : Risiko"
```

**Rechenweg-Terme:**
```
investment-planner.html:552  "Risiko-Budget"
investment-planner.html:553  "Risiko / Stück"
investment-planner.html:554  "Stückzahl"
investment-planner.html:555  "CRV"
```

**Prüfpunkte für App:**
- Alle 10 Sidebar-Modul-Namen — wortgleich?
- Recovery-Spalten „# / Ticker / Akt. Kurs / Fairer Wert / Drawdown / Recovery / p.a." — exakt?
- Watchlist-Badge „1 IN KAUFZONE" + Status-Text „Kaufzone erreicht" — exakt in App?
- Watchlist-Status „X % über FW" / „X % unter FW" — Abkürzung „FW" oder voller Begriff „fairer Wert"?
- Positionsgröße-Felder, insbesondere: „Risiko je Position" (vs. „Risiko-Anteil" o.ä.), „Falsifikationsniveau" (bestätigt), „Chance : Risiko" vs. „CRV" (zwei verschiedene Formen im selben Panel), „Tatsächliches Risiko"
- „Einstiegskurs" vs. „Einstieg" (Rechenweg nutzt kurze Form „Einstieg")

---

## 5. Index-Startseiten-Werkzeug-Panels

Datei: `index.html` — drei kompakte `.gb-tool-ui`-Panels, eins pro Produkt (Z. 462, 474, 486)

### Panel 1 — Fair Value Pro
```
index.html:463  "Beispiel AG"       — [Panel-Titel, fiktiver Ticker]
index.html:463  "7 MODELLE"         — [Sub-Zeile]
index.html:464  "Fair Value"        — [Label]
index.html:465  "Kurs"              — [Label]
index.html:466  "Abstand"           — [Label]
```

### Panel 2 — Portfolio Tracker
```
index.html:475  "Hauptdepot"        — [Panel-Titel]
index.html:475  "14 POSITIONEN"     — [Sub-Zeile]
index.html:476  "Depotwert"         — [Label]
index.html:477  "TWR 2025"          — [Label]
index.html:478  "Heute"             — [Label]
```

### Panel 3 — Investment Planner
```
index.html:487  "Watchlist"         — [Panel-Titel]
index.html:487  "1 IN KAUFZONE"     — [Sub-Zeile]
index.html:488  "Beispiel AG"       — [Row-Label, Ticker]
index.html:489  "Nordwerk SE"       — [Row-Label, Ticker]
index.html:490  "Mikro Werke"       — [Row-Label, Ticker]
index.html:490  "Kaufzone"          — [Status-Wert]
```

**Prüfpunkte für App:**
- „Fair Value" / „Kurs" / „Abstand" — Standard-Labels im FVP-Mockup-Compact?
- „Depotwert" / „TWR 2025" / „Heute" — Standard-Labels im Tracker-Compact?
- „1 IN KAUFZONE" (Badge) + „Kaufzone" (Status) — konsistent mit Planner-Mockup
- Ticker „Beispiel AG" / „Nordwerk SE" / „Mikro Werke" sind frei erfunden, keine echten Werte

---

## Pflege

Diese Datei wird gepflegt, nicht einmalig geschrieben. Bei jeder Mockup-Änderung (Beschriftung geändert, Element hinzugefügt oder entfernt): hier nachziehen. Bei jedem App-Audit: die Einträge einzeln gegen den App-Code bestätigen oder den korrekten App-Begriff eintragen. Grundlage für den künftigen Wächter-Test.
