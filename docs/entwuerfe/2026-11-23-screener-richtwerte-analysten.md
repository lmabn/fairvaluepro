# Entwurf: Website-Anpassungen nach App-Änderung Screener / Richtwerte / Analystenschätzungen

Stand: 23.11.2026 · Noch nichts an echten Seiten geändert · Website zieht nach, sobald App-Änderungen live sind.

## Bildquelle für Investment Planner und Startseite

**Export Nr. 84** von Claude Design (23.11.2026) — **ersetzt Export Nr. 83**: „Website: Bilder mit dem Modulmenü des Investment Planners auf neuem Stand (Entwurf)". Zip in Session-Uploads, LIESMICH und Dateien entpackt nach `/tmp/claude-0/…/scratchpad/export-84/`. Die Bilder sind wie bisher gezeichnete HTML-Ansichten, keine Bilddateien.

Enthalten:
- `ui_kits/website/homepage.html` — Planner-Bild im Abschnitt mit den drei Werkzeugen: Modulliste mit Screener an erster Stelle und gewählt, geöffnete Screener-Ansicht mit Filter-Zeile und Trefferliste (Desktop); **720–899 px (Tablet hochkant): Auswahlfeld „Modul · Screener" statt der langen Liste, max-width 360 px**; iPad Mini/unter 1100 px Trefferliste reduziert; Handy: Auswahlfeld + drei Filter + Karte mit vier Treffern.
- `ui_kits/website/investment-planner.html` — die zwei Bilder (Kopf-Mockup Watchlist, Mockup Termine & News) mit Seitenliste, die mit Screener beginnt; Karte „Screener" mit Lupen-Icon im Modul-Raster; **die zehn Karten stehen bereits in der Reihenfolge der Modulliste** (Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest, Trade Journal).
- Begleitende CSS/Fonts/Tokens unverändert aus dem Design System.

Für diesen Entwurf gilt: **Icon (Lupe) und Mockup-Inhalt werden aus Export Nr. 84 1:1 übernommen; kein selbst gebautes Bild und kein eigenes Icon.** Kartentext ist mit Nr. 84 bestätigt. Was Export Nr. 84 nicht enthält (partner.html, plaene.html, llms.txt, ueber-gobetter.html, FVP-Reverse-DCF-Grafik), wird separat angepasst wie unten beschrieben; die FVP-Grafik kommt in einem späteren Design-Export.

## Was in der App passiert

1. **Investment Planner — neues Modul „Screener"** vor der Watchlist.
   Filter nach Art (Aktie, ETF, Fonds), Börse, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite, Beta.
   Treffer sortiert nach Börsenwert, mit „Auf die Watchlist" und „Bewerten".
   Kein Filter nach KGV oder Fairem Wert.
2. **Investment Planner — „Meine Richtwerte" entfällt als Modul.**
   Der Richtwert für den Rückgang steht im **Stresstest**, der Cash-Korridor im **Rebalancing**.
   Festgelegt werden beide in den **Einstellungen**.
   Es bleibt bei **zehn Modulen**: Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest, Trade Journal.
   Der Planner bekommt eine **Startseite mit drei Schritten**, solange die Watchlist leer ist.
3. **Fair Value Pro — Analystenschätzungen als Vergleich.**
   Erwartetes Umsatzwachstum p. a. neben der eigenen Annahme im Reverse DCF, dazu ein Diagramm zum Umsatz.
   Sie gehen nie in die Rechnung ein; „Rechnung ohne Analystenschätzungen" bleibt richtig.

## Seiten-Übersicht

| Datei | Betroffen |
|---|---|
| `investment-planner.html` | Meta, JSON-LD (SoftwareApplication featureList + FAQ), Hero-CTA-Text, zwei Modul-Mockups (Seitenleiste), Modul-Karte „Meine Richtwerte" → „Screener", neue „Drei-Schritte"-Startseiten-Beschreibung, FAQ-Modulliste |
| `fair-value-pro.html` | Z. 551 „Fertige Screener", Z. 668 „Reverse DCF"-Karte |
| `index.html` | Modul-Mockup im Investment-Planner-Band (aktives Modul + Icons), Lead-Text mit Ticks |
| `partner.html` | Accordion „Investment Planner — mehr dazu" (Modul-Aufzählung) |
| `plaene.html` | Vergleichstabelle-Gruppe „Investment Planner" (Zeile ergänzen) |
| `llms.txt` | Produkt-Zeile Investment Planner |
| `ueber-gobetter.html` | FAQ-Antwort „Was macht der Investment Planner?" sichtbar + im FAQPage-JSON-LD |

Nicht betroffen auf der Marketing-Website:
- `agb.html` § 2 erwähnt „Investment Planner — Werkzeug zur Vorbereitung und Dokumentation eigener Anlageentscheidungen (u. a. Rebalancing-Planung, Trade-Journal, Watchlist, …)" — die Aufzählung ist beispielhaft („u. a.") und bleibt konsistent.
- `datenschutz.html`, `impressum.html`, `community.html`, `daten-quelle.html`, `portfolio-tracker.html`, `akademie/*` — keine Nennung von „Meine Richtwerte" oder GO:BETTER-Modul „Screener".
- `styleguide.html` nennt „Zinsrechner, Rebalancing, Watchlist, Trade-Journal" als Beispiel für Investment-Planner-Charakter — bleibt, nennt keine Richtwerte.

---

## 1) `investment-planner.html`

### 1a) Meta-Beschreibung · Z. 9, 15, 20
Begründung: enthält „Richtwerte" prominent, muss die neue Reihenfolge und den Screener widerspiegeln.

- **alt:** `Von der Watchlist bis zum Verkauf: Richtwerte, Stresstest, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Rebalancing und Trade-Journal.`
- **neu:** `Vom Screener bis zum Verkauf: Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Rebalancing, Stresstest und Trade-Journal.`

Gleicher Text in `<meta name="description">`, `og:description` und `twitter:description`.

### 1b) JSON-LD SoftwareApplication · Z. 22
Begründung: `description` identisch zur Meta spiegeln; `featureList` ohne „Meine Richtwerte", stattdessen „Screener".

- **alt (description):** `Von der Watchlist bis zum Verkauf: Richtwerte, Stresstest, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Rebalancing und Trade-Journal.`
- **neu (description):** `Vom Screener bis zum Verkauf: Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Rebalancing, Stresstest und Trade-Journal.`

- **alt (featureList):**
  ```
  ["Meine Richtwerte und Stresstest",
   "Watchlist mit Terminen und News",
   "Positionsgröße aus Risiko und Falsifikationsniveau",
   "Einstiegsplan",
   "Verkaufsprüfung",
   "Rebalancing über alle Depots",
   "Zinsrechner",
   "Trade Journal mit KI-Prüfung auf Denkfehler"]
  ```
- **neu (featureList):**
  ```
  ["Screener nach Art, Börse, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite und Beta",
   "Watchlist mit Terminen und News",
   "Positionsgröße aus Risiko und Falsifikationsniveau",
   "Einstiegsplan",
   "Verkaufsprüfung",
   "Rebalancing über alle Depots (inklusive Cash-Korridor)",
   "Stresstest mit Richtwert für den Rückgang",
   "Zinsrechner",
   "Trade Journal mit KI-Prüfung auf Denkfehler"]
  ```

### 1c) JSON-LD FAQPage · Z. 22
Antwort auf „Was ist der Investment Planner?" spiegelt die neue Modul-Reihenfolge (parallel zu 1h).

- **alt:** `Werkzeuge für die Entscheidung vor dem Kauf: Meine Richtwerte, Positionsgröße, Einstiegsplan, Watchlist, Termine & News, Verkaufsprüfung, Stresstest, Rebalancing, Zinsrechner und Trade Journal.`
- **neu:** `Werkzeuge vor und nach dem Kauf, in der Reihenfolge der App: Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest und Trade Journal.`

### 1d) Hero-CTA · Z. 405
Begründung: Der Button-Text zählt weiter zehn Module — bleibt richtig, keine Änderung.

- **alt:** `<span>Zehn Module ansehen</span>`
- **neu:** `<span>Zehn Module ansehen</span>` (unverändert)

### 1e) Modul-Mockup Seitenleiste (Watchlist-Mockup) · Z. 412
Begründung: Reihenfolge der App spiegeln, „Meine Richtwerte" raus, „Screener" rein.

- **alt:** `<div class="side"><span>Meine Richtwerte</span><span class="on">Watchlist</span><span>Termine & News</span><span>Positionsgröße</span><span>Einstiegsplan</span><span>Verkaufsprüfung</span><span>Zinsrechner</span><span>Rebalancing</span><span>Stresstest</span><span>Trade Journal <small>KI</small></span></div>`
- **neu:** `<div class="side"><span>Screener</span><span class="on">Watchlist</span><span>Termine & News</span><span>Positionsgröße</span><span>Einstiegsplan</span><span>Verkaufsprüfung</span><span>Zinsrechner</span><span>Rebalancing</span><span>Stresstest</span><span>Trade Journal <small>KI</small></span></div>`

### 1f) Modul-Karten · Z. 483–492
Begründung: Karte „Meine Richtwerte" entfällt; neue Karte „Screener" wird die erste Karte in der Gruppe.

- **alt (Z. 483):**
  `<div class="card"><span class="ico" aria-hidden="true">◎</span><h3>Meine Richtwerte</h3><p>Deine Grenzen, einmal festgelegt: maximaler Rückgang in Euro, Cash-Korridor. Alles andere rechnet dagegen.</p></div>`
- **neu (Z. 483, neue Karte „Screener" — Text wörtlich aus der App, Markup und Icon aus Export Nr. 84):**
  `<div class="card"><span class="ico" aria-hidden="true"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5"><!-- Lupe aus Export Nr. 84 --></svg></span><h3>Screener</h3><p>Aktien, ETFs und Fonds nach Land, Branche und Größe filtern. Treffer setzt du auf die Watchlist oder bewertest sie in Fair Value Pro.</p></div>`

  Icon-Pfad: 1:1 aus `ui_kits/website/investment-planner.html` (Export Nr. 84) übernehmen. Der kurze App-Text ersetzt meinen längeren Vorschlag aus Runde 2; die vollständige Filterliste (Art, Börse, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite, Beta) steht weiterhin in Meta/JSON-LD/llms.txt (1a, 1b, 6a).

- **Stresstest-Karte (Z. 484):** Beschreibung um den Richtwert ergänzen.
  - **alt:** `Was kostet der schlechte Tag? Größter Sektor minus 30 %, größte Position auf null, Krypto halbiert — in Euro, gegen deine Grenze gestellt.`
  - **neu:** `Was kostet der schlechte Tag? Größter Sektor minus 30 %, größte Position auf null, Krypto halbiert — in Euro, gegen den Richtwert für den Rückgang, den du in den Einstellungen festgelegt hast.`

- **Rebalancing-Karte (Z. 491):** Cash-Korridor ergänzen.
  - **alt:** `Soll gegen Ist über alle Depots, mit konkreten Beträgen in Euro: was aufstocken, was kürzen.`
  - **neu:** `Soll gegen Ist über alle Depots, mit konkreten Beträgen in Euro: was aufstocken, was kürzen. Zeigt auch, ob dein Cash-Korridor eingehalten ist.`

- **Verkaufsprüfung-Karte (Z. 489):** unverändert.
- **Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Zinsrechner, Trade Journal:** unverändert.

**Reihenfolge der zehn Karten im Grid:** Die Karten folgen der Modulliste der App. Neue Reihenfolge: Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest, Trade Journal. Die aktuelle Reihenfolge im HTML ist: Meine Richtwerte, Stresstest, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Trade Journal — wird umsortiert. Diese Reihenfolge ist in Export Nr. 84 bereits so geliefert (Änderung A gegenüber Nr. 83).

### 1g) Hinweis-Block „Verbunden mit dem Ökosystem" · Z. 494
Begründung: Der Screener greift nicht auf gespeicherte Bewertungen zu — er übergibt einen Titel per Klick an Fair Value Pro oder die Watchlist. Watchlist und Verkaufsprüfung nutzen weiter die gespeicherten Bewertungen. Stresstest nutzt (zusätzlich zum Rebalancing) die Depot-Daten.

- **alt:** `Watchlist und Verkaufsprüfung greifen auf deine gespeicherten Fair-Value-Pro-Bewertungen zu. Rebalancing nutzt die Depot-Daten aus dem Portfolio Tracker. Alles mit demselben Account — kein Datendoppel.`
- **neu:** `Watchlist und Verkaufsprüfung greifen auf deine gespeicherten Fair-Value-Pro-Bewertungen zu. Der Screener übergibt einen Titel mit einem Klick an Fair Value Pro oder die Watchlist. Rebalancing und Stresstest nutzen die Depot-Daten aus dem Portfolio Tracker. Alles mit demselben Account — kein Datendoppel.`

### 1h) Modul-Mockup Seitenleiste (Termine & News-Mockup) · Z. 552
Begründung: analog 1e.

- **alt:** `<div class="side"><span>Meine Richtwerte</span><span>Watchlist</span><span class="on">Termine & News</span>…`
- **neu:** `<div class="side"><span>Screener</span><span>Watchlist</span><span class="on">Termine & News</span>…`

### 1i) FAQ „Was ist der Investment Planner?" (sichtbar) · Z. 624
Wörtlich identisch zur FAQPage-Antwort in 1c.

- **alt:** `Werkzeuge für die Entscheidung vor dem Kauf: Meine Richtwerte, Positionsgröße, Einstiegsplan, Watchlist, Termine & News, Verkaufsprüfung, Stresstest, Rebalancing, Zinsrechner und Trade Journal.`
- **neu:** `Werkzeuge vor und nach dem Kauf, in der Reihenfolge der App: Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest und Trade Journal.`

### 1j) Neue Startseite „Drei Schritte, solange die Watchlist leer ist"
Begründung: Teil der App-Änderung. Umsetzung **nur als FAQ-Eintrag**, sichtbar und gleichlautend im FAQPage-JSON-LD. Kein eigener Band, keine zusätzliche Modul-Karte.

- **Neuer FAQ-Eintrag** (direkt nach „Was ist der Investment Planner?"), wörtlich:
  - Frage: „Wie fange ich im Investment Planner an?"
  - Antwort: „Solange deine Watchlist leer ist, zeigt der Planner eine Startseite mit drei Schritten: einen Titel finden, im Screener oder über eine Bewertung in Fair Value Pro; ihn auf die Watchlist setzen und die Schwelle festlegen; den Kauf mit Positionsgröße und Einstiegsplan planen."
  - Im sichtbaren FAQ-Block und im FAQPage-JSON-LD wortgleich.

### 1k) CSS-Kommentar · Z. 351
- **alt:** `/* 15.11.: zehn Modul-Karten ohne leere Fläche, Einstiegsplan breiter */`
- **neu:** `/* <neues Datum>: zehn Modul-Karten (Screener statt Richtwerte), Einstiegsplan breiter */`
Nur kosmetisch, keine Build-Wirkung.

### 1l) Bilder / Mockups
Keine SVG- oder PNG-Dateien als externe Assets betroffen. Die zwei sichtbaren App-Mockups bestehen aus HTML-Elementen (`.side` + `.mh` + Tabelle). **Quelle für Markup und Icon: Export Nr. 84** (`ui_kits/website/investment-planner.html`):

- Kopf-Mockup „Watchlist" (Z. 409) — Seitenleiste beginnt jetzt mit „Screener"; Inhalt der Watchlist-Ansicht unverändert.
- Mockup „Termine & News" (Z. 549) — Seitenleiste beginnt jetzt mit „Screener"; Inhalt unverändert.
- Modul-Karte „Screener" (Z. 483) — Lupen-Icon (18 px, Strich 1,5) und Kartentext aus Export Nr. 84.
- Reihenfolge der zehn Karten im Grid: schon in Export Nr. 84 in der Modulliste-Reihenfolge (siehe oben).

Markup und Icon 1:1 aus Export Nr. 84 übernehmen; die exakten SVG-Pfade liegen in `ui_kits/website/investment-planner.html` im entpackten Export.

Kein selbstgezeichneter Screener-Mockup-Block an dieser Stelle. Falls zusätzlich zur Karte eine eigene Screener-Ansicht auf dieser Seite gewünscht ist, kommt sie in einem späteren Design-Export.

---

## 2) `fair-value-pro.html`

### 2a) Zeile 551 „Andere Tools: Annahmen versteckt"
Begründung: „Fertige Screener" als Werkstoff **anderer** Anbieter kollidiert begrifflich mit dem neuen GO:BETTER-Modul „Screener" im Investment Planner. Wort umstellen, Aussage behalten.

- **alt:** `<p>Fertige Screener liefern eine Zahl, aber nicht, welche Annahmen dahinterstecken.</p>`
- **neu:** `<p>Fertige Bewertungs­angebote anderer Anbieter liefern eine Zahl, aber nicht, welche Annahmen dahinterstecken.</p>`

Die Details im `<details class="more">` dahinter bleiben unverändert („Welche Wachstumsrate, welcher Diskontierungssatz? …").

### 2b) Modul-Karte „Reverse DCF" · Z. 668
Begründung: Neue Analystenschätzungen als Vergleich im Reverse DCF; Abgrenzung „nie in der Rechnung" explizit nennen.

- **alt:** `<div class="card"><span class="ico" aria-hidden="true">↺</span><h3>Reverse DCF</h3><p>Zeigt, welches Wachstum der heutige Kurs voraussetzt, neben deiner Annahme.</p></div>`
- **neu:** `<div class="card"><span class="ico" aria-hidden="true">↺</span><h3>Reverse DCF</h3><p>Zeigt, welches Wachstum der heutige Kurs voraussetzt — neben deiner Annahme und dem erwarteten Umsatzwachstum p. a. aus Analystenschätzungen, mit einem Diagramm zum Umsatz. Analystenschätzungen gehen nie in die Rechnung ein.</p></div>`

### 2c) Meta-Beschreibung
Vorschlag **unverändert.** Die aktuelle `<meta name="description">` von Fair Value Pro nennt (wenn gültig) nicht „Reverse DCF" oder „Screener", also kein Handlungsbedarf. Zur Sicherheit vor dem Livegang kurz prüfen.

### 2d) FAQ / weitere Abschnitte
Keine weiteren Treffer zu „Richtwerte" oder „Screener" auf dieser Seite.

- **Neuer FAQ-Eintrag** (sichtbar + im FAQPage-JSON-LD, falls vorhanden; sonst nur sichtbar):
  - Frage: „Nutzt Fair Value Pro Analystenschätzungen in der Rechnung?"
  - Antwort: „Nein. Fair Value Pro rechnet mit den Finanzzahlen deines Datenanbieters und deinen Annahmen. Im Reverse DCF steht das erwartete Umsatzwachstum p. a. aus Analystenschätzungen zum Vergleich daneben, dazu ein Diagramm zum Umsatz. Die Schätzungen gehen nie in die Rechnung ein."

### 2e) Bilder / Mockups
Reverse-DCF-Modul und das Diagramm zum Umsatz (deine Annahme neben Analystenschätzung p. a., dazu die historische Umsatzlinie) werden von Claude Design nach dem Stand der App geliefert. Kein selbst gebautes Bild.

Platzierung auf der Seite: im Modul-Karten-Grid (Reverse-DCF-Karte Z. 668) und ggf. ein Screenshot/Mockup in einem bestehenden Band — konkrete Stelle wird nach Erhalt des Assets festgelegt.

---

## 3) `index.html`

### 3a) Lead-Ticks · Z. 885–886
Begründung: „Richtwerte für Cash-Quote, Positionsgröße und Rückgang" setzt das alte Modul „Meine Richtwerte" voraus. Jetzt stehen die Richtwerte in den Einstellungen, und der Screener ist die erste Eintrittsstelle.

- **alt:**
  ```
  <p>Lege deine Richtwerte in Ruhe fest. Der Planner stellt jede neue Entscheidung dagegen, mit den Zahlen aus deinem Tracker.</p>
  <ul class="ticks">
    <li>Richtwerte für Cash-Quote, Positionsgröße und Rückgang</li>
    <li>Einstiegsplan in Tranchen, Verkaufsprüfung vor jedem Verkauf</li>
    <li>Rebalancing und Stresstest auf deinem echten Bestand</li>
    <li>Zehn Module, alle mit deinen Daten verbunden</li>
  </ul>
  ```
- **neu:**
  ```
  <p>Vom Kandidaten bis zum Verkauf. Der Screener filtert Aktien, ETFs und Fonds; der Planner nimmt die Treffer durch Watchlist, Einstiegsplan und Verkaufsprüfung — mit den Zahlen aus deinem Tracker.</p>
  <ul class="ticks">
    <li>Screener: Aktien, ETFs und Fonds nach Land, Branche und Größe filtern</li>
    <li>Einstiegsplan in Tranchen, Verkaufsprüfung vor jedem Verkauf</li>
    <li>Rebalancing und Stresstest auf deinem echten Bestand</li>
    <li>Zehn Module, alle mit deinen Daten verbunden</li>
  </ul>
  ```

  Lead und Ticks spiegeln den kurzen App-Satz aus der Karte (1f). Der ausführliche Screener-Mockup der Startseite (Filter-Zeile, Trefferliste, Handy-/iPad-Varianten) kommt aus Export Nr. 83 und wird in 3b übernommen — auch er zeigt die drei Haupt-Filter.

### 3b) Modul-Mockup im IP-Band · Z. 761–786
Begründung: Aktives Modul wechselt von „Meine Richtwerte" zu „Screener"; das Modul-Selector-Chip oben, der aktive Eintrag in `.mods`, die Modul-Icons, die Mockup-Überschrift, die Beschreibungs-Zeile sowie der komplette Inhalt (zwei `.tile`-Kacheln + `.card.m-only`-Block) werden ausgetauscht.

**Stellen in `index.html`:**
- Z. 761: Modul-Selector-Chip (zur Zeit „Meine Richtwerte").
- Z. 763: erstes Element in `.mods` (`class="on"`, zur Zeit „Meine Richtwerte" mit Richtwerte-Icon).
- Z. 775: große Überschrift + Beschreibungs-Zeile im Mockup-Inhalt (zur Zeit „Meine Richtwerte" + „Deine Richtwerte, in Ruhe festgelegt, gegen den aktuellen Stand gestellt.").
- Z. 776–786: Mockup-Inhalt mit zwei Richtwert-Kacheln und Stresstest-Block (`.card.m-only`).

**Quelle: Export Nr. 84** (`ui_kits/website/homepage.html`, Ansicht „Investment Planner" im App-Fenster, `#v-ip`). Daraus 1:1 übernommen:

- Modul-Selector-Chip-Text „Screener".
- Modulliste mit zehn Modulen, Screener gewählt; Lupen-Icon 16 px, Strich 1,4 (im Stil der übrigen Modul-Icons). Ab 900 px wie bisher links neben dem Inhalt.
- Geöffnete Screener-Ansicht Desktop: Zeile mit drei gesetzten Filtern (Art · Börse · Börsenwert — Werte „Aktie" · „Xetra" · „ab 2 Mrd. €"), „Titel suchen"-Knopf, Trefferliste mit Spalten Titel · Land · Börsenwert ↓ · Kurs · Dividendenrendite · „Auf die Watchlist" · „Bewerten" (fünf Beispieltitel, „5 Treffer · sortiert nach Börsenwert").
- **Tablet hochkant 720–899 px**: statt der langen Modulliste das Auswahlfeld „Modul · Screener" (max. 360 px breit, `.mods` aus, `.modsel` an — dieselben Regeln wie unter 720 px). Neu in Export Nr. 84 (Änderung B gegenüber Nr. 83).
- iPad Mini und alles unter 1100 px: reduzierte Trefferliste (Titel · Börsenwert · zwei Handlungen) — gleiche `.sct`-Mechanik, Land/Kurs/Dividendenrendite ausgeblendet.
- Handy: Auswahlfeld „Modul · Screener", die drei Filter, Karte mit vier Treffern (Name + Börsenwert). „4 Treffer" statt „5".

**Neue CSS-Klassen aus Export Nr. 84** (für `homepage.html` bzw. `index.html`):
- `.scf` — Filter-Zeile (Felder 32 px, Bezeichnung klein und gedämpft, Wert in Textfarbe; `.scf span.p` als „Titel suchen"-Knopf).
- `.sct` — Trefferliste als Tabelle (Kopf Mono 10,5 px Versalien, Zahlen Mono rechtsbündig, Kürzel klein unter dem Namen, zwei kleine Knöpfe rechts; unter 1100 px blendet `.sct` Land/Kurs/Dividendenrendite aus).
- **720–899 px: `.mods` aus, `.modsel` an, max-width 360 px** (neu in Nr. 84).
- Lupen-Icon (16 px, Strich 1,4) als erstes Element in `.mods`.

Kein eigenes Markup und kein eigenes Icon im Entwurf — bei der Umsetzung wird 1:1 eingesetzt, was Export Nr. 84 liefert.

### 3c) Bilder / Mockups
Icon-Glyph „Screener", Filter-Zeile, Trefferliste, Handy-Karte, Tablet-hoch-Auswahlfeld und die neuen CSS-Klassen `.scf` / `.sct` kommen aus Export Nr. 84. Kein selbst gebautes Bild und kein eigenes Icon im Entwurf.

---

## 4) `partner.html`

### 4a) Accordion „Investment Planner — mehr dazu" · Z. 645
Begründung: Modul-Aufzählung nennt „Meine Richtwerte" an erster Stelle und zeigt eine veraltete Reihenfolge; neue Reihenfolge der App spiegeln.

- **alt:** `Zehn Module in einer Seitenleiste: Meine Richtwerte, Stresstest, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Trade Journal. Alle greifen auf deine gespeicherten Bewertungen und Depot-Daten zu. Das Trade Journal lässt dich eine Kaufbegründung festhalten, die nach dem Speichern nicht mehr änderbar ist; ein optionaler KI-Coach prüft sie gegen 16 bekannte Denkfehler. Teil des MAX-Plans.`
- **neu:** `Zehn Module in einer Seitenleiste in der Reihenfolge der App: Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest, Trade Journal. Aktien, ETFs und Fonds nach Land, Branche und Größe filtern. Treffer setzt du auf die Watchlist oder bewertest sie in Fair Value Pro. Watchlist und Verkaufsprüfung greifen auf deine gespeicherten Bewertungen zu, Rebalancing und Stresstest auf die Depot-Daten. Das Trade Journal lässt dich eine Kaufbegründung festhalten, die nach dem Speichern nicht mehr änderbar ist; ein optionaler KI-Coach prüft sie gegen 16 bekannte Denkfehler. Teil des MAX-Plans.`

### 4b) Kurz-Beschreibung Investment Planner · Z. 644
Begründung: „von der Positionsgröße bis zum Trade Journal" — unverändert ok; alternativ „vom Screener bis zum Trade Journal", um die neue Startstelle zu betonen.

- **alt:** `Zehn Module, mit denen aus Daten ein Plan wird — von der Positionsgröße bis zum Trade Journal.`
- **neu (Vorschlag):** `Zehn Module, mit denen aus Daten ein Plan wird — vom Screener bis zum Trade Journal.`

### 4c) Bilder / Mockups
Keine — reine Textseite.

---

## 5) `plaene.html`

### 5a) Vergleichstabelle Gruppe „Investment Planner" · Z. 702
Begründung: Neue Zeile „Screener" (gleicher Plan-Scope wie Planner-Module, also nur MAX).

- **alt (vereinfacht, nur die Gruppe):**
  ```
  Investment Planner
  ├── Positionsgröße & Einstiegsplan             – – ✓
  ├── Watchlist                                  – – ✓
  ├── Termine & News zur Watchlist               – – ✓
  ├── Trade-Journal mit KI-Hinweisen auf Denkfehler  – – ✓
  └── Rebalancing                                – – ✓
  ```
- **neu:**
  ```
  Investment Planner
  ├── Screener (Art, Börse, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite, Beta)  – – ✓
  ├── Positionsgröße & Einstiegsplan             – – ✓
  ├── Watchlist                                  – – ✓
  ├── Termine & News zur Watchlist               – – ✓
  ├── Trade-Journal mit KI-Hinweisen auf Denkfehler  – – ✓
  ├── Rebalancing (inklusive Cash-Korridor)      – – ✓
  └── Stresstest (mit Richtwert für den Rückgang) – – ✓
  ```

Konkrete Markup-Zeilen für die neuen Spalten (gleiches Muster wie Bestand):
```html
<tr><th scope="row">Screener nach Art, Börse, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite, Beta</th><td><span class="n" aria-label="nicht enthalten">–</span></td><td><span class="n" aria-label="nicht enthalten">–</span></td><td><span class="y" aria-label="enthalten">✓</span></td></tr>
<tr><th scope="row">Stresstest mit Richtwert für den Rückgang</th><td><span class="n" aria-label="nicht enthalten">–</span></td><td><span class="n" aria-label="nicht enthalten">–</span></td><td><span class="y" aria-label="enthalten">✓</span></td></tr>
```

Die bestehende Rebalancing-Zeile wird wortlich leicht ergänzt (Cash-Korridor als Zusatz im Zeilen-Header).

### 5b) Bilder / Mockups
Keine — Tabelle.

---

## 6) `llms.txt`

### 6a) Produkt-Zeile Investment Planner · Z. 13
Begründung: Konsistenz mit der neuen Meta-Beschreibung und FAQ-Antwort.

- **alt:** `[Investment Planner](https://gobetter.trade/investment-planner.html): Von der Watchlist bis zum Verkauf: Richtwerte, Stresstest, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Rebalancing und Trade-Journal.`
- **neu:** `[Investment Planner](https://gobetter.trade/investment-planner.html): Vom Screener bis zum Verkauf: Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Rebalancing, Stresstest und Trade-Journal. Der Screener filtert die Titel einer Börse nach Art, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite und Beta; Treffer gehen mit einem Klick auf die Watchlist oder in die Bewertung.`

### 6b) Produkt-Zeile Fair Value Pro · Z. 11
Begründung: Analystenschätzungen erscheinen in der App und sollen für Snippet-/LLM-Treffer klar als Vergleich (nicht als Rechengröße) markiert sein.

- **alt (Satzteil am Ende der Zeile):** `… Vorlagen und Bewertungen lassen sich in der Community mit anderen Autoren teilen.`
- **neu (neuer Satz davor eingefügt):** `… Analystenschätzungen zum Umsatz stehen als Vergleich daneben und gehen nicht in die Rechnung ein. Vorlagen und Bewertungen lassen sich in der Community mit anderen Autoren teilen.`

---

## 7) `ueber-gobetter.html`

### 7a) FAQ-Antwort „Was macht der Investment Planner?" — sichtbar · Z. 574
Begründung: gleichlautend mit der neuen Investment-Planner-FAQ und dem FAQPage-JSON-LD.

- **alt:** `Er begleitet eine Anlageentscheidung von der Watchlist bis zum Verkauf: Richtwerte, Stresstest, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing und Trade Journal. So hilft er dir, Entscheidungen am Kapitalmarkt mathematisch und psychologisch fundiert zu treffen.`
- **neu:** `Er begleitet eine Anlageentscheidung vom Screener bis zum Verkauf: Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest und Trade Journal. Der Screener filtert Kandidaten, die Watchlist beobachtet, der Rest hilft dir, Entscheidungen am Kapitalmarkt mathematisch und psychologisch fundiert zu treffen.`

### 7b) FAQ-Antwort in FAQPage-JSON-LD · Z. 22
Wörtlich identisch zu 7a an genau der gleichen Frage im `@graph`.

### 7c) Bilder / Mockups
Keine.

### 7d) FAQ-Antwort „Wie rechnet Fair Value Pro?" — sichtbar + FAQPage-JSON-LD
Begründung: Gleicher Zusatzsatz wie in `llms.txt` (6b). Die bestehende Antwort nennt bereits die vier Verfahren und den Komposit — hier den Vergleich zu Analystenschätzungen ergänzen, als klarer Zusatzsatz am Ende.

- **alt (Antwort-Satz am Ende der bestehenden Antwort):** `… Die Gewichtung ist einstellbar; Vorlagen für sieben Branchen setzen passende Werte.`
- **neu (neuer Satz direkt dahinter angefügt):** `… Die Gewichtung ist einstellbar; Vorlagen für sieben Branchen setzen passende Werte. Analystenschätzungen zum Umsatz stehen als Vergleich daneben und gehen nicht in die Rechnung ein.`

Wörtlich gleich im sichtbaren Antwort-Absatz und im FAQPage-JSON-LD-`acceptedAnswer.text` zur selben Frage.

---

## Reihenfolge für den Livegang

1. Abwarten: App-Release mit Screener, entfallenen Richtwerten, Startseite, Analystenschätzungen.
2. Vorschau-PR je Seite (einzeln oder gebündelt), nach dem üblichen Vorschau-Workflow — mit Audits und Playwright-Screenshots für `index.html` und `investment-planner.html`.
3. **Im Vorschau-PR zusätzlich prüfen:** Der selbsttätige Wechsel der drei Ansichten im App-Fenster am Handy (`#v-fvp` / `#v-pt` / `#v-ip`) und am iPad. Claude Design hat dafür in Export Nr. 84 die Planner-Ansicht fest eingeschaltet; die Website-Sitzung übernimmt die Prüfung. Konkret: Screenshot der automatisch wechselnden Ansicht in iPhone 14 (390 × 844) und iPad Mini (768 × 1024) über 2–3 Zyklen. Zusätzlich das Tablet-hoch-Verhalten 720–899 px ansehen (`.mods` aus, `.modsel` an) — Export Nr. 84 hat 720/899/900/1024 px schon geprüft, aber nur mit fest eingeschalteter Planner-Ansicht.
4. Je PR mergen nach Freigabe. Pages-Build grün abwarten, Live-Prüfung.
5. `llms.txt` zuletzt, da Index-Bots sie anstoßen, sobald die Seiten umgestellt sind.

## Offene Fragen (für Ludmila)

Aus Runde 2 beantwortet:
- Drei-Schritte-Startseite → nur FAQ-Eintrag in `investment-planner.html` (1j), sichtbar + im FAQPage-JSON-LD.
- Screener-Icon und Mockup-Inhalt für `index.html` (3b) → kommen aus Export Nr. 84.
- FVP-FAQ-Eintrag zu Analystenschätzungen → kommt (2d).
- Styleguide → bleibt unverändert.

Neu aus Export Nr. 84 (offene Frage 1 in LIESMICH):
- **Leere Fläche am Tablet hochkant** (720–899 px): Das App-Fenster der Startseite ist so hoch wie die höchste der drei Ansichten (Fair Value Pro). Ohne die lange Modulliste bleibt unter der Trefferliste gut ein Drittel des Fensters leer. **Füllen (zum Beispiel mit mehr Treffern) oder so lassen?** Vorschlag Website: so lassen — Treffer-Zahl ist eine App-Information, nicht beliebig aufblähbar. Entscheidung Ludmila.
