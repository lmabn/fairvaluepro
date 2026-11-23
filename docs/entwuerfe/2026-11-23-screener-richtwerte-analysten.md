# Entwurf: Website-Anpassungen nach App-Änderung Screener / Richtwerte / Analystenschätzungen

Stand: 23.11.2026 · Noch nichts an echten Seiten geändert · Website zieht nach, sobald App-Änderungen live sind.

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
- **neu (Z. 483, neue Karte „Screener"):**
  `<div class="card"><span class="ico" aria-hidden="true">⌕</span><h3>Screener</h3><p>Filtert dein Universum nach Art, Börse, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite und Beta. Treffer nach Börsenwert sortiert, mit einem Klick auf die Watchlist oder in die Bewertung. Kein Filter nach KGV oder Fairem Wert — das entscheidest du mit Fair Value Pro.</p></div>`

- **Stresstest-Karte (Z. 484):** Beschreibung um den Richtwert ergänzen.
  - **alt:** `Was kostet der schlechte Tag? Größter Sektor minus 30 %, größte Position auf null, Krypto halbiert — in Euro, gegen deine Grenze gestellt.`
  - **neu:** `Was kostet der schlechte Tag? Größter Sektor minus 30 %, größte Position auf null, Krypto halbiert — in Euro, gegen den Richtwert für den Rückgang, den du in den Einstellungen festgelegt hast.`

- **Rebalancing-Karte (Z. 491):** Cash-Korridor ergänzen.
  - **alt:** `Soll gegen Ist über alle Depots, mit konkreten Beträgen in Euro: was aufstocken, was kürzen.`
  - **neu:** `Soll gegen Ist über alle Depots, mit konkreten Beträgen in Euro: was aufstocken, was kürzen. Zeigt auch, ob dein Cash-Korridor eingehalten ist.`

- **Verkaufsprüfung-Karte (Z. 489):** unverändert.
- **Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Zinsrechner, Trade Journal:** unverändert.

**Hinweis Reihenfolge Karten-Grid:** Die neue Reihenfolge der Karten folgt der App (Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest, Trade Journal). Die aktuelle Reihenfolge im HTML ist: Meine Richtwerte, Stresstest, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Trade Journal — diese wird umsortiert.

### 1g) Hinweis-Block „Verbunden mit dem Ökosystem" · Z. 494
Begründung: Die Watchlist und Verkaufsprüfung nutzen weiter Fair Value Pro; neu auch der Screener für „Bewerten".

- **alt:** `Watchlist und Verkaufsprüfung greifen auf deine gespeicherten Fair-Value-Pro-Bewertungen zu. Rebalancing nutzt die Depot-Daten aus dem Portfolio Tracker. Alles mit demselben Account — kein Datendoppel.`
- **neu:** `Screener, Watchlist und Verkaufsprüfung greifen auf deine gespeicherten Fair-Value-Pro-Bewertungen zu. Rebalancing nutzt die Depot-Daten aus dem Portfolio Tracker. Alles mit demselben Account — kein Datendoppel.`

### 1h) Modul-Mockup Seitenleiste (Termine & News-Mockup) · Z. 552
Begründung: analog 1e.

- **alt:** `<div class="side"><span>Meine Richtwerte</span><span>Watchlist</span><span class="on">Termine & News</span>…`
- **neu:** `<div class="side"><span>Screener</span><span>Watchlist</span><span class="on">Termine & News</span>…`

### 1i) FAQ „Was ist der Investment Planner?" (sichtbar) · Z. 624
Wörtlich identisch zur FAQPage-Antwort in 1c.

- **alt:** `Werkzeuge für die Entscheidung vor dem Kauf: Meine Richtwerte, Positionsgröße, Einstiegsplan, Watchlist, Termine & News, Verkaufsprüfung, Stresstest, Rebalancing, Zinsrechner und Trade Journal.`
- **neu:** `Werkzeuge vor und nach dem Kauf, in der Reihenfolge der App: Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest und Trade Journal.`

### 1j) Neue Startseite „Drei Schritte, solange die Watchlist leer ist"
Begründung: Teil der App-Änderung. Auf der Website sollte das als neuer Hinweis sichtbar sein, damit SEO/Snippet-Treffer die Startseiten-Funktion kennen. Vorschlag: eine neue kleine Sektion unmittelbar nach dem Hero (vor den Modul-Mockups), oder ein neuer FAQ-Eintrag.

- **Vorschlag FAQ (neu, direkt nach „Was ist der Investment Planner?"):**
  - Frage: „Wie fange ich im Investment Planner an?"
  - Antwort: „Solange deine Watchlist leer ist, zeigt der Planner eine Startseite mit drei Schritten: im Screener Kandidaten filtern, Treffer auf die Watchlist nehmen, in Fair Value Pro bewerten. Danach füllen Watchlist, Termine & News und die übrigen Module den Alltag."
  - Gleichlautend im FAQPage-JSON-LD und im sichtbaren FAQ-Block.

### 1k) CSS-Kommentar · Z. 351
- **alt:** `/* 15.11.: zehn Modul-Karten ohne leere Fläche, Einstiegsplan breiter */`
- **neu:** `/* <neues Datum>: zehn Modul-Karten (Screener statt Richtwerte), Einstiegsplan breiter */`
Nur kosmetisch, keine Build-Wirkung.

### 1l) Bilder / Mockups
Keine SVG- oder PNG-Dateien als externe Assets betroffen. Die zwei sichtbaren App-Mockups bestehen aus HTML-Elementen (`.side` + `.mh` + Tabelle). Reines Markup-Update.

**Neu benötigt von Claude Design:**
- Icon-Glyph für „Screener" in `index.html` (siehe 3b). Ein `<svg viewBox="0 0 16 16">`-Icon im gleichen Stil wie die anderen Modul-Icons (Trichter/Filter-Form).
- Eventuell eigenes Mockup der Screener-Ansicht (Filter-Leiste + Treffer-Tabelle + „Auf die Watchlist" / „Bewerten"-Buttons) — wenn du einen dritten App-Mockup-Block auf der Seite willst, der Screener separat zeigt. Nicht zwingend.
- Mockup der „Drei-Schritte-Startseite" (optional, nur wenn 1j als Band und nicht als FAQ umgesetzt wird).

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
Keine weiteren Treffer zu „Richtwerte" oder „Screener" auf dieser Seite. Falls ein neuer FAQ-Eintrag zur Analystenschätzung gewünscht ist:

- **Vorschlag neuer FAQ-Eintrag (sichtbar + im FAQPage-JSON-LD, falls vorhanden):**
  - Frage: „Nutzt Fair Value Pro Analystenschätzungen in der Rechnung?"
  - Antwort: „Nein. Alle Berechnungen beruhen ausschließlich auf deinen Annahmen. Im Reverse DCF zeigt Fair Value Pro das erwartete Umsatzwachstum p. a. aus Analystenschätzungen zum Vergleich neben deiner Annahme, mit einem Diagramm zum Umsatz. Diese Schätzungen gehen nie in die Rechnung ein."

### 2e) Bilder / Mockups
- Neu von Claude Design benötigt: ein kleines Diagramm für den Umsatz-Vergleich (Reverse DCF: deine Annahme vs. Analystenschätzung p. a., plus historische Umsatzlinie). Dateiname-Vorschlag: `assets/svg/fvp-reverse-dcf-umsatz.svg`.

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
  <p>Vom Kandidaten bis zum Verkauf. Der Screener findet Treffer in deinem Universum, der Planner nimmt sie durch Watchlist, Einstiegsplan und Verkaufsprüfung — mit den Zahlen aus deinem Tracker.</p>
  <ul class="ticks">
    <li>Screener mit Filtern für Art, Börse, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite und Beta</li>
    <li>Einstiegsplan in Tranchen, Verkaufsprüfung vor jedem Verkauf</li>
    <li>Rebalancing und Stresstest auf deinem echten Bestand</li>
    <li>Zehn Module, alle mit deinen Daten verbunden</li>
  </ul>
  ```

### 3b) Modul-Mockup im IP-Band · Z. 761–786
Begründung: Aktives Modul wechselt von „Meine Richtwerte" zu „Screener"; Richtwerte-Icon raus, Screener-Icon rein; Beschreibungs-Zeile unterhalb anpassen; die zwei Kacheln (Rückgang + Cash-Korridor) und der kleine „Stresstest"-Block darunter werden ersetzt durch den neuen Screener-Inhalt.

- **alt (Z. 761 Modul-Selector-Chip):** `Meine Richtwerte`
- **neu (Z. 761 Modul-Selector-Chip):** `Screener`

- **alt (Z. 763, erstes Element in `.mods`, mit `class="on"`):** `Meine Richtwerte` + Richtwerte-Icon-SVG
- **neu (Z. 763, erstes Element in `.mods`, mit `class="on"`):** `Screener` + **neues Icon-SVG** (Trichter, Vorschlag):
  `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1.5 2.5h13l-5 6v5l-3-1.5v-3.5z"/></svg>`

- **alt (Z. 775 große Überschrift im Mockup-Inhalt):** `Meine Richtwerte` + Beschreibungs-Zeile „Deine Richtwerte, in Ruhe festgelegt, gegen den aktuellen Stand gestellt."
- **neu (Z. 775):** `Screener` + Beschreibungs-Zeile „Filtert dein Universum. Treffer mit einem Klick auf die Watchlist oder in die Bewertung."

- **alt (Z. 776–786, Inhalt: zwei `.tile`-Kacheln „Rückgang" + „Cash-Quote" und ein `.card.m-only`-Stresstest-Block):** zeigt die Richtwerte-Zahlen.
- **neu (Z. 776–786, Inhalt):** Vorschlag — eine kleine Filter-Leiste und eine Treffer-Tabelle.

  Entwurf-Markup (Beispielwerte, keine echten Firmen):
  ```html
  <div class="rw">
    <div class="tile"><span class="lab">Filter</span><span class="v" style="font-size:14px">Aktie · XETRA · Europa · Industrie · 1–10 Mrd. €</span><span class="n">8 weitere Filter · Sortiert nach Börsenwert</span></div>
    <div class="tile"><span class="lab">Treffer</span><span class="v">47</span><span class="n">Tabelle mit „Auf die Watchlist" und „Bewerten"</span></div>
  </div>
  <div class="card m-only">
    <span class="lab">Top-Treffer (Beispiel)</span>
    <div class="rows">
      <div><span>Beispiel AG</span><span>Börsenwert 7,2 Mrd. €</span></div>
      <div><span>Nordwerk SE</span><span>Börsenwert 4,1 Mrd. €</span></div>
      <div><span>Muster Industrie AG</span><span>Börsenwert 2,9 Mrd. €</span></div>
    </div>
  </div>
  ```

### 3c) Bilder / Mockups
- Neu von Claude Design benötigt: Icon-Glyph „Screener" (16×16 SVG im Stil der übrigen Modul-Icons).
- Optional: SVG-Mockup der Screener-Ansicht für Desktop (falls die HTML-Tile-Lösung oben zu abstrakt wirkt).

---

## 4) `partner.html`

### 4a) Accordion „Investment Planner — mehr dazu" · Z. 645
Begründung: Modul-Aufzählung nennt „Meine Richtwerte" an erster Stelle und zeigt eine veraltete Reihenfolge; neue Reihenfolge der App spiegeln.

- **alt:** `Zehn Module in einer Seitenleiste: Meine Richtwerte, Stresstest, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Trade Journal. Alle greifen auf deine gespeicherten Bewertungen und Depot-Daten zu. Das Trade Journal lässt dich eine Kaufbegründung festhalten, die nach dem Speichern nicht mehr änderbar ist; ein optionaler KI-Coach prüft sie gegen 16 bekannte Denkfehler. Teil des MAX-Plans.`
- **neu:** `Zehn Module in einer Seitenleiste in der Reihenfolge der App: Screener, Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Zinsrechner, Rebalancing, Stresstest, Trade Journal. Alle greifen auf deine gespeicherten Bewertungen und Depot-Daten zu. Der Screener filtert dein Universum und schickt Treffer direkt auf die Watchlist oder in die Bewertung. Das Trade Journal lässt dich eine Kaufbegründung festhalten, die nach dem Speichern nicht mehr änderbar ist; ein optionaler KI-Coach prüft sie gegen 16 bekannte Denkfehler. Teil des MAX-Plans.`

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
- **neu:** `[Investment Planner](https://gobetter.trade/investment-planner.html): Vom Screener bis zum Verkauf: Watchlist, Termine & News, Positionsgröße, Einstiegsplan, Verkaufsprüfung, Rebalancing, Stresstest und Trade-Journal. Der Screener filtert nach Art, Börse, Land, Sektor, Industrie, Börsenwert, Kurs, Dividendenrendite und Beta; Treffer gehen mit einem Klick auf die Watchlist oder in die Bewertung.`

Fair-Value-Pro-Zeile (Z. 11): kein Handlungsbedarf. Die neue Analystenschätzung ist eine interne Darstellungsergänzung, keine SEO-relevante Neuaussage für die Produkt-Kurzbeschreibung.

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

---

## Reihenfolge für den Livegang

1. Abwarten: App-Release mit Screener, entfallenen Richtwerten, Startseite, Analystenschätzungen.
2. Vorschau-PR je Seite (einzeln oder gebündelt), nach dem üblichen Vorschau-Workflow — mit Audits, Playwright-Screenshots für `index.html` und `investment-planner.html`.
3. Je PR mergen nach Freigabe. Pages-Build grün abwarten, Live-Prüfung.
4. `llms.txt` zuletzt, da Index-Bots sie anstoßen, sobald die Seiten umgestellt sind.

## Offene Fragen (für Ludmila)

1. **„Drei-Schritte-Startseite"** auf der Marketing-Seite als neuer Hero-Band, als neue Modul-Karte (im Screener-Mockup) oder nur als FAQ? (Vorschlag oben: FAQ-Eintrag.)
2. **Neues Icon für „Screener"** — soll ich einen Vorschlag selber zeichnen, oder wartet ihr auf Claude Design?
3. **FVP FAQ-Eintrag zu Analystenschätzungen** (2d) — gewünscht, ja/nein?
4. **Styleguide-Beispieltexte** für den Investment Planner (Z. 317 + Z. 789) — mit „Screener" ergänzen oder bewusst beispielhaft (Zinsrechner, Rebalancing, Watchlist, Trade-Journal) belassen?
