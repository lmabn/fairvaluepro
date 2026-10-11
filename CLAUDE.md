# Workflow

## Branch & Merge

- Entwicklung erfolgt auf dem vorgegebenen `claude/<task>`-Branch.
- **Am Ende jeder Aufgabe** automatisch:
  1. Branch nach `origin` pushen
  2. Pull Request gegen `main` öffnen (via `mcp__github__create_pull_request`)
  3. Auto-Merge aktivieren (via `mcp__github__enable_pr_auto_merge`, `merge_method: squash`)
- Direkter Push auf `main` ist serverseitig blockiert (HTTP 403) — immer den PR-Weg gehen.
- Keine zusätzliche Bestätigung für PR-Erstellung nötig; der Auto-Merge ist die User-Default-Präferenz.

## Pages-Deploy-Check nach Merge

Nach jedem Merge auf `main` **vor** jedem Purge-Hinweis an den User:

1. `gh api "repos/lmabn/fairvaluepro/actions/runs?per_page=5"` **ohne** `status`-Filter aufrufen.
2. In der Antwort nach dem Merge-Commit-SHA suchen.
3. Erst wenn `conclusion: success` für diesen SHA bestätigt ist, Purge-Hinweis geben.
4. Falls `status: queued` oder `in_progress` → warten und erneut prüfen; falls > 2 min hängend → User informieren. Ein `queued`-Lauf lässt sich **nicht** per „Re-run" neu anstoßen (nur „Cancel"); Weg ist dann abbrechen und einen trivialen Commit (z. B. `sitemap.xml` lastmod-Bump) nachschieben, der einen frischen Build triggert.
5. Falls **kein** Run existiert → max. 30 s warten (Trigger-Delay), sonst Alarm.

Hintergrund: Der Pages-Workflow ist GitHubs auto-generierter `pages build and deployment` (keine `.github/workflows/`-Datei im Repo, kein eigener Pfadfilter, kein eigenes concurrency-Setup). Wenn ein Build in `queued` hängt und nicht durchstartet, ist die Ursache **vermutlich** eine transiente GitHub-Infrastructure-Verzögerung (nicht belegt — reine Hypothese); Pages hat eine built-in Deployment-Concurrency, mehr nicht.

**Allgemeine Lehre (nicht nur Pages):** Ein Filter, der etwas ausblendet, erzeugt einen Fehlschluss, der wie ein Befund aussieht. „Keinen abgeschlossenen Lauf gefunden" ist nicht „kein Lauf gestartet". Wer eine Abfrage einschränkt, muss beim Auswerten denselben Vorbehalt mitführen. Das gilt für jede Suche, nicht nur für Workflow-Runs.

## FMP-Referral-Link

Monetarisiert — die Provision wird nur zugeordnet, wenn der Kauf über **unseren** Link läuft (nicht über ein früher per Link angelegtes Konto). Der Link ist bewusst mehrfach im HTML hartkodiert, nicht per JS gesetzt: ein Affiliate-Link darf nie an einem Skript hängen, sonst geht Geld verloren.

**URL (exakt):** siehe `EXPECTED` in `scripts/audit-fmp.sh` — dort steht sie einmal und ist die Quelle der Wahrheit.

**Regeln**
- Jeder FMP-Link steht als `<a>` im HTML mit `target="_blank"` und `rel="noopener"`.
- Der Query-Parameter (`couponCode=…`) bleibt unverändert — er trägt die Attribution.
- Der Gutscheincode darf **nirgends abtippbar** stehen (nicht im Fließtext, nicht in Readmes, nicht im Styleguide, nicht in dieser Datei). Der Rabatt kommt ausschließlich über den Link.
- Nach jedem Touch an einer FMP-Stelle lokal `bash scripts/audit-fmp.sh` laufen lassen. Das Script zählt die Vorkommen, prüft dass alle Links identisch sind und `target`/`rel` tragen, und schlägt bei abtippbarem Gutscheincode an. Exit-Code 0 = alles ok.

**Platzierungs-Haltung** — Direktlinks nur dort, wo der Nutzer **jetzt gerade** zum Anbieter will (daten-quelle.html: Setup-Steps, Partner-Card, zwei FAQs). Nicht in `agb.html` / `datenschutz.html` (Legal), nicht in `partner.html` (B2B-Pitch), nicht in `plaene.html` (dort läuft die Konversion über die Erklärseite `daten-quelle.html`).

**Formulierungs-Haltung** — Der Rabatt kommt NUR über den Link. Also nicht „wird automatisch angewendet", nicht „im Onboarding freigeschaltet", sondern aktiv: „25 % Rabatt bekommst du über unseren Partner-Link." Button mit identischem Satz dahinter.

**Offene Frage:** Gilt der Rabatt dauerhaft oder nur auf die erste Zahlung? Mail an Sam läuft. Falls einmalig → alle fünf Stellen präziser machen („auf die erste Zahlung" statt pauschal).

## Konvention — Produktaussagen vs. Sprachregeln

Was wir in CLAUDE.md über das Produkt festhalten, gehört in eine von zwei Schubladen:

- **Sprachregeln** sind Arbeitsanweisungen: wie wir Dinge nennen, formatieren, schreiben. Sie entstehen hier und gelten ab Eintrag.
- **Produktaussagen** sind Behauptungen darüber, wie das Produkt funktioniert — Zahlen, Mechaniken, Plan-Zuordnungen, Modul-Namen, Feldbeschriftungen. Ihre Quelle ist der Bauplan im App-Repo, nicht diese Datei. Jede Produktaussage trägt **Quelle** (welche Datei/welches Modul im App-Repo) und **Datum des Stands**. Wenn diese Kopie und der App-Code auseinanderfallen, ist der App-Code maßgeblich — hier wird nachgezogen, nicht umgekehrt.

## Gestaltung — Funktionen dreistufig darstellen

Für alle Seitenüberarbeitungen gilt: Funktionen erscheinen auf der Seite in drei Stufen, nicht als eine lange Liste.

1. **Fließtext** — verdichtet und verständlich. Was der Nutzer davon hat, nicht wie viele Schalter es gibt.
2. **Kompakte Übersicht** — knapp, scanbar.
3. **Details in Akkordeons** — zum Nachlesen bei Bedarf, eingeklappt.

Die Funktionsliste der App-Sitzung (Stand 2026-10-11: 245 Funktionen) ist **Prüfgrundlage, keine Inhaltsvorlage**. Sie sorgt dafür, dass nichts Wichtiges fehlt und nichts Falsches behauptet wird. Sie kommt nicht als Aufzählung auf die Seite.

Dasselbe gilt für die **Partnerseite**: Ludmila nutzt sie zum Sprechen, nicht zum Vorlesen. Sie muss vollständig genug sein, dass ihr nichts Wesentliches entgeht — und kurz genug, dass ein Zwölfjähriger folgen kann.

## Komposit Fairer Wert — Faktenbasis

### Sprachregeln

1. **Wortmarke**: durchgehend **„Komposit Fairer Wert"**. Keine Varianten (nicht „Komposit-Wert", nicht „Composite Fair Value", nicht „Komposit-Fairer-Wert" mit zwei Bindestrichen, keine Flexionsformen wie „Komposit-Fairen-Wert").
2. **„Fairer Wert" statt „Fair Value"** — durchgehend. Einzige Ausnahme: der Produktname **„Fair Value Pro"**.

### Produktaussage · Mechanik

> **Quelle:** Bauplan im App-Repo (Komposit-Logik in `frontend/trading-desk.js` und zugehörige Module). Änderungen an der Mechanik beginnen dort, nicht hier.
> **Stand:** 2026-10-11.

Sieben Modelle werden berechnet und nebeneinander gezeigt. Drei gehen in den gewichteten Komposit Fairen Wert: **DCF, KGV, EV/EBITDA**. Die Gewichtung ist **einstellbar**; voreingestellt **50 / 25 / 25**. Es gibt Vorlagen für sieben Branchen:

- **Konsum, Gesundheit, Konservativ**: 60 / 20 / 20
- **Energie, Edelmetalle, Rohstoffe**: 30 / 35 / 35
- **Banken, Versicherungen**: 0 / 100 / 0
- **Kryptowährungen**: kein Komposit Fairer Wert

Konsequenz für die Überarbeitung: „nutzergewichtet" (partner.html Z. 412, 413, 458) ist im Kern richtig; „fest 50/25/25" (sieben weitere Stellen) ist zu eng und muss die Einstellbarkeit sowie die Branchen-Vorlagen abbilden.

Nicht einzelfallweise umsetzen — die Site hat mindestens 26 Fundstellen in sechs Dateien zu dieser Mechanik; sie werden pro Seite als Ganzes nachgezogen.

## Daten-Aktualität — Faktenbasis

### Sprachregeln

Nicht belegt und damit **nicht verwenden**:
- **„Live-Kurse"** — es gibt keine Live-Kurse im Produkt.
- **„aktuell und vollständig" ohne Einschränkung** — zu pauschal.
- **Jede Angabe einer festen Verzögerung in Minuten** — ist nicht belegt.

### Produktaussage · Belegte Formulierungen

> **Quelle:** Bauplan im App-Repo (Daten-Pfad, von der App-Sitzung gemessen).
> **Stand:** 2026-10-11.

Drei Formulierungen, 1:1 verwendbar bei der Seitenüberarbeitung:

- **Kurse:** „Kurse und Finanzzahlen werden bei jeder Bewertung direkt bei deinem Datenanbieter abgerufen."
- **Finanzzahlen:** „Finanzzahlen sind so aktuell wie der letzte veröffentlichte Quartals- bzw. Jahresbericht."
- **Watchlist:** „In der Watchlist steht der Kurs deiner letzten Bewertung, mit Datum."

Konsequenz für die Überarbeitung:
- `fair-value-pro.html:668` „in Echtzeit" (H2 der Live-Vorschau-Section) kommt bei der Überarbeitung dieser Seite weg. Die Überschrift darf den Blick ins Werkzeug versprechen, nicht die Aktualität der Daten.
- `fair-value-pro.html:727` „aktuell und vollständig" fällt bei derselben Überarbeitung weg oder wird durch die oben belegte Kurs-/Finanzzahlen-Formulierung ersetzt.

## Funktionsliste — Faktenbasis

> **Quelle:** Bauplan im App-Repo (aus dem Code erhoben, nicht aus der Dokumentation).
> **Stand:** 11.10.2026 (App `main` e73e70d, Frontend bis 09a86f9).

**Datei:** `docs/funktionsliste.md` — 245 Funktionen in vier Bereichen (Kalkulator 79 · Tracker 64 · Planner 48 · Konto/Übergreifendes 54), je mit Was-sie-tut, Ab-Plan, Fundstelle `datei:zeile`, Website-Erwähnung und Status (aktiv / unsicher mit Grund). 126 Funktionen werden heute auf der Website nirgends erwähnt (Tracker über die Hälfte).

**Verwendung bei jedem Seiten-Entwurf** (zusätzlich zur Prüfung gegen Komposit- und Daten-Aktualität-Faktenbasis):
- **Stimmt, was dort steht?** — Jede Website-Aussage gegen die Liste prüfen.
- **Fehlt etwas, das dort hingehört?** — Lücken ermitteln (126 Kandidaten), dreistufig einbauen falls relevant.
- **Prüfgrundlage, keine Inhaltsvorlage.** Kommt nicht als Aufzählung auf die Seite — es gilt die dreistufige Darstellung aus „Gestaltung — Funktionen dreistufig darstellen".

**Zwingend zu beachten vor jedem Entwurf, nichts davon darf hinein:**
- Abschnitt **„Was Ludmila nicht sagen sollte, weil es heute nicht stimmt"** am Dateianfang (acht Punkte, u. a. 10-Jahre-Kursverlauf, 7-Modelle-Framing, Live-Kurse in der Watchlist, Krypto/Rohstoff-Erkennung, PDF-Sperre nur am Knopf, Tracker-Zeiträume).
- Abschnitt **„Offene Rechtsfragen"** (Widerrufs-Zustimmung, Konto-Löschung, AGB-Link nur im Kaufdialog) — ohne Bewertung, aber nichts davon in Entwürfe aufnehmen.

## Ablauf — Inhaltsüberarbeitung Seite für Seite

**Verbindlich** für die bevorstehende aussagenbasierte Überarbeitung aller Produktseiten.

**Grundregeln**
- **Eine Seite nach der anderen.** Nie zwei gleichzeitig.
- **Keine Einzeländerungen außerhalb der laufenden Seite** — Ausnahme: nachweislich falsche Aussagen (wie LIVE-Badges im Planner-Mockup) werden sofort entfernt, auf einem eigenen Mini-Branch.

**Reihenfolge**
1. `partner.html`
2. `fair-value-pro.html`
3. `portfolio-tracker.html`
4. `investment-planner.html`
5. `pro-mode.html`
6. `plaene.html`
7. `index.html`
8. `daten-quelle.html`

**Ablauf je Seite**
1. **Entwurf**: Der komplette Inhalt der Seite — alle Aussagen, dreistufig (Fließtext verdichtet, kompakte Übersicht, Details in Akkordeons; siehe Abschnitt „Gestaltung — Funktionen dreistufig darstellen").
2. **Faktenprüfung**: Entwurf gegen die Faktenbasis — Funktionsliste der App-Sitzung, Komposit-Beschreibung, Begriffsregeln, Datenaktualität.
3. **Vorlage**: Entwurf wird als Text (nicht als Code) vorgelegt.
4. **Umsetzung nach Freigabe**: Einmal umsetzen, Screenshots zeigen, Merge, Purge.
5. **Erst danach die nächste Seite.**

## Vorgemerkt für die Seitenüberarbeitung

Freigegebene Inhalte, die bei der Überarbeitung ihrer jeweiligen Seite wörtlich einzusetzen sind. Hier als Zeiger, damit nichts verlorengeht.

- **Seite 7 (`index.html`)** — Markentext für den Block unter Hero + Produktkarten (drei Absätze, letzte Zeile gestalterisch abgesetzt). Volltext und Gestaltungs-Hinweis in `docs/vorgemerkt-index-markentext.md`. Dort auch der Hinweis zur falschen „Sieben Modelle"-Zeile auf `index.html:415`, die bei derselben Überarbeitung nachgezogen wird.

## Offene To-dos

- **AGB überarbeiten** — `agb.html` enthält noch Fehler und ist unvollständig (u. a. § 2, § 7 nach BYOK-Umstellung, Personal-Plan-Klausel). Aktuell aus dem Footer aller Seiten ausgeblendet. Wenn der User an FVP-Marketing/Rechtssachen arbeitet: hier daran erinnern und AGB-Rework anbieten.
- **Legal-Seiten sind abgeschnitten** — `agb.html`, `impressum.html` und `datenschutz.html` enden mitten im letzten Wort, ohne `</main></body></html>` und ohne Footer. Content-Rekonstruktion + Struktur-Cleanup nötig, bevor die Seiten fürs Impressum/Datenschutz produktiv gestellt werden.
- **Produktseiten-Content-Sektionen** — Hero und CTA sind auf GO:BETTER Design v3 (Geist, gb-btn, gb-hero-product). Die Zwischen-Sektionen (`fair-value-pro.html` Manifesto/Problem/Outputs/How/Product/Features/Social/FAQ · `portfolio-tracker.html` und `investment-planner.html` Mockup/Features/Comparison/Pain/CTA) laufen noch mit Legacy-Klassen. Optik ist durch die Token-Bridge in `tokens.css` (Geist statt Manrope) grob v3-nah, aber Padding-Struktur und Card-Layout entsprechen noch dem alten Muster. Sektionsweise Redesign bei Bedarf.
- **Split-Hero-Layout** — `.gb-hero-split` ist in `gb-shell.css` implementiert (asymmetrisch: Text links, Mockup-Card mit Product-Farbrahmen rechts, Farb-Balken unten). Aktuell nicht aktiv verwendet, der User will erstmal die volle Höhe für die darunterliegenden Visualisierungen. Wenn Split-Hero doch aktiv werden soll: `<section class="gb-hero-split">` auf `fair-value-pro.html`, `portfolio-tracker.html`, `investment-planner.html` — Referenz-Screenshots sind im Design-Handoff `GO-BETTER_Produktseite_offline.html` und den beiden anderen `GO-BETTER_*_offline.html`.
- **Styleguide-Sync mit Design v3** — `styleguide.html` zeigt die Buttons in v2/v3-Mischform. Nach jedem größeren Design-System-Update ist ein Abgleich mit den aktuellen `gb-btn`-Größen (`--m` / `--l`), `data-ground`-Farben und `.on-photo`-Variante fällig.
- **`theme-toggle.js`** — auf allen Bestandsseiten und der Homepage nicht mehr geladen (Light/Dark-Toggle im neuen Design nicht mehr geplant). Nur `styleguide.html` lädt die Datei weiter, weil dort Dark/Light als Referenz nebeneinander gezeigt werden. Wenn der Styleguide auf reines Dark umgestellt wird, kann das Script komplett gelöscht werden.
- **Legacy-CSS in `style.css`** — die Datei ist noch aus der Vor-GO:BETTER-Ära (nav-Element-Selektoren, section {padding}, .btn-neon/.btn-outline/etc.). Wird von der neuen Shell überschrieben, aber Aufräumen würde die Codebasis deutlich vereinfachen. Vorsicht: Reste sind an Stellen wo Legacy-Klassen noch verwendet werden (siehe „Produktseiten-Content-Sektionen" oben).
- **Mobile-Hero: volle Viewport-Höhe vs. Produkt früher zeigen** — Auf `pro-mode.html`, `portfolio-tracker.html`, `investment-planner.html` und `daten-quelle.html` greift mobil `gb-shell.css:119` mit `.gb-hero-product--breathe .gb-hero-inner { min-height: calc(100dvh - var(--nav-h, 68px)); justify-content: center; }`. Hero füllt den ersten Viewport komplett, Pills sitzen zentriert, Mockup kommt erst nach einem Scroll. Bewusst, nicht Bug. Trotzdem: Entscheiden, ob diese Entscheidung bleibt oder ob das Produkt-Schaubild mobil früher sichtbar werden soll (z. B. `min-height: 0`, kürzere Hero-Padding mobil). Separat entscheiden, nicht nebenbei.

## Merker · Marken- und Style-Regeln

- Produkt-Marke im Fließtext ist **"GO:BETTER"** (Wortmarke mit asymmetrischem Doppelpunkt). Nicht mehr „LMABN" — sweep war in PR #117, aber neuer Content trotzdem gleich richtig schreiben.
- Font-Familien: Fließtext **Geist** (`var(--font-sans)`), Zahlen und Codeblöcke **Geist Mono** (`var(--font-mono)`). Keine Manrope, kein Uppercase auf Buttons.
- Buttons ausschließlich `gb-btn` mit `gb-btn--m` (40 px) oder `gb-btn--l` (52 px) und `gb-btn--primary` / `gb-btn--secondary`. Auf Bildhintergrund: `.on-photo` an den Sekundär-Button.
- Product-Farben über `data-product="fvp|pt|td"` am `<body>` (oder am Section-Element) — die `--web-mood`-Variable schaltet auf FVP-Grün (#2EE59D), PT-Gelb (#F0B42A) oder TD-Blau (#5B8CFF).
- Layout: Bestandsseiten haben `.gb-nav-wrap` fixed und im Content `padding-top`-Reserve. Homepage `<body data-level="home">` hat die Nav `sticky`.
