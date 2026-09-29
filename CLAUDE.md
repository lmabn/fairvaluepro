# Workflow

## Branch & Merge

- Entwicklung erfolgt auf dem vorgegebenen `claude/<task>`-Branch.
- **Am Ende jeder Aufgabe** automatisch:
  1. Branch nach `origin` pushen
  2. Pull Request gegen `main` öffnen (via `mcp__github__create_pull_request`)
  3. Auto-Merge aktivieren (via `mcp__github__enable_pr_auto_merge`, `merge_method: squash`)
- Direkter Push auf `main` ist serverseitig blockiert (HTTP 403) — immer den PR-Weg gehen.
- Keine zusätzliche Bestätigung für PR-Erstellung nötig; der Auto-Merge ist die User-Default-Präferenz.

## Offene To-dos

- **AGB überarbeiten** — `agb.html` enthält noch Fehler und ist unvollständig (u. a. § 2, § 7 nach BYOK-Umstellung, Personal-Plan-Klausel). Aktuell aus dem Footer aller Seiten ausgeblendet. Wenn der User an FVP-Marketing/Rechtssachen arbeitet: hier daran erinnern und AGB-Rework anbieten.
- **Legal-Seiten sind abgeschnitten** — `agb.html`, `impressum.html` und `datenschutz.html` enden mitten im letzten Wort, ohne `</main></body></html>` und ohne Footer. Content-Rekonstruktion + Struktur-Cleanup nötig, bevor die Seiten fürs Impressum/Datenschutz produktiv gestellt werden.
- **Produktseiten-Content-Sektionen** — Hero und CTA sind auf GO:BETTER Design v3 (Geist, gb-btn, gb-hero-product). Die Zwischen-Sektionen (`fair-value-pro.html` Manifesto/Problem/Outputs/How/Product/Features/Social/FAQ · `portfolio-tracker.html` und `trading-desk.html` Mockup/Features/Comparison/Pain/CTA) laufen noch mit Legacy-Klassen. Optik ist durch die Token-Bridge in `tokens.css` (Geist statt Manrope) grob v3-nah, aber Padding-Struktur und Card-Layout entsprechen noch dem alten Muster. Sektionsweise Redesign bei Bedarf.
- **Split-Hero-Layout** — `.gb-hero-split` ist in `gb-shell.css` implementiert (asymmetrisch: Text links, Mockup-Card mit Product-Farbrahmen rechts, Farb-Balken unten). Aktuell nicht aktiv verwendet, der User will erstmal die volle Höhe für die darunterliegenden Visualisierungen. Wenn Split-Hero doch aktiv werden soll: `<section class="gb-hero-split">` auf `fair-value-pro.html`, `portfolio-tracker.html`, `trading-desk.html` — Referenz-Screenshots sind im Design-Handoff `GO-BETTER_Produktseite_offline.html` und den beiden anderen `GO-BETTER_*_offline.html`.
- **Styleguide-Sync mit Design v3** — `styleguide.html` zeigt die Buttons in v2/v3-Mischform. Nach jedem größeren Design-System-Update ist ein Abgleich mit den aktuellen `gb-btn`-Größen (`--m` / `--l`), `data-ground`-Farben und `.on-photo`-Variante fällig.
- **`theme-toggle.js`** — auf allen Bestandsseiten und der Homepage nicht mehr geladen (Light/Dark-Toggle im neuen Design nicht mehr geplant). Nur `styleguide.html` lädt die Datei weiter, weil dort Dark/Light als Referenz nebeneinander gezeigt werden. Wenn der Styleguide auf reines Dark umgestellt wird, kann das Script komplett gelöscht werden.
- **Legacy-CSS in `style.css`** — die Datei ist noch aus der Vor-GO:BETTER-Ära (nav-Element-Selektoren, section {padding}, .btn-neon/.btn-outline/etc.). Wird von der neuen Shell überschrieben, aber Aufräumen würde die Codebasis deutlich vereinfachen. Vorsicht: Reste sind an Stellen wo Legacy-Klassen noch verwendet werden (siehe „Produktseiten-Content-Sektionen" oben).

## Merker · Marken- und Style-Regeln

- Produkt-Marke im Fließtext ist **"GO:BETTER"** (Wortmarke mit asymmetrischem Doppelpunkt). Nicht mehr „LMABN" — sweep war in PR #117, aber neuer Content trotzdem gleich richtig schreiben.
- Font-Familien: Fließtext **Geist** (`var(--font-sans)`), Zahlen und Codeblöcke **Geist Mono** (`var(--font-mono)`). Keine Manrope, kein Uppercase auf Buttons.
- Buttons ausschließlich `gb-btn` mit `gb-btn--m` (40 px) oder `gb-btn--l` (52 px) und `gb-btn--primary` / `gb-btn--secondary`. Auf Bildhintergrund: `.on-photo` an den Sekundär-Button.
- Product-Farben über `data-product="fvp|pt|td"` am `<body>` (oder am Section-Element) — die `--web-mood`-Variable schaltet auf FVP-Grün (#2EE59D), PT-Gelb (#F0B42A) oder TD-Blau (#5B8CFF).
- Layout: Bestandsseiten haben `.gb-nav-wrap` fixed und im Content `padding-top`-Reserve. Homepage `<body data-level="home">` hat die Nav `sticky`.
