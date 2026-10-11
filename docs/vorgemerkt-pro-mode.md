# Vorgemerkt für Pro Mode (Seite 5)

## Gewichte-Tabelle · Branchen-Vorlagen für den Komposit Fairen Wert

Als Akkordeon „Gewichtung und Branchen-Vorlagen" direkt nach dem Funktionsraster einbauen (siehe Auftrag Pro-Mode-Überarbeitung, Punkt B). Zeile Kryptowährungen in neutraler Textfarbe, nicht grün. Entfernt aus `fair-value-pro-vorschau.html` beim Abschluss der FVP-Seite.

| Branche | DCF | KGV | EV/EBITDA |
|---|---|---|---|
| Voreinstellung (Standard) | 50 % | 25 % | 25 % |
| Konsum, Gesundheit, Konservativ | 60 % | 20 % | 20 % |
| Energie, Edelmetalle, Rohstoffe | 30 % | 35 % | 35 % |
| Banken, Versicherungen | — | 100 % | — |
| Kryptowährungen | — | — | — (kein Komposit Fairer Wert) |

HTML-Original (unverändert aus `fair-value-pro-vorschau.html` übernommen, inkl. der genutzten Styling-Klassen):

```html
<div class="verfahren-table-wrap">
  <table class="verfahren-table">
    <thead>
      <tr>
        <th>Branche</th>
        <th>DCF</th>
        <th>KGV</th>
        <th>EV/EBITDA</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Voreinstellung (Standard)</td><td>50 %</td><td>25 %</td><td>25 %</td></tr>
      <tr><td>Konsum, Gesundheit, Konservativ</td><td>60 %</td><td>20 %</td><td>20 %</td></tr>
      <tr><td>Energie, Edelmetalle, Rohstoffe</td><td>30 %</td><td>35 %</td><td>35 %</td></tr>
      <tr><td>Banken, Versicherungen</td><td class="no">—</td><td>100 %</td><td class="no">—</td></tr>
      <tr><td>Kryptowährungen</td><td colspan="3" class="part">kein Komposit Fairer Wert</td></tr>
    </tbody>
  </table>
</div>
```

Dazugehöriges CSS (ebenfalls aus `fair-value-pro-vorschau.html` übernommen):

```css
.verfahren-table-wrap { overflow-x: auto; margin: 0 0 1rem; scrollbar-width: thin; scrollbar-color: var(--border2) transparent; }
.verfahren-table-wrap::-webkit-scrollbar { height: 6px; }
.verfahren-table-wrap::-webkit-scrollbar-track { background: transparent; }
.verfahren-table-wrap::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
.verfahren-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; min-width: 560px; }
.verfahren-table th, .verfahren-table td { white-space: nowrap; min-width: 90px; padding: 0.7rem 0.8rem; border-bottom: 1px solid var(--border); color: var(--text-2); vertical-align: middle; }
.verfahren-table th:first-child, .verfahren-table td:first-child { min-width: 200px; white-space: normal; }
.verfahren-table th { font-family: var(--mono); font-size: 0.55rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-3); text-align: left; background: var(--bg3); }
.verfahren-table td:first-child { color: var(--text); font-weight: 500; }
.verfahren-table tr:last-child td { border-bottom: none; }
.verfahren-table .no { color: var(--text-3); }
.verfahren-table .part { color: var(--mid); }
```

Hinweis zur Farbkorrektur Kryptowährungen: Die Klasse `.part` ist derzeit `color: var(--mid)` — in neutraler Textfarbe (`var(--text-3)` oder analog), nicht grün. Dieselbe Farbkorrektur ist auch auf `partner.html` (Komposit-Branchen-Tabelle, Zelle „kein Komposit Fairer Wert") nachzuziehen.
