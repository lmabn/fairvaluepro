# Website-Faktenbasis GO:BETTER

Stand: **23.11.2026**

Diese Datei fasst die Zahlen und Produktstände zusammen, auf die Sitzungen
der Website zurückgreifen (für Meta-Texte, FAQ, Mockups, llms.txt). Sichtbare
Website-Texte werden durch diese Datei nicht automatisch geändert — Updates
hier sind die Grundlage, nicht das Ergebnis.

Vorheriger Stand (ersetzt): **242 Funktionen, Stand 11.10.2026**.

## Werkzeuge und Zahlen

- **Drei Werkzeuge**: Fair Value Pro, Portfolio Tracker, Investment Planner.
- **Funktionen in drei Werkzeugen**: **290+** (gerundet, aus
  `docs/funktionsliste.md` der App übernehmen, sobald die genaue Zahl bekannt
  ist; die „290+" auf der Startseite ist eine konservative Untergrenze).
- **Verfahren im Komposit Fairen Wert**: DCF, KGV, EV/EBITDA; DDM als
  Vergleich.
- **Qualitätsnote**: 12 Kriterien.

## Investment Planner

Modulliste (Stand 23.11.2026, Reihenfolge der App):

1. Screener
2. Watchlist
3. Termine & News
4. Positionsgröße
5. Einstiegsplan
6. Verkaufsprüfung
7. Zinsrechner
8. Rebalancing
9. Stresstest
10. Trade Journal

- **„Meine Richtwerte" entfällt.** Der Richtwert für den Rückgang steht
  im Stresstest, der Cash-Korridor im Rebalancing; festgelegt werden
  beide in den Einstellungen.
- **Screener**: filtert Aktien, ETFs und Fonds nach Land, Branche und
  Größe. Treffer gehen per Klick auf die Watchlist oder in die Bewertung
  in Fair Value Pro.
- **Startseite bei leerer Watchlist**: drei Schritte — einen Titel
  finden (im Screener oder über eine Bewertung in Fair Value Pro); auf
  die Watchlist setzen und die Schwelle festlegen; den Kauf mit
  Positionsgröße und Einstiegsplan planen.

## Fair Value Pro

- **Analystenschätzungen** gehen nicht in die Rechnung ein. Sie stehen
  im Reverse DCF und am Regler „Stage-1 Wachstum" als Vergleich zur
  eigenen Annahme, dazu das Diagramm zum Umsatz im Reiter „Modell".

## Hinweis

Die App-Faktenbasis liegt in `docs/funktionsliste.md` des App-Repos. Diese
Datei hier ist das Website-Spiegelbild; bei Diskrepanz zwischen beiden
gilt die App-Datei.
