# Funktionsliste der App

**Stand:** 11.10.2026, Code auf `main` (e73e70d, Frontend unverändert bis 09a86f9). **Website:** alle 20 öffentlichen Seiten von gobetter.trade laut Sitemap, plus Partnerseite und AGB, als Text abgerufen am 11.10.2026.

**Zweck:** Gesprächsgrundlage für die Partnerseite. Die Liste ist aus dem Code erhoben, nicht aus der Dokumentation. Die Spalte „Auf der Website" zeigt, wo eine Funktion schon beschrieben ist, und wo nicht. „nicht erwähnt" heißt: Lücke.

**Spalten:**
- **Ab Plan:** GO = Gast ohne Konto (Kurzansicht), PRO = registriert, kostenlos (volles Dashboard, eigene Vorlagen lokal), MAX = Abo (Analysen speichern, Vorlagen-Sync, Suchverlauf, PDF, Tracker, Planner, KI-Assistent). Quelle `plans.py` (TIERS), geprüft an der Freischaltung im Code. Zusätze wie „+ eigener FMP-Schlüssel" oder „+ eigener KI-Schlüssel" sind Voraussetzungen über den Plan hinaus.
- **Fundstelle:** `datei:zeile`; ohne Datei im Kalkulator-Teil ist `frontend/fairvaluepro.html` gemeint. Pfade relativ zu `frontend/` bzw. Repo-Wurzel.
- **Auf der Website:** Seite (ohne `.html`) und ein kurzes Zitat. Akademie-Seiten zählen nur, wo die App-Funktion gemeint ist.
- **Status:** „aktiv" oder „unsicher" mit Grund — unsicher heißt: möglicherweise toter Code, nie eingeblendet oder in der Produktion anders wirksam. Nicht geraten.

**Erhoben** in vier Bereichen, je aus dem Code gelesen; Stichproben der Fundstellen und die auffälligsten Aussagen wurden nachgeprüft (u. a. Live-Kurs nur im Entwicklungsmodus `td_api.py:393`, Widerrufs-Zustimmung `plans.py:66–76` gegen AGB § 356, Konto-Löschung `auth.py:795`, Kursverlauf fünf statt zehn Jahre).

## Übersicht

| Bereich | Funktionen | davon auf der Website nicht erwähnt | davon unsicher |
|---|---|---|---|
| Fair Value Pro (Kalkulator) | 79 | 36 | 7 |
| Portfolio Tracker | 64 | 37 | 1 |
| Investment Planner | 48 | 22 | 2 |
| Konto, Schlüssel, Kauf und Übergreifendes | 54 | 31 | 3 |
| **Gesamt** | **245** | **126** | **13** |

**Was Ludmila nicht sagen sollte, weil es heute nicht stimmt** (aus den Auffälligkeiten unten, je mit Beleg):
- „Kursverlauf über zehn Jahre": heute fünf Jahre (Abruf ohne Startdatum, `fmp_provider.py:613–625`; Bauplan, vorrangig).
- „Sieben Modelle": fünf rechnen einen fairen Wert, das Median-KGV liefert eine Linie, die FCF-Rendite ist eine Kennzahl (Bauplan, Abschnitt „Verbindlich: Wie der Komposit Fairer Wert entsteht").
- „Komposit 50/25/25": nur die Grundeinstellung, einstellbar, sieben Branchen-Vorlagen mit eigenen Werten.
- „Live-Kurse in der Watchlist": auf dem Live-Server nie, dort gilt der Kurs der gespeicherten Analyse mit Datum (`td_api.py:384–404`).
- „Alle Unternehmen weltweit": Werte außerhalb der USA brauchen einen höheren Plan beim Datenanbieter.
- Tracker-Zeiträume „1J · 2J · 3J · 5J · Alle": im Code ohne 2J, dafür 1M, 3M, 6M, YTD, 10J, 15J.
- „Krypto und Rohstoffe funktionieren": Bitcoin und Gold bekommen heute keinen Kurs und zeigen dann eine falsche Einordnung (Nachtrag 11.10.2026, Bauplan).
- „PDF-Export nur für MAX": Nur der Knopf ist in Nicht-MAX-Plänen gesperrt; der Druckdialog des Browsers liefert dieselbe Druckansicht (Nachtrag 11.10.2026, Bauplan — siehe auch Auffälligkeit Kalkulator Nr. 1).

**Offene Rechtsfragen, ohne Bewertung:** Widerrufs-Zustimmung — die AGB versprechen zwei getrennte Zustimmungen (§ 356 Abs. 5 BGB), die App hat einen kombinierten Haken mit anderem Wortlaut (`plans.py:66–76`). Konto löschen entfernt auch den gespeicherten Nachweis dieser Zustimmung (`auth.py:795`). AGB-Link nur im Kaufdialog (Bauplan).

## Fair Value Pro (Kalkulator)

Ohne Dateiangabe ist `frontend/fairvaluepro.html` gemeint.

| Funktion | Was sie tut (ein Satz, ohne Fachsprache) | Ab Plan | Fundstelle (datei:zeile) | Auf der Website | Status |
|---|---|---|---|---|---|
| **Zugang & Einstieg** | | | | | |
| Anmelden / Registrieren / Passwort zurücksetzen | Ein Anmeldefenster vor dem Kalkulator bietet Login, Konto anlegen und „Passwort vergessen“. | GO (mit Konto: PRO) | :1475, :1537, :1605, :1780 | fair-value-pro „PRO und MAX mit Registrierung“ | aktiv |
| „Ohne Account testen“ | Öffnet den Kalkulator ohne Konto, fest in der Kurzansicht. | GO | :1589 | plaene „Ohne Account starten“ | aktiv |
| Hinweis „Marktdaten-Schlüssel nötig“ | Ohne eigenen Schlüssel beim Datenanbieter gibt es keine Daten; ein Fenster führt zur Einrichtung. | GO | :4513–4527, api.py:2271 | daten-quelle „Ticker eingeben, Bewertung starten.“ | aktiv |
| Liste „Welche Werte funktionieren in meinem Plan?“ | Fest hinterlegte Liste getesteter US-Werte; ein Klick startet die Bewertung. Sie erscheint auch, wenn der Anbieter-Plan einen Wert nicht abdeckt. | GO | :2014, :4430, :4542, :4561 | nicht erwähnt | aktiv |
| Startkarten (3 Schritte) | Ein Klick auf eine Karte hebt das Suchfeld, die Vorlagen oder den Start-Knopf hervor. | GO | :2775, :6785–6815 | fair-value-pro „In drei Schritten zur Bewertung.“ | aktiv |
| Direktlinks | `?ticker=` (aus Tracker/Planner) füllt den Wert ein und rechnet; `?openDocs` und `?openAccount` öffnen Hilfe bzw. Konto. | GO | :1436–1460, :6690–6735 | nicht erwähnt | aktiv |
| Kauf-Einstieg | Ein gesperrtes Merkmal oder der Website-Link `?checkout=max` öffnet nach der Anmeldung das MAX-Kauffenster. | PRO (Konto nötig) | auth.js:172, upgrade-modal.js:109, :333, :1510 | plaene „Buchung läuft direkt in der App“ | aktiv |
| **Suche & Auswahl** | | | | | |
| Suche nach Kürzel und Firmenname | Ab 2 Zeichen bis zu 8 Treffer direkt vom eigenen Datenanbieter (Kürzel- und Namenssuche zusammengeführt). | GO | :4211–4227, :4232, :4239, data_provider_registry.js:281 | fair-value-pro „Beliebiges Kürzel eintippen — AAPL, SAP, ALV“ | aktiv |
| US-Börsen zuerst / „Andere Börsen einblenden“ | Zuerst US-Treffer, andere Börsen auf Klick, mit dem Hinweis auf einen höheren Anbieter-Plan. | GO | :4247–4290 | nicht erwähnt | aktiv |
| ℹ-Details zum Treffer | Zeigt vor der Auswahl Branche, Land, Börsenwert, ISIN und eine Kurzbeschreibung. | GO | :4304 | nicht erwähnt | aktiv |
| Ersatzsuche über den Server mit Rohstoffliste | Scheitert die Direktsuche, sucht der Server und ergänzt rund 25 Rohstoff-Kürzel (Gold, Öl, Weizen …). | GO | :4342, api.py:2237, api.py:1739 | nicht erwähnt | aktiv (die Rohstoffliste gibt es nur auf diesem Ersatzweg) |
| WKN-/ISIN-Suche | Das Suchfeld nennt „WKN“, eine eigene Umsetzung von WKN oder ISIN in ein Kürzel gibt es aber nicht; die Eingabe geht unverändert an die Anbieter-Suche. | GO | :2009, :4239 | nicht erwähnt | unsicher: kein WKN/ISIN-Code, Treffer hängen allein vom Anbieter ab |
| ISIN/WKN im Kopf der Bewertung | Zeigt die ISIN und bei deutschen ISINs eine daraus abgeleitete WKN. | PRO | :4736–4739, api.py:1374 | nicht erwähnt | aktiv |
| Auswahlanzeige + ✕ | Zeigt den gewählten Wert; ✕ setzt alles zurück. | GO | :4368, :2026, :1709 | nicht erwähnt | aktiv |
| **Bewertungsverfahren & Komposit** | | | | | |
| Bewertung starten | Lädt die Daten und rechnet alle Verfahren für den gewählten Wert. | GO | :4616, api.py:2344 | daten-quelle „Ticker eingeben, Bewertung starten.“ | aktiv |
| DCF (zweistufig) | Rechnet die künftigen Geldüberschüsse (5 Jahre Wachstum, dann Übergang) auf heute zurück. | PRO (GO nur im Komposit) | fair_value_calculator.py:1298, :4778 | fair-value-pro „Two-Stage-DCF-Modell“ | aktiv |
| KGV-Wert | Gewinn je Aktie mal Ziel-KGV. | PRO | fair_value_calculator.py:1397, :4909 | fair-value-pro „KGV-Methode (25 %)“ | aktiv |
| EV/EBITDA-Wert | Wert aus dem Ziel-Vielfachen auf den Betriebsgewinn vor Abschreibungen. | PRO | fair_value_calculator.py:1397, :4909 | fair-value-pro „EV/EBITDA (25 %)“ | aktiv |
| Historisches 5-Jahres-Median-KGV | Vergleichswert aus dem eigenen mittleren KGV der letzten 5 Jahre; auch als Linie im Kursverlauf. | PRO | api.py:2462, api.py:3130–3158, :4945 | fair-value-pro „historisches Median-KGV und FCF-Rendite“ | aktiv |
| DDM (Dividendenmodell) | Wert aus den Dividenden; die Karte erscheint ab 25 % Ausschüttungsquote, sonst ausgegraut mit Begründung. | PRO | fair_value_calculator.py:1304, :4959 | fair-value-pro „Gordon Growth Model (DDM)“ | aktiv |
| PEG-Wert | Wert aus KGV im Verhältnis zum Gewinnwachstum. | PRO | fair_value_calculator.py:1349, :4978 | fair-value-pro „PEG-Ratio als Wachstums-Schnellcheck“ | aktiv |
| FCF-Rendite | Freier Geldüberschuss geteilt durch den Börsenwert, verglichen mit 4,3 % Anleihezins. Eine Kennzahl, kein fairer Wert. | PRO | api.py:2456–2465, :4930 | fair-value-pro „FCF-Rendite“ (dort als Modell gezählt) | aktiv |
| ROIC gegen WACC | Vergleicht die Kapitalrendite mit den Kapitalkosten (wertschaffend oder nicht). | PRO | fair_value_calculator.py:1455, :4815 | pro-mode (Beispielbild) „ROIC 7.9 % Spread −2.1 %“ | aktiv |
| Komposit Fairer Wert | Gewichteter Mittelwert aus DCF, KGV und EV/EBITDA; fehlende Bausteine fallen heraus, die Gewichte werden neu verteilt. | GO (Kurzansicht) | :4701–4707, api.py:2683–2688 | fair-value-pro „gewichteter Komposit-Wert: 50 % DCF, 25 % KGV“ | aktiv |
| Komposit-Gewichte einstellen | Regler für DCF, KGV und EV/EBITDA, auf 100 % umgerechnet; Branchen-Vorlagen setzen eigene Werte. | GO (Seitenleiste) | :2566, :3925, :4197, :3277–3292 | nicht erwähnt (Website nennt feste 50/25/25) | aktiv |
| Krypto-Bewertung | Nur 200-Tage-Linie als Bezug, bei Bitcoin zusätzlich die Mining-Kosten als Untergrenze; Aktienkennzahlen ausgegraut; kein Komposit. | GO | api.py:1999, :2052, :4838 | nicht erwähnt | unsicher: erkennt nur Kürzel wie BTC-USD; das in der Werteliste als „verifiziert“ geführte BTCUSD passt nicht zur Erkennung (:4447) |
| Rohstoff-Bewertung | Produktionskosten als Untergrenze (Gold, Öl …) plus 200-Tage-Linie, 5-Jahres-Mittel und 52-Wochen-Spanne. | GO | api.py:1761, :1779, :1804 | nicht erwähnt | unsicher: erkennt nur „=F“-Kürzel oder die feste Liste; GCUSD aus der Werteliste passt nicht |
| Banken-/Versicherer-Hinweis | Weist darauf hin, dass DCF und EV/EBITDA bei Banken und Versicherern nicht tragen. | PRO | :4762, :6256 | nicht erwähnt | aktiv |
| **Einstellungen & Vorlagen** | | | | | |
| Branchen-Vorlagen (11) | Setzen WACC, Endwachstum, Prognosejahre, EV/EBITDA-Ziel und Komposit-Gewichte je Branche (Tech, Konsum, Finanzen, Banken, Energie, Gesundheit, Konservativ, Edelmetalle, Rüstung, Krypto, Rohstoffe). | GO | :3277–3293, :3779, :2060 | plaene „Branchen-Presets“ | aktiv |
| Wachstumsdeckel der Vorlage „Konservativ“ | Begrenzt das automatisch ermittelte Wachstum auf 5 %. | GO | :3288, :4178, api.py:2352 | nicht erwähnt | aktiv |
| Vorlagen „Krypto“ und „Rohstoffe“ | „Krypto“ setzt alles auf Automatik und blendet einen Hinweis ein; „Rohstoffe“ setzt Aktien-Parameter. | GO | :3291–3292, :3788 | nicht erwähnt | unsicher: die Rohstoff-Bewertung (api.py:1804) nimmt keine Parameter an; der Rohstoff-Hinweis (:2048) wird nie eingeblendet |
| Erweiterte Einstellungen | Regler für Wachstum Jahr 1–5, WACC, Endwachstum, Prognosejahre, FCF-Basis, risikofreien Zins, Beta, Risikoprämie, Ziel-KGV, Ziel-EV/EBITDA, Dividendenwachstum, Bär-/Bulle-Faktor, Szenario- und Komposit-Gewichte, Steuer, Fremdkapitalkosten; jedes Feld steht auf „Auto“, bis man es verstellt. | GO | :2134–2629, :4168 | plaene „Erweiterte Einstellungen (WACC, Beta, ERP)“ | aktiv |
| ?-Erklärungen an den Reglern | Ein Klick auf „?“ klappt eine kurze Erklärung zum Feld auf. | GO | :6768 | fair-value-pro „Erklärungen direkt im Tool“ | aktiv |
| Anpassungsleiste, Zurücksetzen, Neu berechnen | Zeigt, dass eigene Annahmen aktiv sind, setzt sie zurück oder rechnet neu. | GO | :2697, :2628, :3879 | pro-mode (Beispielbild) „Bewertung mit angepassten Annahmen ↺ Zurücksetzen“ | aktiv |
| Eigene Vorlagen (lokal) | Speichert die aktuellen Einstellungen unter einem Namen im Browser; Laden und Löschen per Chip. | PRO | :3406, :3444–3472 | plaene „Eigene Presets lokal (inkl. Export + Import)“ | aktiv |
| Vorlagen exportieren / importieren | Lädt die Vorlagen als Datei herunter bzw. liest eine solche Datei ein. | PRO | :3474, :3493 | fair-value-pro „Vorlagen als Datei exportieren, teilen“ | aktiv |
| Vorlagen übers Konto abgleichen | Beim Speichern oder Importieren geht die Vorlage zusätzlich ans Konto; im Konto-Reiter „Vorlagen“ anlegen, anwenden, löschen. | MAX | :3428, :3434, account.js:586, :646, api.py:754 | plaene „Preset-Sync über alle Geräte“ | aktiv (nur Hochladen: lokales Löschen wird nicht übertragen, Konto-Vorlagen erscheinen nicht in der Seitenleiste) |
| Stage-2-Verlauf, PEG-Zielwert | Der Server kann beide verstellen, im Kalkulator gibt es kein Bedienelement dafür. | – | api.py:2390, :2397 | nicht erwähnt | unsicher: kein Bedienelement |
| **Ansichten & Ergebnis-Anzeige** | | | | | |
| Kurzansicht (Minimal) | Vier Karten: Kurs, Komposit Fairer Wert, KGV, Qualitätsnote, dazu die Einordnung. | GO | :6532, :2810–2847 | plaene „Minimal Mode — 4 Kennzahlen“ | aktiv |
| Pro-Ansicht / Umschalter | Zeigt das volle Dashboard; Gäste bekommen beim Umschalten das Registrierungsfenster. | PRO | :6428, :6441, :1963 | plaene „Pro Mode — volles Dashboard“ | aktiv |
| Reiter Kursverlauf / Unternehmen / Modell | Gliedern das Dashboard; leere Reiter verschwinden, beim Drucken werden alle gezeigt. | PRO | :2990, :6116, :6127, :6167 | nicht erwähnt | aktiv |
| Kopfkarten | Kurs mit Börsenwert, DCF-Wert mit Marge, Komposit mit Gewichten, ROIC, FCF-Rendite, Qualitätsnote. | PRO | :2893–2910, :4769 | pro-mode „Komposit Fairer Wert“ | aktiv |
| Einordnung (Hauptaussage) | Fünf Stufen von „Stark unterbewertet“ bis „Stark überbewertet“, abgeleitet aus dem Abstand zum fairen Wert. | GO | :4742, :6363 | pro-mode „Stark überbewertet — Kurs liegt 687 %“ | aktiv |
| Modellgrenze | Erklärt kein Verfahren den Kurs (auch der Reverse DCF nicht), steht neutral „Außerhalb des Modellbereichs“ statt einer Marge; auch in Kurzansicht, Teilen und Kaufschwelle. | GO | model_boundary.js:44, :4723 | nicht erwähnt | aktiv |
| „Fairer Wert — heute“ (Skala) | Skala mit Bär-bis-Bulle-Zone, Markern je Verfahren und Kurspfeil, darunter eine Werteliste. | PRO | :4998, :3027 | pro-mode (Beispielbild) „Komposit Fair Value“ | aktiv |
| Warnhinweise zur Aussagekraft | Warnt, wenn der DCF schwach oder gar nicht gewichtet ist, WACC und Endwachstum nah beieinander liegen, der WACC auf 6–20 % begrenzt wurde oder die Verfahren weit auseinanderliegen. | PRO | :6273, :4925, :6350 | nicht erwähnt | aktiv |
| Hinweis auf unvollständige Daten | Nennt fehlende Daten, nicht berechnete Bausteine und woraus der Komposit gebildet ist; beim Speichern kommt eine Rückfrage. | PRO | :4053, api.py:1564, account.js:386 | nicht erwähnt | aktiv |
| Value-Falle-Hinweis | Bei günstigem Kurs und Note C/D (oder B mit verfehltem Wachstum) erscheint ein Hinweis mit bis zu drei verfehlten Kriterien. | PRO | :4032 | nicht erwähnt (nur Akademie) | aktiv |
| Kaufschwelle + Sicherheitsmarge | Komposit × (1 − Marge) und Abstand zum Kurs; Marge per Regler 0–60 %, Vorschlag je Qualitätsnote (A 15 % … D 40 %). | PRO | :6326, :2974, :6282 | fair-value-pro „Wachstumsrate, WACC, Sicherheitsmarge.“ | aktiv |
| Eigene Margen-Vorschläge je Note | Übernimmt die eigenen Vorschläge aus dem Konto (Reiter „Einstellungen“). | MAX | :6311, tracker_api.py:1017, account-modal.js:398 | nicht erwähnt | aktiv |
| Reverse DCF | Zeigt, welches Jahreswachstum der heutige Kurs voraussetzt, verglichen mit der eigenen Annahme und der Vergangenheit. | PRO | :3959, api.py:2723 | nicht erwähnt (nur Akademie) | aktiv |
| **Reiter Kursverlauf** | | | | | |
| Kurs gegen fairen Wert | Diagramm über 1, 3, 5, 10 Jahre oder Max; Linien für Börsenkurs, EPS×KGV, DCF, Bär-bis-Bulle-Zone und Median-KGV einzeln ein- und ausblendbar. | PRO | :3006–3019, :5112, :5193, api.py:3068, :3089 | pro-mode „Kursverlauf vs. Fairer Wert“ | unsicher: „10J“ und „Max“ liefern heute nur fünf Jahre (Bauplan, Abruf ohne Startdatum, fmp_provider.py:613–625) |
| **Reiter Unternehmen** | | | | | |
| „Das Unternehmen“ | Beschreibung (englisch), Branche, Sektor, Land, Börse, Mitarbeiter, Website. | PRO | :3998 | nicht erwähnt | aktiv |
| Umsatz, EBIT, Nettogewinn | Balkendiagramm über bis zu 10 Jahre. | PRO | :5346, :5378, api.py:3166 | nicht erwähnt | aktiv |
| Bruttomargen-Verlauf | Linie plus ein Satz, ob die Preissetzungsmacht hält. | PRO | :5401 | nicht erwähnt | aktiv |
| Klassische Kennzahlen | KGV, KBV, KUV, Dividendenrendite, Gewinn je Aktie, 52-Wochen-Spanne. | PRO (KGV auch GO) | :6210, :3119 | nicht erwähnt | aktiv |
| Qualitäts-Scorecard | 12 Kriterien ergeben eine Note von A bis D; unter 9 bewertbaren gibt es „n. b.“, fehlende Kriterien werden genannt. | GO (Note), Details PRO | :5446, :5476, api.py:3365–3641 | pro-mode „Automatische Benotung von A bis D.“ | aktiv |
| **Reiter Modell** | | | | | |
| Modell-Transparenz | Tabelle aller Annahmen, jeweils markiert als automatisch, angepasst oder gedeckelt. | PRO | :4100, :3157 | pro-mode „Modell-Transparenz Jede Annahme sichtbar“ | aktiv |
| Relative Bewertung | Karten für KGV, EV/EBITDA, Median-KGV, DDM und PEG mit Marge je Verfahren und Angabe der Branchenquelle. | PRO | :4909 | pro-mode „Relative Bewertung“ | aktiv |
| Szenarien Bär / Basis / Bulle | Drei Werte mit eigenem Wachstum und WACC plus ein gewichteter Gesamtwert. | PRO | :4826, fair_value_calculator.py:1153, :1428 | pro-mode „Drei-Szenarien-Analyse“ | aktiv |
| WACC-Aufschlüsselung | Alle Bestandteile der Kapitalkosten plus ROIC-Balken. | PRO | :4888 | pro-mode „WACC-Aufschlüsselung“ | aktiv |
| Sensitivitätstabelle | Wachstum × WACC ergibt den fairen Wert, farbig markiert. | PRO | :5306, fair_value_calculator.py:1482 | pro-mode „Sensitivitätstabelle“ | aktiv |
| Haupteinflussfaktoren | Balken zeigen, welche Annahme den Wert am stärksten bewegt. | PRO | :5321, fair_value_calculator.py:1497 | nicht erwähnt | aktiv |
| DCF-Cashflow-Prognose | Diagramm mit Wachstum je Jahr, Cashflow-Linie und der Methode für den Basis-Cashflow. | PRO | :5329, :3238 | nicht erwähnt | aktiv |
| **Kurs & Währung** | | | | | |
| Währung USD / EUR | Schaltet die Anzeige um; der Browser holt den Wechselkurs und merkt ihn sich eine Stunde. | GO | :4405, :4381 | fair-value-pro „zwischen USD und EUR umschalten“ | aktiv |
| Kurs aktualisieren | Knopf holt einen frischen Kurs (Farbblitz bei Änderung, „Stand: vor X Min“); beim Zurückkehren in den Tab nach über 60 s einmal automatisch. | GO | :3660–3767, api.py:2975 | nicht erwähnt | aktiv |
| **Speichern & Pflege** | | | | | |
| Analyse speichern | Speichert mit Name, Notiz und geprüfter ISIN; gibt es den Wert schon, wählt man „Ersetzen“ oder „Zusätzlich anlegen“. Ohne Obergrenze. | MAX | account.js:312, :366, api.py:640, :2707 | plaene „Analysen speichern — unbegrenzt“ | aktiv |
| Meine Werte | Liste der gespeicherten Analysen; ein Klick stellt Bewertung und Einstellungen wieder her, Löschen möglich, Marke „veraltet“ nach 90 Tagen (einstellbar). | MAX | account.js:264, :490, :568 | nicht erwähnt | aktiv |
| Quartals-Check | Je Titel damals gegen heute (fairer Wert, Marge, Note, Kurs); „Neu berechnen“ mit den alten Annahmen; über 20 % Abweichung markiert; Abgleich mit dem Falsifikationsgrund aus dem Planner; Marke „im Depot“. | MAX | account.js:118–262, tracker_api.py:2233, :2280, :2383, :1441, :6738 | nicht erwähnt | aktiv |
| Suchverlauf | Der Server merkt sich die letzten 15 bewerteten Kürzel; im Konto-Reiter „Verlauf“ sucht ein Klick erneut. | MAX | api.py:2423, plans.py:249, account.js:698, :725 | plaene „Suchverlauf — letzte 15 Suchen“ | aktiv |
| Bewertung bleibt erhalten | Die Bewertung bleibt beim Wechsel zu Tracker oder Planner und zurück stehen. | GO | :4664, :6495 | nicht erwähnt | aktiv |
| **Export & Teilen** | | | | | |
| PDF-Export | Der Knopf öffnet den Druckdialog des Browsers; die Druckansicht zeigt alle Reiter. | MAX (am Knopf gesperrt) | :3634, :527, :6167 | plaene „PDF-Export“ | aktiv (siehe Auffälligkeiten: Sperre nur am Knopf) |
| Teilen | Menü mit WhatsApp, Telegram, X, LinkedIn, Text kopieren und System-Teilen; der Text enthält Kurs, fairen Wert, Marge, Einordnung und einen App-Link. | GO | :3521, :3548, :3557 | nicht erwähnt | aktiv |
| **KI & Hilfe** | | | | | |
| KI-Assistent | Chat im Seitenpanel zur aktuellen Bewertung mit Schnellfragen, FAQ-Chips und Verlauf je Wert (bis 20 Werte / 50 Nachrichten im Browser), mit eigenem Schlüssel für Claude, ChatGPT oder Gemini. | MAX + eigener KI-Schlüssel | :5580–6100, :7717, fvp_ai_assistant.js:1 | fair-value-pro „Bring-your-own-Key: Claude, ChatGPT oder Gemini“ | aktiv |
| „✦ Erklären“ an Kennzahlen | 9 Karten (DCF, Komposit, ROIC, FCF-Rendite, Qualität, KGV, KBV, KUV, Dividendenrendite) öffnen den Chat mit einer vorbereiteten Frage. | Knopf ab PRO sichtbar, Antwort nur MAX | :6073, :2901–2910, :3119–3122 | pro-mode „öffnet ein ‚?‘-Button ein Chat-Panel“ | aktiv |
| Lesezeichen „Was bedeuten die Kennzahlen?“ | Aufklappbarer Streifen mit 5 Kurzerklärungen, startet zugeklappt. | GO | :2719, :5541 | nicht erwähnt | aktiv |
| Dokumentation + Glossar | Hilfefenster mit 12 Reitern, darunter ein Glossar mit 62 Begriffen und „Haftung“. | GO | :5559, :6822, :7250 | fair-value-pro „Glossar mit 62 Fachbegriffen“ | aktiv |
| **Bedienung & Darstellung** | | | | | |
| Seitenleiste & Handy-Ansicht | Breite der Seitenleiste per Ziehen (gemerkt); am Handy ausklappbar, Kennzahlen werden umsortiert. | GO | :6404, :6621, :6158 | nicht erwähnt | aktiv |
| Tageslimit-Hinweis | Sollte ein Tageslimit für Bewertungen anzeigen. | – | :4692 | nicht erwähnt | unsicher: immer ausgeblendet, es gibt kein Limit mehr (plans.py daily_limit = None) |

**Auffälligkeiten (Kalkulator)**
1. **PDF-Sperre nur am Knopf:** Das Druck-Stylesheet (:527–534) erzwingt die volle Ansicht. Mit dem Druckbefehl des Browsers bekämen damit wohl auch GO und PRO das volle Dashboard; die Daten lädt das Frontend ohnehin. Aus dem Code gelesen, nicht im Browser getestet.
2. **Komposit und Modellzahl:** Die Website nennt den Komposit fest mit 50/25/25; im Code lassen sich die Gewichte per Regler und Branchen-Vorlage ändern (z. B. Banken 0/100/0). Die „7 Modelle“ zählen die FCF-Rendite mit, die keinen fairen Wert liefert. Siehe Bauplan, Abschnitt „Verbindlich: Wie der Komposit Fairer Wert entsteht“.
3. **Abdeckung:** Die Website sagt „Alle börsennotierten Unternehmen weltweit … SAP, ALV“. Werte außerhalb der USA brauchen einen höheren Plan beim Datenanbieter (Hinweise :4280, :4542).
4. **„?“-Knöpfe:** pro-mode nennt einen „?“-Button an EV/EBITDA, den der Code dort nicht hat. Dafür gibt es Erklär-Knöpfe an Komposit, FCF-Rendite, KBV, KUV und Dividende, die die Website nicht nennt.
5. **Krypto und Rohstoffe:** Die Werteliste führt BTCUSD und GCUSD als „verifiziert“, die Erkennung greift aber nur bei BTC-USD bzw. „=F“-Kürzeln; diese Werte laufen vermutlich als Aktie durch. Nicht im Browser geprüft.

## Portfolio Tracker

| Funktion | Was sie tut (ein Satz, ohne Fachsprache) | Ab Plan | Fundstelle (datei:zeile) | Auf der Website | Status |
|---|---|---|---|---|---|
| **Zugang & Sperre** | | | | | |
| Seitenzugang | Ohne Anmeldung geht es zur Login-Seite. Wer angemeldet ist, aber kein MAX hat, sieht statt des Trackers eine Upgrade-Karte. Der Server liefert die Seite selbst ohne Sperre aus, die Daten sind gesperrt. | MAX | tracker.js:59, tracker.js:14411-14421, premium-gate.js:85-108, api.py:1017 | plaene: Tabelle „Portfolio Tracker – – ✓“; portfolio-tracker „Portfolio Tracker ist Teil des MAX-Plans.“ | aktiv |
| Server-Sperre | Alle Tracker-Schnittstellen lehnen Nicht-MAX ab. | MAX | tracker_api.py:803-3068 (jede Route mit @require_tier), plans.py:97/111/126, :220 | agb „MAX … sowie Portfolio Tracker und Investment Planner“ | aktiv |
| **Depots** | | | | | |
| Depot anlegen | Legt ein neues Depot mit Name und Farbe an, auch aus dem Leerzustand („Erstes Depot anlegen“). | MAX | tracker.js:5354, tracker.html:1413-1418, tracker_api.py:906 | portfolio-tracker „Beliebig viele Depots, eine Gesamtansicht.“ | aktiv |
| Depot umbenennen, Farbe ändern, löschen | Menü am Depot: Umbenennen, Farbe ändern, Depot löschen (Snapshots werden mitgelöscht). | MAX | tracker.js:4388-4393, 5372, 5389, 5401; tracker_api.py:931/961 | portfolio-tracker „mit eigener Farbe und Zusammenfassung oben“ | aktiv |
| Depot auf- und zuklappen, Rendite im Kopf | Klappt die Depot-Karte auf oder zu und zeigt im Kopf die Rendite mit und ohne Einzahlungen. | MAX | tracker.js:4310-4400, 5268 | nicht erwähnt | aktiv |
| **Snapshots** | | | | | |
| Neuer Snapshot (+ Snapshot) | Ein Datum für alle Depots auf einmal, je Depot Wert sowie optional Einzahlung und Auszahlung. Der Import-Einstieg steht direkt daneben. | MAX | tracker.js:5116-5266, tracker.html:1265, tracker_api.py:1157 | portfolio-tracker „Ein-/Auszahlungen pro Snapshot“ | aktiv |
| Positionen fortführen | Ein neuer Snapshot übernimmt die Positionen samt letztem Kurs vom Vorgänger. Ein- und Auszahlungen werden nicht übernommen. | MAX | tracker_api.py:1157-1265 | nicht erwähnt | aktiv |
| Snapshot bearbeiten | Datum (Kalenderauswahl), Wert, Verrechnungskonto, Einzahlung und Auszahlung direkt in der Tabelle ändern. Mit Cashflow-Buchungen ist die Summe gesperrt. | MAX | tracker.js:5423, 4234, 7837, 4478; tracker_api.py:1265 | nicht erwähnt | aktiv |
| Verrechnungskonto-Spalte | Trennt den Cash-Anteil vom Depotwert und warnt, wenn Cash zusätzlich als Position geführt und damit doppelt gezählt wird. | MAX | tracker.js:4565, 6655 | nicht erwähnt | aktiv |
| Rendite je Snapshot | Zeigt je Zeile die Veränderung zum Vorgänger, mit und ohne Ein- und Auszahlungen. Der erste Snapshot trägt die Marke „Start“. | MAX | tracker.js:4470, 4514-4515, 4568-4569 | nicht erwähnt | aktiv |
| Notiz, Snapshot löschen | Notiz je Snapshot (erscheint im Chart-Tooltip) und Löschen-Knopf. | MAX | tracker.js:5535, 5560, 5569, 4524-4525 | nicht erwähnt | aktiv |
| **Positionen** | | | | | |
| Position hinzufügen + Ticker-Suche | Mit Marktdaten-Schlüssel Suche nach Name oder Ticker mit Detailkarte (Sektor, Börse, ISIN, Beschreibung) und Übernahme von Name, Ticker, ISIN und Währung. Ohne Schlüssel nur Name von Hand. | MAX (Suche: + eigener FMP-Schlüssel) | tracker.js:5912, 5979, 6030 | nicht erwähnt | aktiv |
| Name, Stück, Kurs direkt bearbeiten | Felder in der Positionstabelle. Beim Kurs ist die Währung wählbar. | MAX | tracker.js:4780-4803, 6465, 6610 | nicht erwähnt | aktiv |
| Stammdaten & Ticker | Sektor, Land, Anlageklasse, ISIN (mit Prüfung) und Ticker von Hand setzen. Das geht ohne Schlüssel, die Ticker-Suche darin braucht einen. | MAX | tracker.js:6774-6890 | nicht erwähnt | aktiv |
| FMP-Kopplung „🔗“ | Ergänzt bei allen unvollständigen Positionen des Snapshots Ticker, ISIN, Sektor, Land, Währung und Anlageklasse, nach Prüfung in einer Übersicht. | MAX + eigener FMP-Schlüssel | tracker.js:4843, 6228, 6296 | nicht erwähnt | aktiv |
| Kurs aktualisieren (einzeln) | Holt den aktuellen Kurs einer Position und rechnet ihn in Euro um. Bei einem alten Snapshot fragt es, ob ein heutiger angelegt werden soll. Die FV-Einordnung wird mit aufgefrischt. | MAX + eigener FMP-Schlüssel (FV-Teil ohne) | tracker.js:6720-6740, 7539, 7576, 7606 | datenschutz „Marktdaten werden unmittelbar vom Browser … abgerufen“ | aktiv |
| „↻ Alle Kurse“ | Holt die Kurse aller Positionen des Snapshots auf einmal, jeden Ticker nur einmal. | MAX + eigener FMP-Schlüssel | tracker.js:4845, 7668 | nicht erwähnt | aktiv |
| Snapshot-Auswahl „Stand:“, Bereich einklappen | Wählt, welcher Snapshot bei den Positionen angezeigt wird, und klappt den Bereich ein. | MAX | tracker.js:4838, 5907, 5297 | nicht erwähnt | aktiv |
| Position löschen | Entfernt die Position aus diesem Snapshot. | MAX | tracker.js:6742, tracker_api.py:1711 | nicht erwähnt | aktiv |
| **Kauf-Historie / Einstand / Gewinn** | | | | | |
| Kauf-Historie | Käufe und Verkäufe mit Datum, Stück, Preis und Gebühr. Daraus ergeben sich Stückzahl, Ø-Kaufkurs und realisierter Gewinn. Warnt bei Verkauf über den Bestand. | MAX | tracker.js:6891, 7344-7525, 7112 | portfolio-tracker „Einstandswerte berechnet die App deterministisch“ | aktiv |
| Ø-Kaufkurs von Hand | Freies Feld für den Kaufkurs (mit Währung), nur solange keine Käufe eingetragen sind. | MAX | tracker.js:7394-7401 | nicht erwähnt | aktiv |
| Fußzeile Positionen | Summe Marktwert, Buchgewinn (Einstand ein-/ausblendbar), Ordergebühren und realisierter Gewinn („offen“, wenn ein Wert fehlt). | MAX | tracker.js:4888-5000, 5024 | nicht erwähnt | aktiv |
| „Im Zeitraum verkauft“ | Einklappbare Liste der verkauften Positionen mit Erlös und realisiertem Gewinn. | MAX | tracker.js:5063-5110, 5328 | nicht erwähnt | aktiv |
| **Kennzahlen & Chart** | | | | | |
| Kennzahlen-Karten | Aktueller Gesamtwert, Vermögenszuwachs (mit Einzahlungen), Rendite TWR (ohne Einzahlungen) und Rendite p.a. IRR. Hinweise auf fortgeschriebene Stände und nicht umgerechnete Fremdwährung. | MAX | tracker.js:3122-3255, 3001, 2927, 2987 | portfolio-tracker „IRR p. a. (geldgewichtet) und TWR (zeitgewichtet)“ | aktiv |
| Jahresziel | Zielrendite in % p.a. speichern. Eine Karte zeigt den Abstand der IRR zum Ziel mit Stufentext. | MAX | tracker.js:4117-4128, 4284, 3188-3210, 3323 | portfolio-tracker „persönliches Jahresziel mit Fortschritts-KPI“ | aktiv |
| „Einordnung – was deine Zahlen bedeuten“ | Aufklappbarer Text, der die eigenen TWR-, IRR- und Zuwachs-Zahlen erklärt. | MAX | tracker.html:1219, tracker.js:3257 | nicht erwähnt | aktiv |
| Jahr wählen / Jahr anlegen | Kennzahlen, Chart und Tabellen für ein Kalenderjahr. Zusätzliche Jahre lassen sich anlegen (nur im Browser gespeichert). | MAX | tracker.html:1207, tracker.js:2786, 7859 | nicht erwähnt | aktiv |
| Mehrjahresansicht | Zeiträume Letzter Monat, 3 und 6 Monate, YTD, 1, 3, 5, 10 und 15 Jahre, Alle. | MAX | tracker.html:1202, tracker.js:2066-2077, 7731-7800 | portfolio-tracker „1J · 2J · 3J · 5J · Alle“ (abweichend) | aktiv |
| Portfolio-Verlauf-Chart | Linie je Depot plus Gesamtsumme. Notizen und fortgeschriebene Stände sind markiert, die IRR-Linie läuft auf einer zweiten Achse. | MAX | tracker.html:1232, tracker.js:3818-4093 | portfolio-tracker „↗ Depotverlauf“ | aktiv |
| Benchmark-Vergleich | Vergleichswert (S&P 500 über SPY oder eigener Ticker) mit eigener Karte. Bei Fremdwährung wird mit historischen EZB-Kursen umgerechnet. | MAX + eigener FMP-Schlüssel | tracker.js:3447-3530, 3560; tracker.html:1237-1252 | nicht erwähnt | aktiv |
| Übergabe an den Zinsrechner | Gibt Depotwert und TWR an den Zinsrechner im Investment Planner weiter. | MAX | tracker.js:3139-3147 | nicht erwähnt | aktiv |
| **Reiter Vergleich** | | | | | |
| Depot-Karten | Je Depot eine Kurve und die Ø Rendite p.a. (mit Einzahlungen, TWR und IRR). Unter einem Jahr Gesamtrendite. | MAX | tracker.js:1494-1550, 1443, 2294 | portfolio-tracker „Vergleich-Tab zeigt jedes Jahr aufgeklappt“ | aktiv |
| Renditeübersicht-Tabelle | Tabelle über Depots und Jahre, Jahresblöcke aufklappbar, neueste oben. | MAX | tracker.js:2384-2770, tracker.html:1281-1286 | portfolio-tracker „Excel-artige Vergleichstabelle … aufklappbaren Jahresblöcken“ | aktiv |
| **Reiter Cashflow** | | | | | |
| Cashflow-Übersicht | Summe Ein- und Auszahlungen sowie Netto je Depot, umschaltbar „Brutto“ oder „Nur extern“ (ohne Umbuchungen). | MAX | tracker.js:1584-1735 | portfolio-tracker „Umschalter ‚alle Bewegungen‘ ↔ ‚nur externes Kapital‘“ | aktiv |
| Cashflow-Buchungen je Snapshot (💰) | Einzelbuchungen mit Datum, Richtung, Kategorie (Kapital, Ausschüttung, Kosten, Steuer), Betrag und Text, mit Merker „Umbuchung“. Nur „Kapital“ zählt als Ein- oder Auszahlung. | MAX | tracker.js:4520, 5592-5906; tracker_api.py:1332-1411 | portfolio-tracker „Kennzeichnung interner Umbuchungen zwischen eigenen Depots“ | aktiv |
| Erträge & Kosten | Summe Ausschüttungen, Kosten, Steuern und Netto-Ertrag je Depot und je Jahr, mit Ausschüttungsrendite. | MAX | tracker.js:1752, 1840-2060 | nicht erwähnt | aktiv |
| Ausschüttungen nach Monat / je Titel | Monatsübersicht der Ausschüttungen, Jahresbelege getrennt, dazu Ausschüttungen je Titel mit Angabe, wie viel davon einem Titel zugeordnet ist. | MAX | tracker.js:1950-2005 | nicht erwähnt | aktiv |
| **Reiter Allokation** | | | | | |
| Voraussetzungs-Checkliste | Zählt, wie viele Positionen Kurs, Sektor, Land und Anlageklasse haben. Ein Klick führt zu den fehlenden. | MAX | tracker.js:1167-1240 | nicht erwähnt | aktiv |
| Aufteilung in 5 Dimensionen | Nach Wert gewichtete Tabellen für Sektor, Land, Anlageklasse, Währung und FV-Einordnung, mit Hinweis, wenn die Positionen den Depotwert nicht abdecken. | MAX | tracker.js:1250-1258, 1342-1360; allocation.js | index „Renditen, Allokation, Fair Value je Position.“ | aktiv |
| Klumpenrisiko | Warnt bei Einzeltitel, Fonds, Krypto oder Sektor über der Schwelle. Unter 80 % Abdeckung gibt es keine Aussage. Die Schwellen stehen zentral im Konto unter „Einstellungen“. | MAX | tracker.js:1260-1340; account-modal.js:773-776 | nicht erwähnt | aktiv |
| Größter gemessener Rückgang | Zeigt den bisher größten Wertrückgang vom Höchststand mit Zeitraum, nur als Beobachtung. Die Grenze liegt im Planner. | MAX | tracker.js:1406-1424, 2953 | nicht erwähnt (nur Planner) | aktiv |
| Cash-Quote | Zeigt den Cash-Anteil am Depot. Den Korridor prüft der Planner. | MAX | tracker.js:1392-1438 | nicht erwähnt (nur Planner) | aktiv |
| **Universal-Import** | | | | | |
| Import aus Datei/Text | PDF (im Browser gelesen), CSV, TXT oder eingefügter Text. Die KI liest Snapshot, Positionen, Cashflows und Trades heraus, danach folgt eine Prüf-Vorschau. Auch aus dem Snapshot-Fenster erreichbar. | MAX + eigener KI-Schlüssel | tracker.html:1304-1331, tracker.js:7971, 8602, 8665, 8746, 13538, 13326 | portfolio-tracker „PDF-Finanzreport, CSV-Export oder kopierter Text“ | aktiv |
| Bild-Import (auch mehrere) | JPG, PNG oder WebP gehen verkleinert und gemeinsam an die KI. Vorher kommt der Hinweis, dass Bilder nicht geschwärzt werden. | MAX + eigener KI-Schlüssel | tracker.js:8169-8330 | portfolio-tracker „Screenshot(s) aus der Broker-App“ | aktiv |
| Mehrfach-Upload | Mehrere Dokumente einzeln auslesen und zu einer Vorschau zusammenführen. Warnung, wenn sie zu verschiedenen Depots gehören. | MAX + eigener KI-Schlüssel | tracker.js:8918, 8842, 8782 | nicht erwähnt | aktiv |
| KI-Anbieter-Wahl | Pro Import wählbar zwischen Anthropic, OpenAI und Gemini (nur Anbieter mit hinterlegtem Schlüssel). | MAX + eigener KI-Schlüssel | tracker.html:1310-1314, tracker.js:8455-8488 | portfolio-tracker „KI-Import über deinen eigenen Anbieter-Schlüssel“ | aktiv |
| Privacy-Layer + Vorschau | Schwärzt im Browser Kontoname, IBAN, Steuer-ID, Adresse, Nummern, E-Mail und Telefon. In der Vorschau kann man nachschwärzen, zurücksetzen und die Dokumentart wählen (8 Arten). | MAX + eigener KI-Schlüssel | privacy_scrubber.js:1-40, 427-440, 553-590; tracker.html:1357-1370 | portfolio-tracker „Privacy-Layer vor jedem Versand“ | aktiv |
| Eigene Schwärzungsbegriffe | Zusätzliche Begriffe, die bei jedem Import entfernt werden (nur im Browser gespeichert). | MAX | tracker.html:1398-1407, tracker.js:8084-8130, privacy_scrubber.js:246-278 | nicht erwähnt | aktiv |
| Ticker-Aufruf | Eine zweite KI-Anfrage schickt nur Bezeichnung, ISIN, WKN und Währung, um fehlende Börsenkürzel zu bestimmen. Das wird vorher offengelegt. | MAX + eigener KI-Schlüssel | tracker.js:8490, 8301-8303 | nicht erwähnt | aktiv |
| Import-Modi | Komplett überschreiben, nur Positionen ergänzen oder nur Cashflows ergänzen. Erkennt doppelte Titel und vermutete Wertpapier-Abgänge; „Summe übernehmen“ setzt den Wert. | MAX + eigener KI-Schlüssel | tracker.js:10250, 10700-10760, 10805, 10034, 11503 | nicht erwähnt | aktiv |
| Kontoauszug-Gegenprobe | Rechnet die Buchungen gegen Anfangs- und Endstand des Kontos und meldet Abweichungen. | MAX + eigener KI-Schlüssel | tracker.js:465, 563, 14222 | portfolio-tracker „rechnet die App die Buchungen gegen Anfangs- und Endstand“ | aktiv |
| Wertpapierabrechnungen / Bündel | Käufe und Verkäufe mit Gebühr und Steuer gehen als Kauf-Historie an einen bestehenden Snapshot. Gebühren werden zugeordnet, Gegenbuchungen nicht doppelt gezählt. | MAX + eigener KI-Schlüssel | tracker.js:12277, 9335, 9446, 9531, 7220 | portfolio-tracker „Ein-/Auszahlungen und Trades heraus“ | aktiv |
| Wertpapier-Eingang / Mehrmonats-Import | Eigene Vorschauen für Einlieferungen (Übertrag zwischen Depots) und für CSV-Dateien über mehrere Monate (ein Snapshot je Monat). | MAX + eigener KI-Schlüssel | tracker.js:9625, 9889 | nicht erwähnt | aktiv |
| Lernende Korrekturen | Merkt sich deine Korrekturen je Broker im Browser und gibt sie beim nächsten Import als Hinweis an die KI. | MAX + eigener KI-Schlüssel | tracker.js:14255-14335, 11787 | portfolio-tracker „Korrekturen … werden lokal gemerkt“ | aktiv |
| Bestandsabgleich („Nur abgleichen“) | Stellt den Bankauszug gegen den eigenen Snapshot (Stückzahl abweichend, nur im Auszug, nur erfasst, gleicher Name mit anderer ISIN), ohne etwas zu speichern. | MAX + eigener KI-Schlüssel | tracker.js:10540-10620, 6995 | nicht erwähnt | aktiv |
| **Jahreswerte** | | | | | |
| Steuerbescheinigung / Kosteninformation (KI) | Wird als Jahresbuchung zum 31.12. erfasst. Die Steuer wird gegen den Text gegengeprüft, eine bestehende Buchung ersetzt. | MAX + eigener KI-Schlüssel | tracker.js:9069, 9033, 9163; tracker_api.py:1435 | nicht erwähnt | aktiv |
| Jahreswerte von Hand | Steuern oder Kosten ohne KI-Schlüssel eintragen. Es entsteht dieselbe Jahresbuchung. | MAX | tracker.html:1389-1395, tracker.js:9324 | nicht erwähnt | aktiv |
| **Bewertungsbezug** | | | | | |
| FV-Einordnung je Position | Unterbewertet, Fair, Überbewertet, „Außerhalb des Modellbereichs“ oder „Nicht beurteilbar“, gelesen aus der gespeicherten Kalkulator-Analyse. | MAX (Analysen speichern = MAX) | tracker.js:4670, 1070-1080; model_boundary.js | portfolio-tracker „unterbewertet, fair bewertet oder überbewertet“ | aktiv |
| Automatischer FV-Abgleich | Beim Öffnen und bei Rückkehr in den Browser-Tab werden fehlende oder neuere Einordnungen nachgeladen. | MAX | tracker.js:14340-14408; tracker_api.py:1970, 2009 | portfolio-tracker „aktualisiert sich bei jedem Tab-Wechsel“ | aktiv |
| Quartals-Check-Marke „fällig“ | Markiert Positionen, deren Analyse älter als die eingestellte Frist ist, und verlinkt auf „Meine Werte“. | MAX | tracker.js:1480, 4679; tracker_api.py:2233 | nicht erwähnt | aktiv |
| **Währung** | | | | | |
| Mehrwährung / Umrechnung | Fremdwährungskurse werden über den Tageskurs des Servers umgerechnet gespeichert, ohne Marktdaten-Schlüssel als Kurs von Hand. Die Herkunft steht im Tooltip, nicht umgerechnete Positionen werden gewarnt. | MAX | tracker.js:169-262, 6496-6610, 4721-4740 | nicht erwähnt | aktiv |
| Basiswährung | Einstellung default_currency (Standard EUR) am Server, in der Oberfläche nicht änderbar. | MAX | tracker_api.py:757/782 | nicht erwähnt | unsicher: kein Bedienelement zum Ändern gefunden |
| **Export & Schnittstellen** | | | | | |
| Excel-Export | .xlsx mit Übersicht, Snapshots, Positionen (mit FV-Einordnung bzw. „nicht beurteilbar“) und einem Blatt je Depot. | MAX | tracker.js:7802; tracker_api.py:3067, 2682-2835 | portfolio-tracker „Übersichts-Sheet plus ein eigenes Blatt pro Depot“ | aktiv |
| Daten für den Planner | Depotwerte und aktuelle Positionen für Rebalancing, Richtwerte und Stresstest (nur lesen). | MAX | tracker_api.py:2419, 2472; trading-desk.js:143 | investment-planner „Rebalancing nutzt die Depot-Daten aus dem Portfolio Tracker“ | aktiv |

**Auffälligkeiten (Tracker)**
1. Die Allokation hat 5 Dimensionen statt 4: Sektor, Land, Anlageklasse, Währung und die FV-Einordnung (tracker.js:1258). Stresstest, Drawdown-Grenze und Cash-Korridor liegen im Planner. Im Tracker bleiben nur „Größter gemessener Rückgang“ und die Cash-Quote als Beobachtung; der Stresstest ist dort nur noch ein Rest-Kommentar (tracker.js:1011).
2. Die Website nennt die Zeiträume „1J · 2J · 3J · 5J · Alle“. Der Code hat kein „2J“, dafür 1M, 3M, 6M, YTD, 10J und 15J. Beim Excel-Export sagt die Website „Übersicht + Blatt je Depot“; tatsächlich kommen die Blätter „Snapshots“ und „Positionen“ dazu.
3. Kursabruf, „Alle Kurse“, FMP-Kopplung, Ticker-Suche und Benchmark brauchen einen eigenen FMP-Schlüssel. Ohne Schlüssel kommt ein Dialog „Zugang einrichten →“. Den Benchmark-Vergleich erwähnt die Website gar nicht.
4. Auf der Website fehlen u. a.: Kauf-Historie mit realisiertem Gewinn, Ausschüttungen nach Monat bzw. je Titel, Jahreswerte (Steuerbescheinigung/Kosteninformation, auch von Hand), Bestandsabgleich, ISIN-Abgleich, Mehrwährung, Quartals-Check-Marke und die Einordnungen „Außerhalb des Modellbereichs“ und „Nicht beurteilbar“.
5. Toter Code: Client-Methode getFv (tracker.js:121) wird nirgends aufgerufen.

## Investment Planner

| Funktion | Was sie tut (ein Satz, ohne Fachsprache) | Ab Plan | Fundstelle (datei:zeile) | Auf der Website | Status |
|---|---|---|---|---|---|
| **Rahmen (gilt für alle Module)** | | | | | |
| Seitenzugang mit Sperre | Nur angemeldete Nutzer kommen auf die Seite. Wer nicht MAX hat, sieht statt der Module eine Upgrade-Karte. | MAX (+ Login) | frontend/trading-desk.js:37, :3244-3256; frontend/premium-gate.js:76-90 | plaene: Tabelle „Investment Planner – – ✓“; investment-planner „Investment Planner ist Teil des MAX-Plans“ | aktiv |
| Server-Sperre | Alle Planner-Schnittstellen lehnen Nutzer ohne MAX ab (gilt auch für Tracker-Daten). | MAX | td_api.py:122 ff. (`@require_tier("td_modules")`, jeder Endpunkt); plans.py:98/112/127, :220 | wie oben | aktiv |
| Modul-Leiste und Direktlinks | Seitenleiste mit 10 Modulen, Direktlinks per `#modul`, das zuletzt geöffnete Modul wird gemerkt. | MAX | frontend/trading-desk.js:3108-3125, :3159-3181 | investment-planner „Zehn Module ansehen“ | aktiv |
| Einstellungen wirken sofort | Änderungen an Risiko je Position oder Mindest-CRV in den Einstellungen zeichnen das offene Modul ohne Neuladen neu. | MAX | frontend/trading-desk.js:3262; frontend/account-modal.js:761-772, :885 | nicht erwähnt | aktiv |
| **Meine Richtwerte** | | | | | |
| Rückgang-Grenze | Zeigt den selbst festgelegten maximalen Rückgang in % und dazu den aktuellen Euro-Betrag. | MAX + Wert in den Einstellungen + Tracker-Depotwert | frontend/trading-desk.js:136-185 | investment-planner „maximaler Rückgang in Euro, Cash-Korridor“ | aktiv |
| Cash-Quote gegen Korridor | Zeigt die aktuelle Cash-Quote aus dem jüngsten Snapshot und ob sie unter, im oder über deinem Korridor liegt. | MAX + Tracker-Snapshots/Positionen + Korridor in den Einstellungen | frontend/trading-desk.js:187-209; frontend/allocation.js (cashQuote) | investment-planner „Cash-Korridor“ | aktiv |
| Verweis auf Einstellungen | Geändert wird nur zentral in den Einstellungen; ein Link öffnet sie direkt. | MAX | frontend/trading-desk.js:165; frontend/account-modal.js:398 | nicht erwähnt | aktiv |
| **Stresstest** | | | | | |
| Vier feste Szenarien | Rechnet in Euro durch: größter Sektor −30 %, größte Einzelaktie −100 %, alles −20 % außer Cash, Krypto −50 %. | MAX + Tracker-Positionen | frontend/trading-desk.js:315-347 | investment-planner „Größter Sektor minus 30 %, größte Position“ | aktiv |
| Eigenes Szenario | Frei wählbar nach Sektor, Land, Anlageklasse oder Einzelposition, mit eigenem Rückgang in %. | MAX + Tracker-Positionen | frontend/trading-desk.js:349-432 | nicht erwähnt | aktiv |
| Abgleich mit der eigenen Grenze | Jedes Szenario sagt, ob der Verlust innerhalb deiner Rückgang-Grenze liegt oder um wie viel er sie überschreitet. | MAX + Grenze in den Einstellungen | frontend/trading-desk.js:288-294 | investment-planner „in Euro, gegen deine Grenze gestellt“ | aktiv |
| Hinweis auf Abdeckung | Weist darauf hin, wenn die erfassten Positionen nur einen Teil des Depotwerts abdecken. | MAX | frontend/trading-desk.js:378-382 | nicht erwähnt | aktiv |
| **Watchlist** | | | | | |
| Eintrag hinzufügen mit Suche | Sucht live nach Aktie, ETF oder Krypto und fügt den Treffer hinzu. Doppelte Einträge werden abgelehnt, auch bei anderer Ticker-Schreibweise derselben ISIN. | MAX + Marktdaten-Schlüssel (Suche läuft über FMP) | frontend/trading-desk.js:1385-1515; td_api.py:135-168; api.py:2237 | nicht erwähnt (nur Mockup „+ Eintrag“) | aktiv |
| Fairer Wert aus gespeicherter Analyse | Holt den fairen Wert aus der jüngsten gespeicherten Bewertung. Ohne Bewertung gibt es den Link „Im Kalkulator durchrechnen“. | MAX + gespeicherte Analyse | td_api.py:326-381, :407-480; frontend/trading-desk.js:1301-1315 | investment-planner „greifen auf deine gespeicherten FVP-Bewertungen zu“ | aktiv |
| Kaufzonen-Alarm (Glocke) | Markiert „🔔 Kaufzone erreicht“ und zählt im Kopf „N in Kaufzone“. Es gibt keine Push- oder Mail-Benachrichtigung. | MAX + gespeicherte Analyse | frontend/trading-desk.js:1261-1270, :1319-1323; td_api.py:552-556 | investment-planner „ist er beim nächsten Blick markiert“; index „Watchlist 1 IN KAUFZONE“ | aktiv |
| Alarmschwelle | Schwelle ist die gespeicherte Sicherheitsmarge, sonst der Wert je Qualitätsnote aus den Einstellungen, sonst die Watchlist-Schwelle (änderbar 0–80 %). | MAX | td_api.py:220-238, :552-556; frontend/trading-desk.js:1529-1546 | investment-planner „sobald er die Sicherheitsmarge erreicht“ | unsicher: die angezeigte und änderbare Schwelle wirkt nur, wenn die Analyse keine Sicherheitsmarge und keine Note trägt |
| Modellgrenze | Liegt der Kurs außerhalb des Bereichs, in dem das Bewertungsmodell aussagekräftig ist, schweigt der Alarm und der Grund wird angezeigt. | MAX | frontend/trading-desk.js:1261-1265, :1328-1335 | nicht erwähnt | aktiv |
| Live-Kurs mit Rückfall | Versucht einen Live-Kurs. Klappt das nicht, gilt der Kurs der gespeicherten Analyse mit Datum. Anzeige „Live“ oder Datum. | MAX | td_api.py:384-404, :537-546; frontend/trading-desk.js:1324-1326 | investment-planner Mockup „293 € LIVE“ | unsicher: Live-Kurs nur, wenn YAHOO_DEV_MODE gesetzt ist (td_api.py:393). Auf dem Live-Server kommt immer der Rückfall |
| Hinweis „veraltet / unvollständig“ | Kennzeichnet Einträge, deren Analyse älter als deine Überholt-Frist ist oder Bausteine fehlen. | MAX (Frist in den Einstellungen) | frontend/trading-desk.js:1365-1376; td_api.py:241-264, :520-526 | nicht erwähnt | aktiv |
| Aktualisieren, Löschen, Auto-Refresh | Neu prüfen per Knopf und automatisch beim Zurückwechseln in den Tab. Löschen mit Rückfrage. | MAX | frontend/trading-desk.js:1381, :1517-1528, :1551-1560 | nicht erwähnt | aktiv |
| **Positionsgröße** | | | | | |
| Stückzahl aus Risiko | Stückzahl = (Depotwert × Risiko je Position) ÷ (Einstieg − Falsifikationsniveau), abgerundet. Dazu Positionswert, Depotanteil und Risikobudget. | MAX | frontend/trading-desk.js:2011-2034, :2324-2390 | investment-planner „Wie viele Stück, bei welchem Risiko?“ | aktiv |
| Chance-Risiko-Verhältnis (CRV) | Rechnet Ziel (fairer Wert) gegen Risiko, vergleicht mit deiner Mindestschwelle und blendet die Rechnung ab, wenn das Ziel unter dem Einstieg liegt. | MAX + gespeicherte Analyse (für das Ziel) | frontend/trading-desk.js:2023-2031, :2361-2371, :2393-2416 | investment-planner „Mit Prüfung des Chance-Risiko-Verhältnisses“ | aktiv |
| Klumpen-Hinweis | Ab 15 % Depotanteil ein neutraler Hinweis, ab 20 % „Das Klumpenrisiko steigt“. | MAX | frontend/trading-desk.js:2351-2358 | nicht erwähnt | aktiv |
| ATR-Vorschlag | Lädt die typische Tagesschwankung (ATR 14) und schlägt als Falsifikationsniveau „Einstieg − 2 × ATR“ vor, umgerechnet in die Anzeigewährung. | MAX + Marktdaten-Schlüssel (im FMP-Free-Plan meist nur US-Werte) | frontend/trading-desk.js:2169-2220; api.py:3235 ff. | nicht erwähnt | aktiv |
| Position aus der Watchlist | Auswahl nur aus Watchlist-Werten mit gespeicherter Analyse, mit Analyse-Datum und Warnhinweisen. | MAX + Watchlist + gespeicherte Analyse | frontend/trading-desk.js:2063-2111, :2222-2237 | nicht erwähnt | aktiv |
| Werte merken | Depotwert, Risiko und CRV werden dauerhaft am Server gespeichert. Einstieg und Falsifikation je Ticker sowie die letzte Position nur im Browser. | MAX | frontend/trading-desk.js:2138-2166, :2246-2260, :2437-2448; td_api.py:986-1063 | nicht erwähnt | aktiv |
| **Einstiegsplan** | | | | | |
| Tranchen-Plan | 1–4 Tranchen mit Auslöser, Anteil in %, Kurs, Stück und Betrag. Die Gesamtstückzahl kommt aus der Positionsgröße. | MAX + Depotwert aus der Positionsgröße + Watchlist/Analyse | frontend/trading-desk.js:2458-2492, :2626-2690 | investment-planner „in wie vielen Tranchen, zu welchen Kursen“ | aktiv |
| Kursvorschläge je Tranche | Mit ATR im Abstand von ½ ATR, sonst gleichmäßig zwischen Einstieg und Falsifikation. | MAX (ATR: + Marktdaten-Schlüssel) | frontend/trading-desk.js:2468-2481, :2582-2604 | nicht erwähnt | aktiv |
| Zwei Pflichtfragen | „Woran erkenne ich, dass ich falsch liege?“ und „Was müsste eintreten…?“ müssen ausgefüllt sein, sonst ist Speichern gesperrt. | MAX | frontend/trading-desk.js:2561-2565, :2692-2714; td_api.py:1160-1166 | investment-planner „was eintreten müsste, damit die These aufgeht“ | aktiv |
| Gespeicherte Pläne | Ein Plan je Wert, den man öffnen, ersetzen (mit Rückfrage) oder löschen kann. Markiert, wenn Plan oder Analyse veraltet bzw. unvollständig sind. | MAX | frontend/trading-desk.js:2764-2880; td_api.py:1118-1225 | nicht erwähnt | aktiv |
| **Verkaufsprüfung** | | | | | |
| Damals gegen heute | Stellt Einstieg, fairen Wert und Falsifikation aus dem Plan dem heutigen Kurs und der jüngsten Analyse gegenüber, samt den damals notierten Antworten. | MAX + gespeicherter Einstiegsplan | frontend/trading-desk.js:2921-3021; td_api.py:1361-1418 | investment-planner „Deine damalige These gegen den heutigen Kurs“ | aktiv |
| Heutiger Kurs | Holt den aktuellen Kurs. Ohne ihn ist die Prüfung trotzdem möglich, dann nur anhand der Freitexte. | MAX + Marktdaten-Schlüssel (optional) | frontend/trading-desk.js:2973-2981, :3017-3018 | nicht erwähnt | aktiv |
| Drei Fragen mit Einordnung | Ja/Nein/unklar zu Falsifikation, These und Änderung am Unternehmen. Ergebnis ist eine neutrale Einordnung (Annahme hinfällig / fairer Wert erreicht / nur Kursrückgang / unklar), keine Empfehlung. | MAX | frontend/trading-desk.js:2899-2919, :3022-3058; td_api.py:1240-1305 | investment-planner „Ein Kursrückgang allein ist kein Grund.“ | aktiv |
| Prüf-Verlauf | Jede Prüfung wird dauerhaft gespeichert (nicht löschbar) und als Verlauf angezeigt. | MAX | frontend/trading-desk.js:3060-3097; td_api.py:1421 ff. | nicht erwähnt | aktiv |
| **Recovery Return** | | | | | |
| Rangliste Aufholpotenzial | Listet alle gespeicherten Bewertungen (max. 50) mit Abstand zum fairen Wert, nötiger Erholung in % und nötiger Jahresrendite, sortiert nach Potenzial. | MAX + gespeicherte Analysen | frontend/trading-desk.js:504-640, :705-725; td_api.py:874-971 | investment-planner „rankt deine gespeicherten Bewertungen nach Aufholpotenzial“ | aktiv |
| Zeithorizont-Regler | 1–120 Monate. Unter 1 Jahr erscheint ein Hinweis zur Aussagekraft. | MAX | frontend/trading-desk.js:686-703 | partner „p.a.-Wachstum über variablen Zeithorizont“ | aktiv |
| Manuelle Einträge | Eigener Ticker mit Kurs und Zielkurs in 4 Währungen, nur im Browser gespeichert. Läuft ohne Server, liegt aber hinter der MAX-Seitensperre. | MAX (Seitensperre) | frontend/trading-desk.js:488-502, :644-680 | nicht erwähnt | aktiv |
| **Zinsrechner** | | | | | |
| Endkapital mit Zinseszins | Startkapital und Sparrate (monatlich/jährlich), Zins, Laufzeit bis 80 J. Zeigt Endkapital, eingezahlt, Zinsgewinn und effektive Jahresrendite, live bei jeder Eingabe. Läuft ohne Server, liegt aber hinter der MAX-Seitensperre. | MAX (Seitensperre) | frontend/trading-desk.js:734-778, :780-946 | investment-planner „live aktualisiert ohne Berechnen-Button“ | aktiv |
| Inflation / Kaufkraft | Rechnet das Endkapital in heutige Kaufkraft um (Standard 2,5 %). | MAX | frontend/trading-desk.js:769-776, :920-946 | investment-planner „Realwert in heutiger Kaufkraft“ | aktiv |
| Verlaufsgrafik und Jahrestabelle | Linien für nominal, real und eingezahlt plus aufklappbare Tabelle pro Jahr. | MAX | frontend/trading-desk.js:948-1040 | investment-planner „mit Verlauf und Jahrestabelle“ | aktiv |
| Übernahme aus dem Tracker | Ein Klick übernimmt den Depotwert und die Jahresrendite (nur bei mindestens 1 Jahr Zeitraum) aus dem Tracker. | MAX + vorher den Tracker geöffnet (Übergabe im Browser) | frontend/trading-desk.js:881-913 | nicht erwähnt | aktiv |
| **Rebalancing** | | | | | |
| Ziel-Gewichte je Depot | Ziel-% pro Tracker-Depot eintragen. Speichern geht nur, wenn die Summe 100 % ergibt. | MAX + Depots im Tracker | frontend/trading-desk.js:1049-1151; td_api.py:728-777 | investment-planner „verknüpft direkt mit deinen Tracker-Depots“ | aktiv |
| Soll-Ist-Vergleich mit Beträgen | Zeigt je Depot Ist gegen Ziel und „Stocke um X € auf / Kürze um X €“, dazu eine Liste konkreter Verschiebungen (ab 50 €). | MAX + Tracker-Snapshots | frontend/trading-desk.js:1154-1240; td_api.py:780-871 | investment-planner „was aufstocken, was kürzen“ | aktiv |
| **Trade Journal** | | | | | |
| Eintrag mit unveränderlicher Begründung | Typ (Kauf/Nachkauf/Reduzierung/Verkauf/Halten), Ticker, Stück, Datum, Begründung ab 30 Zeichen. Die Begründung ist nach dem Speichern nicht mehr änderbar (der Server lässt es nicht zu). | MAX | frontend/trading-desk.js:1791-1946; td_api.py:540-555, :633-698 | investment-planner „nach dem Speichern unveränderlich“ | aktiv |
| KI-Bias-Coach (16 Denkfehler) | Prüft die Begründung vor dem Speichern auf 16 Denkfehler und zeigt Zitat, Erklärung und Gegenstrategie. Das Ergebnis wird mit dem Eintrag gespeichert. | MAX + eigener KI-Schlüssel (Claude/OpenAI/Gemini) | frontend/trading-desk.js:1839-1924; frontend/td_bias_catalog.js:14-185 | investment-planner „prüft die KI sie auf 16 Denkfehler“; akademie_modul-07-psychologie „Der KI-Bias-Coach (Investment Planner)“ | aktiv |
| Nachträgliche KI-Prüfung | Prüft gespeicherte Einträge erneut. Das Ergebnis wird nur angezeigt, nicht gespeichert. | MAX + eigener KI-Schlüssel | frontend/trading-desk.js:1720-1789 | nicht erwähnt | aktiv |
| Rückblick-Urteil (Verdient/Glück/Pech/Strafe) | Zwei Ja/Nein-Fragen (Prozess gut? Ergebnis gut?) ergeben das Urteil. Es bleibt änderbar, mit Zählung über alle Einträge. | MAX | frontend/trading-desk.js:1574-1580, :1600-1603, :1669-1690, :1963-1969; td_api.py:557-573, :615-630 | nicht erwähnt (nur Akademie-Lehrstoff, modul-07 „verdient / Glück / Pech / Strafe“) | aktiv |
| Ergebnis-Notiz | „Was ist daraus geworden?“ mit Datum, nachträglich eintragbar und änderbar. | MAX | frontend/trading-desk.js:1620-1631, :1971-1988 | nicht erwähnt | aktiv |
| Löschen | Eintrag nach Rückfrage endgültig löschen. | MAX | frontend/trading-desk.js:1948-1960; td_api.py:701-713 | nicht erwähnt | aktiv |

**Auffälligkeiten (Planner)**
- **Live-Kurse:** Watchlist und Recovery Return nutzen Live-Kurse nur im Dev-Modus (td_api.py:384-404). Auf dem Live-Server erscheint nie das „Live“-Abzeichen. Die Website zeigt im Mockup trotzdem „LIVE“.
- **Watchlist-Suche:** Ohne Marktdaten-Schlüssel lässt sich kein Watchlist-Eintrag anlegen, denn die Suche braucht FMP (frontend/trading-desk.js:1471-1475). Die Website erwähnt diese Voraussetzung nicht.
- **Veralteter Upgrade-Text:** Der Text auf der Upgrade-Karte nennt nur 4 Module (frontend/trading-desk.js:3252).
- **Richtwerte/Stresstest aktualisieren nicht sofort:** Die Seite reagiert nur auf Änderungen an Risiko je Position und Mindest-CRV (frontend/trading-desk.js:3265). Rückgang-Grenze oder Cash-Korridor zeigen bis zum Modulwechsel oder Neuladen noch die alten Werte.
- **Zinsrechner und manuelles Recovery Return:** Beide laufen komplett im Browser, sind wegen der Seitensperre aber nur für MAX erreichbar.

## Konto, Schlüssel, Kauf und Übergreifendes

| Funktion | Was sie tut (ein Satz, ohne Fachsprache) | Ab Plan | Fundstelle (datei:zeile) | Auf der Website | Status |
|---|---|---|---|---|---|
| **Konto** | | | | | |
| Registrierung | Legt mit Name, E-Mail und Passwort (mindestens 8 Zeichen) ein kostenloses Konto an. Gesperrte Adressen werden abgewiesen. | GO → wird PRO | api.py:254, auth.py:488, fairvaluepro.html:1537 | pro-mode „Registriere dich mit deiner E-Mail — kein Abo“ | aktiv |
| Anmeldung | Meldet mit E-Mail und Passwort an (/login und /register führen zum Anmeldefenster im Kalkulator), danach geht es zum Wunschziel oder zum Onboarding. | PRO (Konto) | api.py:288, api.py:971-977, fairvaluepro.html:1475, auth-ui.js:110 | nicht erwähnt | aktiv |
| Login-Sperre | Nach 10 Fehlversuchen in 10 Minuten je Internetadresse wird gesperrt, in den letzten 3 Versuchen mit Restzahl. | alle | auth.py:544, api.py:290-304 | nicht erwähnt | aktiv |
| E-Mail-Bestätigung | Ein Link (24 Stunden gültig) bestätigt die Adresse und meldet direkt an. Das Konto ist auch unbestätigt nutzbar, nur der Kauf braucht die Bestätigung. | PRO (Konto) | api.py:324, auth.py:299, auth.js:117-148 | nicht erwähnt | aktiv |
| Bestätigungsmail erneut senden | Höchstens alle 2 Minuten, nur aus dem Kaufdialog erreichbar. | PRO (Konto) | api.py:459, upgrade-modal.js:142 | nicht erwähnt | aktiv |
| Passwort vergessen / zurücksetzen | Schickt einen Link (1 Stunde gültig), ohne zu verraten, ob die Adresse existiert. Damit setzt man ein neues Passwort. | PRO (Konto) | api.py:503-566, auth.py:365-408, fairvaluepro.html:1605-1660 | nicht erwähnt | aktiv |
| Passwort ändern | Ändert das Passwort im Konto-Fenster und meldet alle anderen Geräte ab. | PRO (Konto) | api.py:447, account-modal.js:453 | nicht erwähnt | aktiv |
| Abmelden (alle Geräte) | Beendet die Sitzung auf allen Geräten; ist der Server nicht erreichbar, nur auf diesem Gerät, mit Hinweis. | PRO (Konto) | api.py:478, auth.js:321, header.js:347 | nicht erwähnt | aktiv |
| Sitzungsdauer | Eine Anmeldung gilt 7 Tage, früher endet sie durch Abmelden oder Passwortwechsel. | PRO (Konto) | auth.py:28, auth.py:250-287 | nicht erwähnt | aktiv |
| Profil | Name sowie X/Twitter, LinkedIn und Website bearbeiten. Die E-Mail ist nicht änderbar. | PRO (Konto) | api.py:440, account-modal.js:189-201, :428 | nicht erwähnt | aktiv |
| Konto löschen | Löscht nach Passworteingabe alle Kontodaten unwiderruflich und beendet ein laufendes Abo sofort. | PRO (Konto) | api.py:488, auth.py:778-797, account-modal.js:492 | plaene „Presets und Daten werden unwiderruflich gelöscht“ | aktiv |
| Konto-Fenster „Mein Account“ | Ein Fenster auf allen Seiten mit den Reitern Profil, Plan, API-Schlüssel, Einstellungen und ISIN-Abgleich. Gäste sehen nur API-Schlüssel. Direkt aufrufbar über ?openAccount=. | PRO (Konto); Gast nur Schlüssel | account-modal.js:170-187, :363, fairvaluepro.html:1451/6693 | nicht erwähnt | aktiv |
| Einstellungen (zentrales Regelwerk) | 14 persönliche Schwellen an einer Stelle, je mit Begründung und „Standard zurücksetzen“: Sicherheitsmarge je Note, Mindest-CRV, Risiko je Position, Klumpen, Rückgang, Cash-Spanne, „Analyse veraltet nach“. Gilt für alle drei Werkzeuge. | MAX (Server prüft) | account-modal.js:759-958, tracker_api.py:1017-1024, td_api.py:986-1006 | nicht erwähnt | aktiv |
| ISIN-Abgleich | Zeigt nur an, wo ISIN und Ticker von Positionen und Analysen auseinanderlaufen. Ändert nichts. | MAX | account-modal.js:300, :1051, tracker_api.py:3053 | nicht erwähnt | aktiv |
| Fenster „Meine Analysen & Vorlagen“ | Nur im Kalkulator: gespeicherte Analysen mit „veraltet“-Marke, Vorlagen und die letzten 15 Suchen. Ohne MAX steht dort eine Sperrkachel mit Upgrade-Knopf. | MAX (Listen) | fairvaluepro.html:1899-1960, account.js:10-55, :264, :586, :698, :761 | plaene „Analysen speichern — unbegrenzt“, „Suchverlauf — letzte 15 Suchen“ | aktiv |
| Quartals-Check | Klappt in „Meine Werte“ je Titel auf: damals gegen heute, plus Abgleich mit dem Grund, der die Kaufthese widerlegen würde. | MAX | account.js:125-262, tracker_api.py:2233 | nicht erwähnt | aktiv |
| **Schlüssel** | | | | | |
| Marktdaten-Schlüssel (FMP) eintragen, testen, entfernen | Den eigenen Anbieterschlüssel hinterlegen und mit einem Testabruf prüfen. Ein weiterer Anbieter ist als „Bald“ geführt. | alle (auch Gast) | account-modal.js:226-258, :646-749, data_client.js:28-94, data_provider_registry.js:94-119 | daten-quelle „Im Onboarding wählst du deinen Daten-Partner“ | aktiv |
| Schlüssel nur im Browser | Speichert den Schlüssel nur im Browser. Kalkulator und Planner reichen ihn je Abfrage im Header an den Server, der Tracker fragt direkt beim Anbieter. | alle | data_client.js:18-48, api.py:787-830, api.py:139-178 | daten-quelle „ausschließlich in deinem Browser ( localStorage )“ | aktiv |
| KI-Schlüssel (Claude, ChatGPT, Gemini) mit Test | Schlüssel je Anbieter hinterlegen und testen, Anfragen gehen direkt vom Browser zum Anbieter. Hinterlegen geht immer, genutzt wird er in MAX-Funktionen. | Hinterlegen alle; Nutzung MAX | account-modal.js:260-292, :526-642, ai_client.js:57-59, :267, :300 | fair-value-pro „Claude (Anthropic), ChatGPT (OpenAI) oder Gemini (Google)“ | aktiv |
| Schlüssel-Status an den Server | Meldet nur „Schlüssel ja/nein“ plus Anbietername, nie den Schlüssel, bei Login, Speichern und im Onboarding. | PRO (Konto) | api.py:412, auth.js:273, account-modal.js:629/738 | nicht erwähnt | aktiv |
| Link „Höheren Plan beim Anbieter ansehen“ | Erscheint bei Meldungen, dass der Anbieter-Plan nicht reicht, und führt über den Partnerlink zum Anbieter. | alle | data_provider_registry.js:375-408 (genutzt u. a. fairvaluepro.html, tracker.js, trading-desk.js) | daten-quelle „in zweierlei Hinsicht begrenzt“ | aktiv |
| **Onboarding** | | | | | |
| Onboarding-Assistent (3 Schritte) | Datenanbieter (Pflicht, mit Test), KI (optional), „Geschafft“. Pflicht nach der ersten Anmeldung; Gäste können ihn ohne Abschlussmarke durchlaufen. | alle; Pflicht für PRO | onboarding.html:166-534, api.py:390, :979, fairvaluepro.html:1507 | daten-quelle „Das Einrichten dauert ca. fünf Minuten“ | aktiv |
| FMP-Partnerlink (25 % Rabatt) | Knopf „Konto anlegen — 25 % Rabatt“ öffnet die Anbieterseite mit Gutschein im Link. | alle | data_provider_registry.js:108, onboarding.html:194, :376 | daten-quelle „25 % Rabatt … über unseren Partner-Link“ | aktiv |
| Tester-Umgehung im Onboarding | Schritt 1 ohne Anbieter überspringen, nur solange kein Anbieter verfügbar ist. | alle | onboarding.html:488-496 | nicht erwähnt | unsicher: FMP ist verfügbar, der Link erscheint damit nie |
| Gastzugang „Ohne Account testen“ | Startet den Kalkulator ohne Konto in der Kurzansicht, nur für die Sitzung. | GO | fairvaluepro.html:1589, :1808, account.js:735 | plaene „Ohne Account starten“ | aktiv |
| **Kauf & Abo** | | | | | |
| Kaufdialog „Auf MAX wechseln“ | Zeigt MAX-Vorteile, Preise (9,99 €/Monat oder 89,91 €/Jahr) und den AGB-Hinweis. Ist der Verkauf geschlossen, gibt es nur „MAX anfragen“ per Mail. | PRO (Konto) | upgrade-modal.js:52-138, :217, plans.py:20-21, :138 | plaene „MAX wählen →“ | aktiv |
| Verkaufsschalter | Kaufknöpfe nur, wenn Stripe aktiv und FVP_CHECKOUT_OPEN=1 gesetzt ist. | – | plans.py:303, api.py:1113, upgrade-modal.js:25-31 | nicht erwähnt | aktiv (live belegt: /config meldet checkout_open true) |
| E-Mail-Bestätigung als Kaufvoraussetzung | Ohne bestätigte Adresse erscheint statt der Kaufknöpfe „Bestätigungsmail senden“; der Server lehnt den Kauf ebenfalls ab. | PRO (Konto) | api.py:575, upgrade-modal.js:66-70, :121-124 | nicht erwähnt | aktiv |
| Zustimmung zum sofortigen Leistungsbeginn | Pflichthaken mit Textversion, mit Volltext gespeichert und der Zahlung zugeordnet. | PRO (Konto) | plans.py:66-76, :358, :371, api.py:577-581, upgrade-modal.js:74-77, :175 | agb „Beide Zustimmungen werden im Bestellprozess separat eingeholt“ (abweichend, siehe Auffälligkeiten) | aktiv |
| Stripe-Bezahlseite (monatlich/jährlich) | Leitet zur Bezahlseite von Stripe und danach auf die Ausgangsseite zurück. Nur die zwei hinterlegten Preise sind erlaubt. | PRO (Konto) | api.py:569, plans.py:56, :329, upgrade-modal.js:184 | plaene „Buchung läuft direkt in der App“ | aktiv |
| Rückkehr nach dem Kauf | Meldet „aktiv“ erst, wenn die Datenbank die neue Stufe bestätigt (bis zu 6 Nachfragen); bei Abbruch eine Meldung. | PRO (Konto) | auth.js:154-161, upgrade-modal.js:303-354 | nicht erwähnt | aktiv |
| Kauf-Einstieg von der Website | Merkt sich die Kaufabsicht 24 Stunden über Registrierung, Bestätigung und Onboarding hinweg und öffnet danach den Kaufdialog. | GO → PRO | auth.js:64-96, :172-187, upgrade-modal.js:333 | plaene „MAX wählen →“ | aktiv |
| Reiter „Plan“ und Kundenportal | Zeigt den aktuellen Plan. MAX führt zum Stripe-Kundenportal (verwalten, Rechnungen, kündigen) und zur Kündigungsseite, andere Pläne bekommen einen Upgrade-Knopf. | PRO / MAX | upgrade-modal.js:233-294, api.py:600, plans.py:381 | agb „oder über die Account-Verwaltung möglich“ | aktiv |
| Abo-Folgen (Webhook) | Nach bezahltem Kauf wird MAX gesetzt; endet das Abo (unbezahlt nach Mahnlauf, gekündigt), geht es auf PRO zurück. Während des Mahnlaufs bleibt MAX. Von Hand vergebene Stufen bleiben unberührt. | MAX | api.py:615, plans.py:424-533 | agb „bleibt der MAX-Zugang … während des … Mahnlaufs erhalten“ | aktiv |
| Kündigungsschaltfläche (/kuendigen) | Formular ohne Anmeldung: ordentlich oder außerordentlich, nächstmöglich oder zum Wunschdatum. Kündigt bei Stripe, vergibt eine Vorgangsnummer und schickt Bestätigungen. | alle (wirkt für MAX) | kuendigen.html:48-141, api.py:1022-1047, kuendigung.py:92, :157, :251 | plaene „Verträge hier kündigen“ | aktiv |
| Sperre für Tracker und Planner | Ohne MAX erscheint statt des Werkzeugs eine Upgrade-Karte, ohne Anmeldung geht es zum Login mit Rücksprung. | MAX | premium-gate.js:76-108, tracker.js:59, trading-desk.js:37 | plaene MAX-Liste „Portfolio Tracker / Investment Planner“ | aktiv |
| **Recht** | | | | | |
| Fußzeile mit Rechtslinks | Impressum, Datenschutz und „Verträge hier kündigen“ auf jeder App-Seite. Ein AGB-Link fehlt (siehe Bauplan, offene Rechtsfrage). | alle | header.js:286-298, home.html:174-186, fairvaluepro.html:1889-1894, onboarding.html:538 | index „Verträge hier kündigen“ | aktiv |
| Impressum / Datenschutz der App | Eigene Rechtsseiten der App; die Datenschutzerklärung deckt Schlüssel, Telemetrie und Stripe ab. | alle | api.py:1050-1057, impressum.html, datenschutz.html | datenschutz „Für die App … gilt eine eigene“ | aktiv |
| AGB-Link | Verweis im Kaufdialog auf gobetter.trade/agb.html; die AGB liegen nicht in der App. | PRO | upgrade-modal.js:83-97 | agb (eigene Seite) | aktiv |
| **Oberfläche** | | | | | |
| Startseite | Persönliche Begrüßung mit drei Werkzeugkarten samt Planabzeichen; Fußzeile mit Anmelden/Abmelden und Dokumentation. | alle | home.html:150-186, :205-255, api.py:941 | nicht erwähnt | aktiv |
| Kopfleiste | Logo, drei Werkzeuge mit aktiver Markierung und Konto-Menü mit Name, Planabzeichen und Reitern. Gäste sehen nur API-Schlüssel, Anmelden und Registrieren. | alle | header.js:193-282, :300-345 | nicht erwähnt | aktiv |
| Dunkelmodus | Schalter in der Kopfleiste, im Browser gemerkt; Rechtsseiten folgen der Einstellung. | alle | header.js:161-181, :265, impressum.html:8-12 | nur im App-Mockup (pro-mode) | aktiv |
| Meldungen und Dialoge | Einheitliche Kurzmeldungen und Fenster für Hinweis, Bestätigen und Eingabe. | alle | ui.js:176, :224-373, :382 | nicht erwähnt | aktiv |
| Hilfetexte am Handy | Auf schmalen Bildschirmen erscheint der Hilfetext beim Antippen und verschwindet nach 5 Sekunden. | alle | ui.js:101-136 | nicht erwähnt | aktiv |
| Passwort-Auge | Blendet Passwort- und Schlüsselfelder ein und aus. | alle | account-modal.js:338 | nicht erwähnt | aktiv |
| Dokumentation | Doku-Fenster im Kalkulator, auch von der Startseite erreichbar. | alle | fairvaluepro.html:5559, :1458, home.html:179 | nur im App-Mockup (pro-mode) | aktiv |
| Alte Adressen | /app → /fairvalue und /trading-desk → /investment-planner, Parameter bleiben erhalten. | alle | api.py:951, :1007 | nicht erwähnt | aktiv |
| **Betrieb (nicht für Nutzer)** | | | | | |
| Admin-Bereich (/admin) | Für den Betreiber: Nutzerliste mit Onboarding-, Daten- und KI-Spalten, Funnel-Kennzahlen; Plan setzen (markiert „manuell“), Nutzer löschen, E-Mails sperren. | nur Betreiber | admin.html:406-930, api.py:986-1110, auth.py:588 | nicht erwähnt | aktiv |
| Admin „Tageslimit zurücksetzen“ | Zählt und setzt ein Tageslimit für Bewertungen zurück. | nur Betreiber | admin.html:845-868, api.py:1071, plans.py:162 | nicht erwähnt | unsicher: kein Plan hat ein Tageslimit, es greift nie |
| /healthz, /config | Erreichbarkeitsprüfung bzw. öffentliche Plan-, Preis- und Verkaufsangaben. | – | api.py:930, api.py:1113, plans.py:156 | nicht erwähnt | aktiv |
| Seiten-Platzhalter und Cache-Schutz | Setzt Plan- und Preistexte in jede Seite ein und hängt Versionskennungen an Skripte, damit nichts Altes geladen wird. | – | api.py:880-916, plans.py:37 | nicht erwähnt | aktiv |
| Sicherheitsköpfe | Strenge Inhaltsregeln, Schutz vor Einbettung, HTTPS-Pflicht; Antworten mit Schlüssel werden nie zwischengespeichert. | – | api.py:84-178 | nicht erwähnt | aktiv |
| Diagnose-Endpunkte | Wechselkurs-Prüfung (vom Tracker genutzt), FMP-Einzelabruf und Anbieter-Test ohne Anmeldung. | – | api.py:1290-1291, :1318, :3021 | nicht erwähnt | aktiv (Wechselkurs); unsicher: fmp-test/test-provider ohne Nutzer in der Oberfläche |
| Tageslimit-Abfrage (/auth/rate-limit) | Sollte verbleibende Bewertungen des Tages liefern. | PRO | api.py:431, auth.js:355 | nicht erwähnt | unsicher: kein Aufrufer, Limit inaktiv |

**Auffälligkeiten (Konto und Übergreifendes)**
1. **Widerruf weicht von den AGB ab:** Die AGB (§ 356 Abs. 5 BGB) sagen, beide Zustimmungen würden „im Bestellprozess separat eingeholt“. Die App hat einen einzigen Haken, dessen Text von Wertersatz und Erlöschen „mit vollständiger Vertragserfüllung“ spricht (plans.py:66-76). Offene Rechtsfrage, ohne Bewertung.
2. **Konto löschen entfernt den Nachweis der Zustimmung:** Es löscht auch `checkout_consents` (auth.py:795). Ein laufendes Abo wird sofort bei Stripe beendet (api.py:493-496), der Löschdialog sagt dazu nichts.
3. **Doppelt geöffnet in der Sperre:** Der Upgrade-Knopf öffnet den Kaufdialog (premium-gate.js:68) und über einen zweiten Listener gleichzeitig das Konto-Fenster (premium-gate.js:100-104).
4. **Onboarding lädt ui.js nicht:** Die Meldung „E-Mail bestätigt“ erscheint dort nicht. Den Dev-Knopf `#devOnboardingBtn` aus CLAUDE.md gibt es im Code nicht mehr.
5. **Toter oder veralteter Code:** `_aiKeyRowHtml`/`testAiKey` (account-modal.js:318, `testAiKey` nirgends definiert); `verify_url`-Zweig (fairvaluepro.html:1573); Kommentare „Gäste 3/Tag“ (fairvaluepro.html:1601, :6732) gegen „kein Limit“ (:4693).
