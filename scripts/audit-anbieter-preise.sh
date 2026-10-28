#!/usr/bin/env bash
# Prüft, dass die Anbieter-Preise und Summen in plaene.html / plaene-vorschau.html
# mit data/anbieter-preise.json übereinstimmen. Rechnet die Summen aus Rohwerten
# (MAX-Preis + Ultimate / EUR-USD-Kurs) selbst nach, beide Zahlweisen (monatlich
# und jährlich pro Monat), und meldet jede Abweichung mit Datei und Zeile.
# Exit-Code 0 = alles ok. Nach jedem Touch an einer Preisstelle lokal laufen lassen.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
json="$root/data/anbieter-preise.json"

if [ ! -f "$json" ]; then
  echo "FEHLER: $json fehlt." >&2
  exit 1
fi

python3 - <<PYEOF
import json, pathlib, re, sys

root = pathlib.Path("$root")
d = json.loads((root / "data/anbieter-preise.json").read_text(encoding="utf-8"))

ult_m   = d["ultimate"]["monatlich_usd"]
ult_y_m = d["ultimate"]["jaehrlich_usd_pro_monat"]
ult_y_y = d["ultimate"]["jaehrlich_usd_pro_jahr"]
rate    = d["wechselkurs"]["usd_pro_eur"]
kurs_dt = d["wechselkurs"]["datum"]
sum_m   = d["summen"]["monatlich_eur_ganze_euro"]
sum_y   = d["summen"]["jaehrlich_eur_ganze_euro_pro_monat"]
max_m   = d["max_eur"]["monatlich"]
max_y_m = d["max_eur"]["jaehrlich_pro_monat"]

errs = []

# 1) 12 × jährlich-pro-Monat == jährlich-pro-Jahr
if ult_y_m * 12 != ult_y_y:
    errs.append(f"Ultimate jährlich: 12 × {ult_y_m} ({ult_y_m*12}) ≠ {ult_y_y}")

# 2) Summen gegenchecken
calc_sum_m = round(max_m   + ult_m   / rate)
calc_sum_y = round(max_y_m + ult_y_m / rate)
if calc_sum_m != sum_m:
    errs.append(f"Monatssumme JSON {sum_m} ≠ berechnet {calc_sum_m} (= {max_m} + {ult_m}/{rate})")
if calc_sum_y != sum_y:
    errs.append(f"Jahressumme JSON {sum_y} ≠ berechnet {calc_sum_y} (= {max_y_m} + {ult_y_m}/{rate})")

# 3) HTML-Stellen prüfen — nur Dateien, die den Anbieter-Preisblock bereits
#    enthalten (Marker „Ultimate"). plaene.html wird so vor dem Nachzug nicht
#    fehlerhaft gemeldet.
targets = [root / "plaene.html", root / "plaene-vorschau.html"]
rate_candidates = {f"{rate:.4f}".replace(".", ","), f"{rate:g}".replace(".", ",")}
usd_year_formatted = f"{ult_y_y:,}".replace(",", ".")  # 1.188

for f in targets:
    if not f.exists():
        continue
    txt = f.read_text(encoding="utf-8")
    if "Ultimate" not in txt:
        # Datei enthält den Anbieter-Block noch nicht; wird nicht geprüft.
        continue
    needles = [
        (f"{ult_m} $",              f"Ultimate monatlich {ult_m} $"),
        (f"{ult_y_m} $",            f"Ultimate jährlich pro Monat {ult_y_m} $"),
        (f"{usd_year_formatted} $", f"Ultimate jährlich pro Jahr {usd_year_formatted} $"),
        (f"≈ {sum_m} €",       f"Summe monatlich ≈ {sum_m} €"),
        (f"≈ {sum_y} €",       f"Summe jährlich (pro Monat) ≈ {sum_y} €"),
        (kurs_dt,                        f"EZB-Kursdatum {kurs_dt}"),
    ]
    for needle, label in needles:
        if needle not in txt:
            errs.append(f"{f.name}: {label} nicht gefunden (gesucht: {needle!r})")
    if not any(c in txt for c in rate_candidates):
        errs.append(f"{f.name}: EZB-Kurs {rate} nicht gefunden (gesucht eine von {rate_candidates!r})")

if errs:
    print("Preis-/Summen-Konsistenz fehlgeschlagen:")
    for e in errs:
        print(" -", e)
    sys.exit(1)

print("OK: Anbieter-Preise und Summen konsistent mit data/anbieter-preise.json")
PYEOF
