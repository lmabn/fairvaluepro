#!/usr/bin/env bash
# Prüft, dass die Anbieter-Preise und Summen in plaene.html / plaene-vorschau.html
# mit data/anbieter-preise.json übereinstimmen. Rechnet beide Summen (Partner-Link
# und regulär) je Zahlweise ungerundet selbst nach und vergleicht mit den auf
# ganze Euro gerundeten Werten in der JSON. Prüft außerdem, dass der Partner-
# Rabatt zum regulären Preis passt (reg × (1 − rabatt%) = partner, auf Cent genau).
# Warnt, wenn das EZB-Kursdatum älter als 30 Tage ist.
# Exit-Code 0 = alles ok. Nach jedem Touch an einer Preisstelle lokal laufen.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
json="$root/data/anbieter-preise.json"

if [ ! -f "$json" ]; then
  echo "FEHLER: $json fehlt." >&2
  exit 1
fi

python3 - <<PYEOF
import json, pathlib, re, sys, datetime, math

root = pathlib.Path("$root")
d = json.loads((root / "data/anbieter-preise.json").read_text(encoding="utf-8"))

ult_reg_m   = d["ultimate"]["regulaer_monatlich_usd"]
ult_reg_y_m = d["ultimate"]["regulaer_jaehrlich_usd_pro_monat"]
ult_reg_y_y = d["ultimate"]["regulaer_jaehrlich_usd_pro_jahr"]
ult_par_m   = d["ultimate"]["partner_monatlich_usd"]
ult_par_y_m = d["ultimate"]["partner_jaehrlich_usd_pro_monat"]
ult_par_y_y = d["ultimate"]["partner_jaehrlich_usd_pro_jahr"]
rabatt      = d["ultimate"]["partner_rabatt_prozent"]
rate        = d["wechselkurs"]["usd_pro_eur"]
kurs_dt     = d["wechselkurs"]["datum"]
sum_par_m   = d["summen_partner_eur"]["monatlich_ganze_euro"]
sum_par_y   = d["summen_partner_eur"]["jaehrlich_pro_monat_ganze_euro"]
auf_par_m   = d["summen_partner_eur"]["aufschluesselung_monatlich_ultimate_eur_cent"]
auf_par_y   = d["summen_partner_eur"]["aufschluesselung_jaehrlich_ultimate_eur_cent"]
sum_reg_m   = d["summen_regulaer_eur"]["monatlich_ganze_euro"]
sum_reg_y   = d["summen_regulaer_eur"]["jaehrlich_pro_monat_ganze_euro"]
max_m       = d["max_eur"]["monatlich"]
max_y_m     = d["max_eur"]["jaehrlich_pro_monat"]

errs = []
warns = []

# 1) Jährlich-pro-Monat × 12 == Jährlich-pro-Jahr (regulär und partner)
if ult_reg_y_m * 12 != ult_reg_y_y:
    errs.append(f"Ultimate regulär: 12 × {ult_reg_y_m} ({ult_reg_y_m*12}) ≠ {ult_reg_y_y}")
if round(ult_par_y_m * 12, 2) != ult_par_y_y:
    errs.append(f"Ultimate Partner: 12 × {ult_par_y_m} ({round(ult_par_y_m*12,2)}) ≠ {ult_par_y_y}")

# 2) Partner-Rabatt-Konsistenz: regulär × (1 − rabatt%) == partner (auf Cent genau)
fact = (100 - rabatt) / 100.0
exp_par_m   = round(ult_reg_m   * fact, 2)
exp_par_y_m = round(ult_reg_y_m * fact, 2)
if exp_par_m != ult_par_m:
    errs.append(f"Partner monatlich {ult_par_m} ≠ {exp_par_m} ({ult_reg_m} × {fact})")
if exp_par_y_m != ult_par_y_m:
    errs.append(f"Partner jährlich pro Monat {ult_par_y_m} ≠ {exp_par_y_m} ({ult_reg_y_m} × {fact})")

# 3) Summen nachrechnen (ungerundet, dann für die ganze-Euro-Anzeige gerundet)
calc_sum_par_m = round(max_m   + ult_par_m   / rate)
calc_sum_par_y = round(max_y_m + ult_par_y_m / rate)
calc_sum_reg_m = round(max_m   + ult_reg_m   / rate)
calc_sum_reg_y = round(max_y_m + ult_reg_y_m / rate)
if calc_sum_par_m != sum_par_m: errs.append(f"Partner-Summe monatlich JSON {sum_par_m} ≠ berechnet {calc_sum_par_m}")
if calc_sum_par_y != sum_par_y: errs.append(f"Partner-Summe jährlich (pro Monat) JSON {sum_par_y} ≠ berechnet {calc_sum_par_y}")
if calc_sum_reg_m != sum_reg_m: errs.append(f"Regulär-Summe monatlich JSON {sum_reg_m} ≠ berechnet {calc_sum_reg_m}")
if calc_sum_reg_y != sum_reg_y: errs.append(f"Regulär-Summe jährlich (pro Monat) JSON {sum_reg_y} ≠ berechnet {calc_sum_reg_y}")

# 4) Aufschlüsselungs-Cent-Beträge (Ultimate in €)
exp_auf_m = round(ult_par_m   / rate, 2)
exp_auf_y = round(ult_par_y_m / rate, 2)
if exp_auf_m != auf_par_m: errs.append(f"Aufschlüsselung monatlich Ultimate {auf_par_m} € ≠ berechnet {exp_auf_m} €")
if exp_auf_y != auf_par_y: errs.append(f"Aufschlüsselung jährlich Ultimate {auf_par_y} € ≠ berechnet {exp_auf_y} €")

# 5) Kursdatum älter als 30 Tage?
try:
    dt = datetime.date.fromisoformat(kurs_dt)
    heute = datetime.date.today()
    alt_tage = (heute - dt).days
    if alt_tage > 30:
        warns.append(f"EZB-Kursdatum {kurs_dt} ist {alt_tage} Tage alt (> 30) — bitte aktualisieren.")
except Exception as e:
    errs.append(f"Kursdatum {kurs_dt!r} unlesbar: {e}")

# 6) HTML-Stellen prüfen — nur Dateien mit „Ultimate"-Marker.
targets = [root / "plaene.html", root / "plaene-vorschau.html"]
rate_candidates = {f"{rate:.4f}".replace(".", ","), f"{rate:g}".replace(".", ",")}
usd_year_reg_fmt = f"{ult_reg_y_y:,}".replace(",", ".")   # 1.188
usd_year_par_fmt = f"{ult_par_y_y:,}".replace(",", ".")   # 891 (keine Trennung)
# Ultimate-Partner-Beträge im Format "104,25" / "74,25"
par_m_str = f"{ult_par_m:.2f}".replace(".", ",")
par_y_m_str = f"{ult_par_y_m:.2f}".replace(".", ",")
auf_m_str = f"{auf_par_m:.2f}".replace(".", ",")
auf_y_str = f"{auf_par_y:.2f}".replace(".", ",")
# Deutsches Datumsformat TT.MM.JJJJ aus ISO
try:
    kurs_dt_de = datetime.date.fromisoformat(kurs_dt).strftime("%d.%m.%Y")
except Exception:
    kurs_dt_de = kurs_dt

for f in targets:
    if not f.exists():
        continue
    txt = f.read_text(encoding="utf-8")
    if "Ultimate" not in txt:
        # Datei enthält den Anbieter-Block noch nicht; nicht prüfen.
        continue
    needles = [
        (f"{ult_reg_m} $",            f"Ultimate regulär monatlich {ult_reg_m} $"),
        (f"{ult_reg_y_m} $",          f"Ultimate regulär jährlich pro Monat {ult_reg_y_m} $"),
        (f"{usd_year_reg_fmt} $",     f"Ultimate regulär jährlich pro Jahr {usd_year_reg_fmt} $"),
        (f"{par_m_str} $",            f"Ultimate Partner monatlich {par_m_str} $"),
        (f"{par_y_m_str} $",          f"Ultimate Partner jährlich pro Monat {par_y_m_str} $"),
        (f"{usd_year_par_fmt} $",     f"Ultimate Partner jährlich pro Jahr {usd_year_par_fmt} $"),
        (f"≈ {sum_par_m} €",     f"Partner-Summe monatlich ≈ {sum_par_m} €"),
        (f"≈ {sum_par_y} €",     f"Partner-Summe jährlich (pro Monat) ≈ {sum_par_y} €"),
        (f"≈ {sum_reg_m} €",     f"Regulär-Summe monatlich ≈ {sum_reg_m} €"),
        (f"≈ {sum_reg_y} €",     f"Regulär-Summe jährlich (pro Monat) ≈ {sum_reg_y} €"),
        (f"{auf_m_str} €",            f"Aufschlüsselung monatlich Ultimate {auf_m_str} €"),
        (f"{auf_y_str} €",            f"Aufschlüsselung jährlich Ultimate {auf_y_str} €"),
        (kurs_dt_de,                       f"EZB-Kursdatum {kurs_dt_de}"),
    ]
    for needle, label in needles:
        if needle not in txt:
            errs.append(f"{f.name}: {label} nicht gefunden (gesucht: {needle!r})")
    if not any(c in txt for c in rate_candidates):
        errs.append(f"{f.name}: EZB-Kurs {rate} nicht gefunden (gesucht eine von {rate_candidates!r})")

for w in warns:
    print("WARN:", w)
if errs:
    print("Preis-/Summen-Konsistenz fehlgeschlagen:")
    for e in errs:
        print(" -", e)
    sys.exit(1)
print("OK: Anbieter-Preise und Summen konsistent mit data/anbieter-preise.json")
PYEOF
