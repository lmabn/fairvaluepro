#!/usr/bin/env bash
# Audit: Kein FMP-Partner-Link auf der Website (Ludmila 15.11.2026, Punkt 5.1 Etappe 4).
#
# Seit die neuen Produktseiten live sind, läuft der Rabatt beim Datenanbieter
# nur noch über die App. Auf der Website darf deshalb:
#   1. KEIN <a href="…financialmodelingprep…"> mit couponCode stehen.
#   2. Der Gutscheincode "gobetter25" NIRGENDS abtippbar auftauchen
#      (nicht im Fließtext, nicht in Scripts, nicht in Daten).
#
# Website-Content prüft dieses Skript. Interne Dateien (CLAUDE.md, docs/, scripts/,
# data/) sind durch _config.yml-Exclude ohnehin nicht öffentlich — werden auch
# hier ausgenommen, damit sie ohne Fehler Doku-Hinweise auf frühere Regelstände
# enthalten dürfen (z. B. historische Erwähnung).
#
# Exit-Code 0 bei OK, 1 bei Verstoß.

set -euo pipefail
cd "$(dirname "$0")/.."

FAIL=0

echo "FMP-Audit (keine Partner-Links mehr auf der Website)"
echo

# --- 1. Kein href mit couponCode im HTML ---
BAD_HREFS=$(grep -rnE 'href="[^"]*financialmodelingprep[^"]*couponCode[^"]*"' --include='*.html' . || true)
if [ -n "$BAD_HREFS" ]; then
  echo "FEHLER: FMP-Links mit couponCode im HTML:"
  echo "$BAD_HREFS"
  FAIL=1
fi

# Ein allgemeiner Link auf financialmodelingprep.com ohne couponCode ist erlaubt
# (z. B. im Impressum/Datenschutz als Anbietername). Melde zur Info:
INFO_HREFS=$(grep -rnE 'href="[^"]*financialmodelingprep[^"]*"' --include='*.html' . \
  | grep -v couponCode || true)
if [ -n "$INFO_HREFS" ]; then
  echo "Hinweis: FMP-Links ohne couponCode (erlaubt):"
  echo "$INFO_HREFS"
  echo
fi

# --- 2. Gutscheincode "gobetter25" darf nirgends im veröffentlichten
# Website-Content stehen. Ausgenommen: interne Dateien (nicht veröffentlicht).
BAD_COUPON=$(git ls-files -- \
    '*.html' '*.css' '*.js' '*.json' '*.xml' '*.txt' '*.md' \
    ':(exclude)CLAUDE.md' \
    ':(exclude)docs/*' \
    ':(exclude)scripts/*' \
    ':(exclude)data/*' \
    2>/dev/null \
  | xargs -d '\n' grep -nE 'gobetter25' 2>/dev/null || true)
if [ -n "$BAD_COUPON" ]; then
  echo "FEHLER: Gutscheincode 'gobetter25' im veröffentlichten Website-Content:"
  echo "$BAD_COUPON"
  FAIL=1
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "OK: Kein FMP-Partner-Link mit Gutschein auf der Website, kein abtippbarer Code."
  exit 0
else
  echo "AUDIT FAILED. Oben stehen die Fundstellen."
  exit 1
fi
