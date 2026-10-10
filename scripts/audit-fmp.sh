#!/usr/bin/env bash
# Audit der FMP-Referral-Links im Repo.
#
# Prüft:
#   1. Jeder FMP-href zeigt exakt auf die EXPECTED-URL (Parameter inklusive).
#   2. Jeder FMP-Link hat target="_blank" und rel="noopener".
#   3. Der Gutscheincode "gobetter25" steht nirgends abtippbar — nur als
#      Query-Parameter innerhalb der Referral-URL.
#
# Exit-Code 0 bei OK, 1 bei Verstoß. Zählt die Fundstellen.

set -euo pipefail
cd "$(dirname "$0")/.."

EXPECTED='https://site.financialmodelingprep.com/pricing-plans?couponCode=gobetter25'
FAIL=0

echo "FMP-Referral-Audit"
echo "  Erwartet: $EXPECTED"
echo

# --- 1. Alle href-Werte zu financialmodelingprep müssen EXPECTED sein ---
BAD_HREFS=$(grep -rnEo 'href="[^"]*financialmodelingprep[^"]*"' --include='*.html' . \
  | grep -v "href=\"$EXPECTED\"" || true)
if [ -n "$BAD_HREFS" ]; then
  echo "FEHLER: Abweichende FMP-Links (href stimmt nicht mit EXPECTED):"
  echo "$BAD_HREFS"
  FAIL=1
fi

# --- 2. Jede <a>-Zeile mit FMP-href muss target und rel haben ---
MISSING_ATTRS=$(grep -rnE 'href="[^"]*financialmodelingprep[^"]*"' --include='*.html' . \
  | awk -F: '{
      file=$1; line=$2;
      $1=""; $2=""; sub(/^: : /,"");
      content=$0;
      ok_target = (content ~ /target="_blank"/);
      ok_rel    = (content ~ /rel="noopener"/);
      if (!ok_target || !ok_rel) {
        missing = "";
        if (!ok_target) missing = missing " target=\"_blank\"";
        if (!ok_rel)    missing = missing " rel=\"noopener\"";
        printf "%s:%s FEHLT:%s\n", file, line, missing;
      }
    }' || true)
if [ -n "$MISSING_ATTRS" ]; then
  echo "FEHLER: FMP-Links ohne target/rel:"
  echo "$MISSING_ATTRS"
  FAIL=1
fi

# --- 3. Gutscheincode darf nirgends abtippbar stehen ---
BAD_COUPON=$(grep -rnE 'gobetter25' --include='*.html' --include='*.md' --include='*.js' --include='*.css' --include='*.xml' --include='*.json' . \
  | grep -v "couponCode=gobetter25" \
  | grep -v "scripts/audit-fmp.sh" || true)
if [ -n "$BAD_COUPON" ]; then
  echo "FEHLER: Gutscheincode 'gobetter25' außerhalb der Referral-URL gefunden:"
  echo "$BAD_COUPON"
  FAIL=1
fi

# --- Report ---
COUNT=$(grep -rF -c "href=\"$EXPECTED\"" --include='*.html' . 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
echo
if [ "$FAIL" -eq 0 ]; then
  echo "OK: $COUNT FMP-Referral-Link(s) im Repo, alle korrekt."
  exit 0
else
  echo "AUDIT FAILED. Oben stehen die Fundstellen."
  exit 1
fi
