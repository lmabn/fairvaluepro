#!/usr/bin/env bash
# Audit: Keine Design-System-Kommentare im veröffentlichten HTML (Ludmila 20.11.2026).
# Fängt Kommentare wie <!-- @dsCard ... --> oder <!-- @dsPage ... -->, die aus dem
# Export des Design Systems stammen und nicht an Besucher ausgeliefert werden sollen.
#
# Prüft HTML-Dateien im Root und unter akademie/.
# Rückgabe: 0 wenn keine Treffer, 1 wenn mindestens einer gefunden.

set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

targets=( *.html akademie/*.html )

# Muster: HTML-Kommentar, dessen Inhalt mit "@ds" + Großbuchstabe beginnt
# (deckt @dsCard, @dsPage, @dsNote, @dsSection usw. ab).
pattern='<!--\s*@ds[A-Z]'

hits=$(grep -nP "$pattern" "${targets[@]}" 2>/dev/null || true)
if [ -n "$hits" ]; then
  echo "FEHLER: Design-System-Kommentare im veröffentlichten HTML gefunden:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "OK: Keine Design-System-Kommentare im veröffentlichten HTML."
