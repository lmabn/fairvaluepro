#!/usr/bin/env bash
# Audit: Keine typografischen Anführungszeichen in HTML-Attributen (Ludmila 19.11.2026).
# Ein `="` mit typografischem Zeichen (z. B. =”) macht Attribute für Browser unsichtbar —
# <meta robots noindex> und <link canonical> werden dadurch unwirksam.
#
# Prüft die ausgelieferten HTML-Dateien im Root und unter akademie/.
# Rückgabe: 0 wenn keine Treffer, 1 wenn mindestens einer gefunden.

set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

# Ausgeliefertes HTML — interne Arbeitsdateien aus der Pfad-Liste bewusst draußen.
targets=( *.html akademie/*.html )

# Muster: Attribut-Grenze '=' direkt gefolgt von einem typografischen Doppelanführungszeichen.
pattern='=[\xe2\x80\x9c\xe2\x80\x9d\xe2\x80\x9e\xe2\x80\xb3\xe2\x80\xb6\xc2\xab\xc2\xbb]'

hits=$(grep -nP "$pattern" "${targets[@]}" 2>/dev/null || true)
if [ -n "$hits" ]; then
  echo "FEHLER: Typografische Anführungszeichen in HTML-Attributen gefunden:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "OK: Keine typografischen Anführungszeichen in HTML-Attributen."
