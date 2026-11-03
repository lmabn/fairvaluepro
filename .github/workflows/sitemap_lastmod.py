#!/usr/bin/env python3
"""Setzt lastmod in sitemap.xml automatisch aus der Versionsgeschichte.

Für jede <url> wird der <loc> auf einen Pfad im Repo abgebildet und das
Datum des letzten Commits bestimmt, der diese Datei geändert hat
(git log -1 --format=%cs). Nur <lastmod>-Zeilen werden angefasst.

Zwei Modi:
  * default (ohne Flag): schreibt sitemap.xml bei Abweichung, exit 0
    — zum lokalen Lauf vor dem Push.
  * --check: schreibt nichts, exit 1 bei Abweichung — für den
    PR-Workflow, der die Prüfung durchreicht statt zu pushen.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SITEMAP = ROOT / "sitemap.xml"
HOST = "https://gobetter.trade/"


def loc_to_path(loc: str) -> Path | None:
    """Ordnet eine Sitemap-Loc einem Pfad im Repo zu, oder None."""
    if not loc.startswith(HOST):
        return None
    rel = loc[len(HOST):]
    rel = urllib.parse.unquote(rel)
    if not rel or rel.endswith("/"):
        rel = rel + "index.html"
    p = (ROOT / rel).resolve()
    try:
        p.relative_to(ROOT)
    except ValueError:
        return None
    if not p.is_file():
        return None
    return p


def git_last_change_date(path: Path) -> str | None:
    """Datum des letzten Commits, der diese Datei geändert hat (YYYY-MM-DD)."""
    rel = path.relative_to(ROOT)
    r = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", str(rel)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        return None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Nichts schreiben; exit 1 bei Abweichung (für PR-Prüfung).",
    )
    args = parser.parse_args()

    src = SITEMAP.read_text(encoding="utf-8")

    # Jede <url>-Gruppe durchlaufen; <loc> extrahieren, <lastmod> ersetzen.
    url_pat = re.compile(r"<url>\s*(.*?)\s*</url>", re.DOTALL)
    loc_pat = re.compile(r"<loc>([^<]+)</loc>")
    lastmod_pat = re.compile(r"<lastmod>([^<]+)</lastmod>")

    changes: list[tuple[str, str, str]] = []
    missing: list[str] = []

    def repl(m: re.Match[str]) -> str:
        block = m.group(0)
        loc_m = loc_pat.search(block)
        if not loc_m:
            return block
        loc = loc_m.group(1).strip()
        path = loc_to_path(loc)
        if path is None:
            missing.append(loc)
            return block
        new_date = git_last_change_date(path)
        if not new_date:
            missing.append(loc)
            return block
        lm_m = lastmod_pat.search(block)
        if not lm_m:
            return block
        old_date = lm_m.group(1).strip()
        if old_date == new_date:
            return block
        changes.append((loc, old_date, new_date))
        return block[: lm_m.start()] + f"<lastmod>{new_date}</lastmod>" + block[lm_m.end():]

    new_src = url_pat.sub(repl, src)

    if changes:
        print(f"[sitemap-lastmod] {len(changes)} Datum/Daten aktualisiert:")
        for loc, old, new in changes:
            print(f"  {loc}: {old} → {new}")
    else:
        print("[sitemap-lastmod] keine Änderungen nötig")

    if missing:
        print(f"[sitemap-lastmod] HINWEIS — {len(missing)} URL(s) ohne Zuordnung oder ohne git-Historie:")
        for loc in missing:
            print(f"  {loc}")

    if new_src != src:
        if args.check:
            print(
                "[sitemap-lastmod] FEHLER — sitemap.xml ist nicht aktuell. "
                "Lokal 'python3 .github/workflows/sitemap_lastmod.py' laufen "
                "lassen und das Ergebnis mit committen."
            )
            return 1
        SITEMAP.write_text(new_src, encoding="utf-8")
        print(f"[sitemap-lastmod] sitemap.xml neu geschrieben.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
