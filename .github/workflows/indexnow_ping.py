#!/usr/bin/env python3
"""IndexNow-Ping: nach jedem Merge auf main die geänderten Sitemap-Adressen
melden. Kein Konto, kein neuer Dienst — der Schlüssel liegt als
<KEY>.txt im Wurzelverzeichnis.

Logik
-----
- Diff gegen `HEAD~1` findet die geänderten HTML-Dateien (plus
  sitemap.xml selbst, falls sie sich bewegt hat).
- Mapping Datei → URL: index.html → `/`; akademie/index.html →
  `/akademie/`; sonst `/<pfad>`. Nur URLs, die als `<loc>` in der
  Sitemap stehen, werden gemeldet — damit landen Beta-/noindex-
  Seiten nie beim IndexNow-Dienst.
- Hat sich die Sitemap selbst verändert, meldet der Workflow alle
  Sitemap-URLs (Reindex).
- `workflow_dispatch` mit `reindex_all=true` meldet ebenfalls alle
  Sitemap-URLs (für manuelle Reindex-Pings).
- POST an `https://api.indexnow.org/indexnow` mit JSON-Payload.
  Erfolg = HTTP 200 oder 202; alles andere beendet mit exit 1.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

KEY = "425120dcb111ae6666ca4e678904b1bb"
HOST = "gobetter.trade"
KEY_LOCATION = f"https://{HOST}/{KEY}.txt"
INDEXNOW_API = "https://api.indexnow.org/indexnow"


def sh(*args: str) -> str:
    out = subprocess.run(args, check=True, capture_output=True, text=True)
    return out.stdout


def sitemap_urls() -> list[str]:
    xml = Path("sitemap.xml").read_text(encoding="utf-8")
    return re.findall(r"<loc>([^<]+)</loc>", xml)


def file_to_url(rel: str) -> str | None:
    if rel == "index.html":
        return f"https://{HOST}/"
    if rel == "akademie/index.html":
        return f"https://{HOST}/akademie/"
    if rel.endswith(".html"):
        return f"https://{HOST}/{rel}"
    return None


def changed_files() -> list[str]:
    try:
        out = sh("git", "diff", "--name-only", "HEAD~1", "HEAD")
    except subprocess.CalledProcessError as e:
        print("[indexnow] git diff schlug fehl:", e.stderr, file=sys.stderr)
        return []
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return [f for f in files if f.endswith(".html") or f == "sitemap.xml"]


def main() -> int:
    all_urls = sitemap_urls()
    print(f"[indexnow] Sitemap enthält {len(all_urls)} URLs.")
    reindex_all = os.environ.get("REINDEX_ALL", "false").lower() == "true"

    if reindex_all:
        print("[indexnow] REINDEX_ALL=true → alle Sitemap-URLs melden.")
        urls = all_urls[:]
    else:
        files = changed_files()
        print(f"[indexnow] Geänderte Dateien: {files or '(keine)'}")
        urls = []
        if "sitemap.xml" in files:
            print("[indexnow] sitemap.xml selbst geändert → alle Sitemap-URLs melden.")
            urls = all_urls[:]
        else:
            sm = set(all_urls)
            for f in files:
                u = file_to_url(f)
                if u and u in sm:
                    urls.append(u)
                elif u:
                    print(f"[indexnow]   {f} → {u} nicht in Sitemap, übersprungen")

    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    if not unique:
        print("[indexnow] Keine indexierbaren Änderungen — nichts zu melden.")
        return 0

    print(f"[indexnow] An IndexNow zu melden ({len(unique)}):")
    for u in unique:
        print(f"  {u}")

    payload = {
        "host": HOST,
        "key": KEY,
        "keyLocation": KEY_LOCATION,
        "urlList": unique,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        INDEXNOW_API,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            code = resp.status
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        code = e.code
        body = e.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[indexnow] Netzfehler: {e}", file=sys.stderr)
        return 1

    print(f"[indexnow] HTTP {code}")
    print(f"[indexnow] Antwort: {body!r}")
    if code in (200, 202):
        print("[indexnow] OK — IndexNow nimmt Melde-Request an.")
        return 0
    print(f"[indexnow] FAIL — HTTP {code}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
