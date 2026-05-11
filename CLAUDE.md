# Workflow

## Branch & Merge

- Entwicklung erfolgt auf dem vorgegebenen `claude/<task>`-Branch.
- **Am Ende jeder Aufgabe** automatisch:
  1. Branch nach `origin` pushen
  2. Pull Request gegen `main` öffnen (via `mcp__github__create_pull_request`)
  3. Auto-Merge aktivieren (via `mcp__github__enable_pr_auto_merge`, `merge_method: squash`)
- Direkter Push auf `main` ist serverseitig blockiert (HTTP 403) — immer den PR-Weg gehen.
- Keine zusätzliche Bestätigung für PR-Erstellung nötig; der Auto-Merge ist die User-Default-Präferenz.
