#!/usr/bin/env python3
"""
Scan the current folder (recursively) for lines related to "approved"
and encoding/QuantumSubmethod usage, to understand how the Django DB
is structured and how whitelisting is implemented.

Usage:
    python scan_approved_references.py

Run it from the `gui` folder of your project.
"""

import os
from pathlib import Path

# Keywords we care about
KEYWORDS = [
    "approved",
    "QuantumSubmethod",
    "encoding",
    "submethod",
    "QuantumMethod",
]

# Files we care about most (these will be highlighted if present)
SPECIAL_FILE_NAMES = {
    "models.py",
    "views.py",
    "admin.py",
    "serializers.py",
    "settings.py",
    "urls.py",
    "tasks.py",
    "cli.py",
}


def iter_project_files(base: Path):
    """Yield Python and template files in the project."""
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        # Focus on Python + templates, but you can add more extensions
        if path.suffix in {".py", ".html", ".txt", ".json"}:
            yield path


def scan_file(path: Path):
    """Scan a single file for the KEYWORDS, return list of (lineno, line, hits)."""
    results = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return results

    for lineno, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        hits = [kw for kw in KEYWORDS if kw.lower() in lowered]
        if hits:
            results.append((lineno, line, hits))
    return results


def main():
    base = Path(__file__).resolve().parent
    print(f"Scanning from base directory: {base}\n")

    for path in sorted(iter_project_files(base)):
        matches = scan_file(path)
        if not matches:
            continue

        special_marker = ""
        if path.name in SPECIAL_FILE_NAMES:
            special_marker = "  <<-- IMPORTANT (django core file)"

        print(f"\n=== {path}{special_marker} ===")
        for lineno, line, hits in matches:
            hits_str = ", ".join(hits)
            # Show the line trimmed to avoid crazy-long output
            trimmed = line.rstrip()
            if len(trimmed) > 200:
                trimmed = trimmed[:200] + " [...]"
            print(f"{lineno:5d}: [{hits_str}] {trimmed}")

    print("\nScan complete.")
    print("Look especially at:")
    print("  - main/models.py (QuantumSubmethod definition)")
    print("  - any views/serializers/admin that reference 'approved' or 'QuantumSubmethod'")
    print("  - settings.py for DATABASES / INSTALLED_APPS")


if __name__ == "__main__":
    main()
