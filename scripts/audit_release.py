from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scrac.config import load_paper_config


FORBIDDEN_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".vscode",
    "__pycache__",
    "best_model_architecture.md",
    "experiment_results.md",
    "ONE_STEP_FM_ANALYSIS.md",
    "ONE_STEP_FM_README.md",
    "project.md",
    "read.txt",
}

DOCUMENT_RISK_PHRASES = (
    "legacy",
    "historical version",
    "does not prove",
    "should not be interpreted",
    "we do not claim",
    "rather than",
    "not intended",
    "limitation",
)

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SENSITIVE_PATTERNS = {
    "email address": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
    ),
    "local user path": re.compile(
        r"(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Z]:\\Users\\[^\\\s]+)",
        re.IGNORECASE,
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential assignment": re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|password)"
        r"\s*[:=]\s*['\"][^'\"]+['\"]",
        re.IGNORECASE,
    ),
    "identity metadata field": re.compile(
        r"^\s*(?:author|authors|maintainer|contact(?: name)?|affiliation|institution|e-?mail)\s*[:=]",
        re.IGNORECASE | re.MULTILINE,
    ),
}


def iter_text_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]


def main() -> None:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        if path.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden release item: {path.relative_to(ROOT)}")

    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in DOCUMENT_RISK_PHRASES:
            if phrase in text:
                errors.append(
                    f"document risk phrase in {path.relative_to(ROOT)}: {phrase!r}"
                )

    audit_path = Path(__file__).resolve()
    for path in iter_text_files():
        if path.resolve() == audit_path:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                errors.append(
                    f"potential {label} in {path.relative_to(ROOT)}"
                )

    for path in ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")

    try:
        load_paper_config(ROOT / "configs" / "paper.yaml")
    except Exception as exc:
        errors.append(f"paper configuration validation failed: {exc}")

    expected_dataset_files = (
        ROOT / "datasets" / "README.md",
        ROOT / "datasets" / "download_geo.sh",
        ROOT / "datasets" / "prepare_h5ad.py",
    )
    for path in expected_dataset_files:
        if not path.is_file():
            errors.append(f"missing dataset release file: {path.relative_to(ROOT)}")

    if errors:
        print("RELEASE AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("RELEASE AUDIT PASSED")


if __name__ == "__main__":
    main()
