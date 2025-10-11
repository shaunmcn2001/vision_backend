import pathlib
import re

# Flags:
#   - allow .eq(  on ee.* types and calls, but block obvious string-literal misuse on plain vars.
# Strategy:
#   - Grep for .eq(" or .eq('
#   - Whitelist lines where 'ee.' appears within ~80 chars before ".eq(" (likely legit) 
#   - Fail on anything else.


def test_no_string_eq_misuse_outside_ee():
    root = pathlib.Path(".")
    bad_hits = []
    pattern = re.compile(r"\.eq\((?:\"[^\"]*\"|'[^']*')\)")
    for p in root.rglob("*.py"):
        if any(part in {"venv", ".venv", "env", ".git", "__pycache__", "site-packages", "build", "dist"} for part in p.parts):
            continue
        if p.as_posix().endswith("tools/fix_eq_misuse.py"):
            # The refactor script documents the rule with string literals; ignore it.
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                # whitelist: 'ee.' appears shortly before '.eq('
                idx = line.find(".eq(")
                prefix = line[max(0, idx - 80):idx]
                if "ee." in prefix:
                    continue
                bad_hits.append(f"{p}:{i}: {line.strip()}")
    assert not bad_hits, "Found string-literal `.eq(...)` likely on Python vars:\n" + "\n".join(bad_hits)
