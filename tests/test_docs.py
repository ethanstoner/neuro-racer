"""The devlog is a deliverable, so broken links in it are bugs.

An image reference that silently 404s on GitHub is invisible locally, and
evaluate.py used to name screenshots by generation alone -- so two runs whose
champions shared a generation number would quietly overwrite each other.
"""
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "devlog"
MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _markdown_files():
    return sorted(DOCS.glob("*.md")) + [ROOT / "README.md", ROOT / "PROJECT_STATUS.md"]


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: p.name)
def test_every_local_link_resolves(path):
    missing = []
    for target in MD_LINK.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (path.parent / target).exists():
            missing.append(target)
    assert not missing, f"{path.name} references missing files: {missing}"


def test_devlog_entries_are_numbered_contiguously():
    names = sorted(p.name for p in DOCS.glob("*.md"))
    numbers = [int(n.split("-")[0]) for n in names]
    assert numbers == list(range(1, len(numbers) + 1)), names


def test_readme_links_every_devlog_entry():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for entry in DOCS.glob("*.md"):
        assert entry.name in readme, f"{entry.name} is not linked from the README"
