"""Download every source in data/sources.yaml to data/raw/<id>.txt."""

import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "sources.yaml"
RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "agora-downloader/0.1"


def load_sources(manifest: Path = MANIFEST) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def download(source: dict[str, str], raw_dir: Path = RAW_DIR) -> Path:
    target = raw_dir / f"{source['id']}.txt"
    if target.exists():
        print(f"skip  {source['id']} (already at {target.relative_to(ROOT)})")
        return target

    request = urllib.request.Request(source["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")

    # Write to a temp file first so an interrupted download is never mistaken for a finished one.
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".txt.part")
    partial.write_text(text, encoding="utf-8", newline="")
    partial.replace(target)
    print(f"saved {source['id']} -> {target.relative_to(ROOT)} ({len(text):,} chars)")
    return target


def main() -> None:
    for source in load_sources():
        download(source)


if __name__ == "__main__":
    main()
