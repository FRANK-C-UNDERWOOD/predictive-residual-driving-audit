"""Pin and fetch an official GitHub snapshot without repository-history transfer."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "vendor/spikefpn_snapshot"
API = "https://api.github.com/repos/EMI-Group/spikefpn"
HEADERS = {"User-Agent": "SCI-research-reproduction/1.0"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90) as response:
        return json.load(response)


def main():
    commit = get_json(API + "/commits/main")["sha"]
    tree = get_json(API + "/git/trees/" + commit + "?recursive=1")
    if tree.get("truncated"):
        raise RuntimeError("Incomplete upstream tree")
    members = [item for item in tree["tree"] if item["type"] == "blob" and (
        item["path"].endswith(".py")
        or item["path"] in ("LICENSE", "README.md", "requirements.txt", "weights/gad_weight.pth")
    )]
    TARGET.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        path = (TARGET / item["path"]).resolve()
        path.relative_to(TARGET.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/EMI-Group/spikefpn/" + commit + "/" + item["path"]
        expected_size = item["size"]
        hasher = hashlib.sha1(("blob %d\0" % expected_size).encode())
        temporary = path.with_suffix(path.suffix + ".part")
        print("Fetching " + item["path"] + " (%d bytes)" % expected_size, flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90) as response:
            count = 0
            with temporary.open("wb") as stream:
                while True:
                    chunk = response.read(1024**2)
                    if not chunk:
                        break
                    count += len(chunk)
                    stream.write(chunk)
                    hasher.update(chunk)
        if count != expected_size or hasher.hexdigest() != item["sha"]:
            raise RuntimeError("Official Git blob verification failed: " + item["path"])
        temporary.replace(path)
        return dict(path=item["path"], bytes=count, git_blob_sha1=item["sha"],
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, members))
    receipt = dict(repository="https://github.com/EMI-Group/spikefpn", commit=commit,
                   tree_sha=tree["sha"], files=records, method="GitHub API commit pin and verified raw Git blobs")
    (TARGET / "_UPSTREAM.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Official snapshot verified at commit " + commit, flush=True)


if __name__ == "__main__":
    main()
