import pathlib, re, shutil, subprocess, sys, tempfile, urllib.request

BASE = "https://deb.debian.org/debian/pool/main"
ARCHS = ["armel", "armhf", "arm64", "i386", "mipsel", "mips64el", "ppc64el"]
PKGS = [("b", "busybox", "busybox-static"), ("c", "coreutils", "coreutils"),
        ("b", "bash", "bash"), ("d", "dash", "dash"), ("g", "grep", "grep"),
        ("s", "sed", "sed"), ("g", "gawk", "gawk"), ("c", "curl", "curl"),
        ("w", "wget", "wget"), ("o", "openssh", "openssh-client"),
        ("n", "nano", "nano"), ("p", "procps", "procps"),
        ("t", "tar", "tar"), ("f", "findutils", "findutils"),
        ("u", "util-linux", "util-linux")]
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/root/lab/samples/clean_multiarch")
CACHE = pathlib.Path("/root/lab/debs")


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def newest(listing, pkg, arch):
    names = set(re.findall(rf'href="({re.escape(pkg)}_[^"]+_{arch}\.deb)"', listing))
    key = lambda n: [int(x) for x in re.findall(r"\d+", n[len(pkg) + 1:n.rfind("_")])]
    return max(names, key=key) if names else None


OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
stats = {a: 0 for a in ARCHS}
for letter, src, pkg in PKGS:
    try:
        listing = get(f"{BASE}/{letter}/{src}/").decode("utf-8", "replace")
    except Exception as e:
        print("немає каталогу", src, e)
        continue
    for arch in ARCHS:
        name = newest(listing, pkg, arch)
        if not name:
            continue
        deb = CACHE / name
        if not deb.exists():
            try:
                deb.write_bytes(get(f"{BASE}/{letter}/{src}/{name}"))
            except Exception as e:
                print("не завантажено", name, e)
                continue
        with tempfile.TemporaryDirectory() as tmp:
            if subprocess.run(["dpkg-deb", "-x", str(deb), tmp], capture_output=True).returncode:
                print("не розпаковано", name)
                continue
            dest = OUT / arch
            dest.mkdir(exist_ok=True)
            for p in pathlib.Path(tmp).rglob("*"):
                if not p.is_file() or p.is_symlink():
                    continue
                posix = "/" + p.relative_to(tmp).as_posix()
                if "/bin/" not in posix and "/sbin/" not in posix:
                    continue
                with open(p, "rb") as f:
                    if f.read(4) != b"\x7fELF":
                        continue
                shutil.copy(p, dest / f"{pkg}__{p.name}")
                stats[arch] += 1
    print("оброблено", pkg, flush=True)
print("ELF за архітектурами:", stats)