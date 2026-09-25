"""Host launcher: python isaac/run.py smoke|collect|analyze|all.

Pins the installed image, captures container output, and mounts only this project.
Existing simulation result files are never overwritten without --overwrite.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/isaac-sim:4.5.0"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "smoke", "collect", "analyze", "all"])
    p.add_argument("--out", type=Path, help="Output directory inside claude_try2")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--decompose", action="store_true")
    a = p.parse_args()
    out = (a.out or ROOT / "results" / ("isaac_smoke" if a.mode == "smoke" else "isaac")).resolve()
    rel = out.relative_to(ROOT)  # refuse an output path outside the mounted project
    inputs = ROOT / "isaac" / "inputs"
    if a.mode == "prepare" or (a.mode in ("smoke", "collect", "all") and not (inputs / "manifest.json").exists()):
        subprocess.run([sys.executable, str(ROOT / "isaac" / "prepare.py")], cwd=ROOT, check=True)
    if a.mode in ("smoke", "collect", "all"):
        subprocess.run(["docker", "image", "inspect", IMAGE], check=True, stdout=subprocess.DEVNULL)
        tasks = ["hold", "press", "seed1"] if a.mode == "smoke" else ["seed1", "seed2", "seed3", "seed11", "seed12", "press"]
        expected = [out / f"{v}_{task}.npz" for v in ("nom", "ref") for task in tasks]
        if not a.overwrite and any(f.exists() for f in expected):
            p.error("Some output files already exist. Choose --out or explicitly use --overwrite.")
        out.mkdir(parents=True, exist_ok=True)
        name = "fr3-isaac-" + uuid.uuid4().hex[:8]
        # -v handles the comma in the workspace name; --mount's CSV syntax does not.
        cmd = ["docker", "run", "--rm", "--name", name, "--gpus", "all",
               "-e", "ACCEPT_EULA=Y", "-e", "PRIVACY_CONSENT=N", "-e", "OMP_NUM_THREADS=1",
               "-v", str(ROOT) + ":/work", "-w", "/work", "--entrypoint", "/isaac-sim/python.sh", IMAGE,
               "-u", "/work/isaac/collect.py", "--out", "/work/" + rel.as_posix(), "--tasks", *tasks]
        if a.overwrite:
            cmd.append("--overwrite")
        with (out / "docker.log").open("w", encoding="utf-8") as log:
            log.write("Image: " + IMAGE + "\n")
            log.flush()
            print(f"Starting {name}; log: {out / 'docker.log'}", flush=True)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(("ISAAC ", "SAVED ")) or "Traceback" in line:
                        print(line.rstrip(), flush=True)
                code = process.wait()
            except KeyboardInterrupt:
                subprocess.run(["docker", "stop", "-t", "2", name], check=False, stdout=subprocess.DEVNULL)
                process.wait()
                raise
        if code:
            raise RuntimeError(f"Isaac exited with status {code}; inspect {out / 'docker.log'}")
        if not all(f.exists() for f in expected):
            raise RuntimeError("Isaac exited before all outputs were committed; inspect docker.log")
        image_id = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
        (out / "container.json").write_text(json.dumps(dict(image=IMAGE, image_id=image_id), indent=2), encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "isaac" / "validate.py"), "--data", str(out)], cwd=ROOT, check=True)
    if a.mode in ("analyze", "all"):
        cmd = [sys.executable, "-u", str(ROOT / "run_isaac.py"), "--data", str(out)]
        if a.decompose:
            cmd.append("--decompose")
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
