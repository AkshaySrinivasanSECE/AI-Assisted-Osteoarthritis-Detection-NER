"""Run the complete Python and frontend verification suite with one command."""

from pathlib import Path
import shutil
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"


def project_python():
    candidates = [
        ROOT_DIR / ".venv" / "Scripts" / "python.exe",
        ROOT_DIR / ".venv" / "bin" / "python",
    ]
    return next((str(path) for path in candidates if path.exists()), sys.executable)


def npm_command():
    executable = shutil.which("npm.cmd") or shutil.which("npm")
    if executable is None:
        raise RuntimeError("npm is required to run the frontend test suite.")
    return executable


def main():
    python = project_python()
    npm = npm_command()
    checks = [
        (
            "Python data, ML, API, and explainability tests",
            [python, "-m", "unittest", "discover", "-s", "tests", "-v"],
            ROOT_DIR,
        ),
        ("Frontend behavior tests", [npm, "test"], FRONTEND_DIR),
        ("Frontend lint", [npm, "run", "lint"], FRONTEND_DIR),
        ("Frontend production build", [npm, "run", "build"], FRONTEND_DIR),
    ]

    results = []
    for name, command, working_directory in checks:
        print(f"\n{'=' * 80}\n{name}\n{'=' * 80}", flush=True)
        completed = subprocess.run(command, cwd=working_directory, check=False)
        results.append((name, completed.returncode))

    print(f"\n{'=' * 80}\nFINAL TEST REPORT\n{'=' * 80}")
    for name, return_code in results:
        print(f"{'PASS' if return_code == 0 else 'FAIL'}  {name}")

    failures = [name for name, return_code in results if return_code != 0]
    if failures:
        print(f"\nFailed checks: {len(failures)}")
        return 1

    print(f"\nAll {len(results)} test groups passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
