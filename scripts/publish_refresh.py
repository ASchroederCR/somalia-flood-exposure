"""Commit the refreshed app data and push it to GitHub (which redeploys Pages).

Used by the daily refresh task as one fixed, pre-approvable command:
    python scripts/publish_refresh.py

Stages only app/data and the district table, refuses to commit if the FloodHub API key appears in
anything staged, skips when nothing changed, pushes to origin/main with one rebase retry, never force-pushes.
Exit codes: 0 pushed or nothing to do, 2 key found (nothing committed), 3 rebase conflict, 4 push failed.
"""
import datetime, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATHS = ["app/data", "data/processed/district_exposure.csv"]
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


def main():
    git("add", *PATHS)

    # never print the key; only test whether it is in any staged file
    key = (ROOT / "FloodHub API Key.txt").read_text().strip().split()[-1]
    if git("grep", "--cached", "-F", "-q", key, check=False).returncode == 0:
        git("reset")
        print("STOP: the FloodHub API key appears in a staged file. Unstaged everything; nothing committed.")
        return 2

    others = [l[3:] for l in git("status", "--porcelain").stdout.splitlines()
              if l[:2].strip() and not any(l[3:].startswith(p) for p in PATHS)]
    if others:
        print("Left unstaged (not refresh data):", ", ".join(others))

    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("No data changes since the last commit; nothing committed or pushed.")
        return 0

    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    git("commit", "-m", f"Daily FloodHub refresh {day}", "-m", TRAILER)
    sha = git("rev-parse", "--short", "HEAD").stdout.strip()

    if git("push", "origin", "main", check=False).returncode != 0:
        if git("pull", "--rebase", "origin", "main", check=False).returncode != 0:
            git("rebase", "--abort", check=False)
            print(f"STOP: commit {sha} made, but rebasing onto origin/main conflicted; rebase aborted, nothing pushed.")
            return 3
        sha = git("rev-parse", "--short", "HEAD").stdout.strip()
        if git("push", "origin", "main", check=False).returncode != 0:
            print(f"STOP: commit {sha} made, but the push failed after rebasing.")
            return 4

    print(f"Pushed {sha} to origin/main.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
