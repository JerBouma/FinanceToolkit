"""Checks a merged release PR and writes its release notes; used by the Release workflow.

A release PR has the head branch feature/vX.Y.Z. Any other branch is not a release: the outputs say so
and the workflow stops without an error. A release PR whose files, tag or description aren't ready
fails here, before anything is built or published.

Usage: python3 .github/scripts/release.py prepare
"""

import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = re.compile(r"feature/v(\d+\.\d+\.\d+)")


def output(**values: str) -> None:
    """Pass values to the next steps of the workflow (printed when run outside of GitHub Actions)."""
    with open(
        os.environ.get("GITHUB_OUTPUT", os.devnull), "a", encoding="utf-8"
    ) as file:
        for key, value in values.items():
            file.write(f"{key}={value}\n")
            print(f"{key}={value}")


def file_versions() -> dict[str, str]:
    """Every place the release version is written, as each channel reads it."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    server = json.loads((ROOT / "server.json").read_text())
    bundle = ROOT / "financetoolkit" / "mcp_server" / "mcpb"
    bundle_project = tomllib.loads((bundle / "pyproject.toml").read_text())["project"]
    pin = next(
        (
            dep.split("==")[1]
            for dep in bundle_project["dependencies"]
            if dep.startswith("financetoolkit[mcp]==")
        ),
        "missing",
    )
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = next(
        package["version"]
        for package in lock["package"]
        if package["name"] == "financetoolkit"
    )

    versions = {
        "pyproject.toml": project["version"],
        "server.json version": server["version"],
        "mcpb/manifest.json": json.loads((bundle / "manifest.json").read_text())[
            "version"
        ],
        "mcpb/pyproject.toml version": bundle_project["version"],
        "mcpb/pyproject.toml financetoolkit[mcp] pin": pin,
        "uv.lock": locked,
    }
    for index, package in enumerate(server["packages"]):
        versions[f"server.json packages[{index}]"] = package["version"]

    return versions


def environment_problem() -> str | None:
    """The publish job runs once the "release" environment approves it, so that approval has to exist.

    GitHub creates an environment a job names without any protection when it doesn't exist yet, which
    would publish without an approval; the release stops instead.
    """
    request = urllib.request.Request(
        f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}/environments/release",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            environment = json.load(response)
    except urllib.error.HTTPError as error:
        return f'the "release" environment could not be read (HTTP {error.code}); create it with a required reviewer'

    if not any(
        rule["type"] == "required_reviewers"
        for rule in environment.get("protection_rules", [])
    ):
        return 'the "release" environment has no required reviewer, so publishing would not wait for an approval'
    if environment.get("can_admins_bypass", True):
        return (
            'the "release" environment lets administrators skip the approval '
            '(turn off "Allow administrators to bypass")'
        )

    return None


def release_notes(body: str) -> str:
    """The PR description as release text: a short introduction followed by a "## Package" section."""
    body = body.replace("\r\n", "\n").strip()
    intro = body.split("\n## ", 1)[0].strip()

    if not intro or intro.startswith("## "):
        raise ValueError(
            "The PR description needs a short introduction before its first header."
        )
    if not re.search(r"(?m)^## Package\s*$", body):
        raise ValueError('The PR description needs a "## Package" section.')

    return body + "\n"


def prepare() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pull_request = event["pull_request"]

    branch = BRANCH.fullmatch(pull_request["head"]["ref"])
    if not branch:
        print(
            f"::notice::{pull_request['head']['ref']} is not a release branch (feature/vX.Y.Z), nothing to release."
        )
        output(release="false")
        return

    version = branch.group(1)
    sha = pull_request["merge_commit_sha"]
    problems = [
        f"{name} has {found}"
        for name, found in file_versions().items()
        if found != version
    ]

    # A fixed git command on a version the branch pattern validated.
    command = ["git", "rev-parse", "-q", "--verify", f"refs/tags/v{version}^{{commit}}"]
    tagged = subprocess.run(  # noqa: S603
        command,  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    ).stdout.strip()
    if tagged and tagged != sha:
        problems.append(
            f"the tag v{version} already exists on {tagged[:7]}, not on the merge commit {sha[:7]}"
        )

    if problem := environment_problem():
        problems.append(problem)

    try:
        (ROOT / "release-notes.md").write_text(
            release_notes(pull_request.get("body") or ""), encoding="utf-8"
        )
    except ValueError as error:
        problems.append(str(error))

    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        sys.exit(
            f"Release v{version} is not ready: {len(problems)} problem(s), see above."
        )

    print(f"Releasing v{version} from {sha}.")
    output(release="true", version=version, sha=sha)


if __name__ == "__main__":
    if sys.argv[1:] != ["prepare"]:
        sys.exit(__doc__)
    prepare()
