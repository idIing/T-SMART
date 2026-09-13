"""Run-provenance manifest — a self-describing reproducibility block.

Every final run embeds one of these so a committed log answers, on its own,
"what code, what data, what command produced this?". Pure stdlib; never raises
(a missing git binary or absent package degrades to ``None``, it does not abort
an eval).
"""
from __future__ import annotations

import os
import subprocess
import sys

_PKGS = ("numpy", "scipy", "statsmodels", "google-genai", "ruptures", "openai", "python-dotenv")


def _git(args, cwd):
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, timeout=10
        )
        return out.decode().strip()
    except Exception:
        return None


def _pkg_versions(names=_PKGS):
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover - importlib.metadata is stdlib >=3.8
        return {}
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
        except Exception:
            out[n] = None
    return out


def _tsqa_path(root=None):
    """Directory of the ``tsqa`` package that actually got imported.

    Resolved via a real submodule rather than ``tsqa.__file__``: run from the
    repo root, the outer ``tsqa/`` project directory (which has no
    ``__init__.py``) claims the name as an implicit namespace package and
    ``tsqa.__file__`` is ``None``, which silently blanked this field.
    """
    try:
        from . import provenance as _self
        # .../tsqa/eval/provenance.py -> .../tsqa
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(_self.__file__)))
        return _relativize(pkg_dir, root)
    except Exception:
        return None


def _repo_root(cwd):
    """Absolute path of the git checkout containing ``cwd``, or None."""
    return _git(["rev-parse", "--show-toplevel"], cwd)


def _relativize(path, root):
    """Express ``path`` relative to the repo root, without leaking the machine.

    Paths inside the checkout become repo-relative ("." , "tsqa/tsqa"). Anything
    outside collapses to ``"<outside-repo>"``: that still answers the question
    this field exists for -- did a stale editable install resolve somewhere other
    than this checkout? -- while keeping usernames and directory layout out of a
    log that gets committed and published.
    """
    if not path:
        return path
    if not root:
        return "<unknown-root>"
    try:
        rel = os.path.relpath(path, root)
    except ValueError:  # different drive on Windows
        return "<outside-repo>"
    return "<outside-repo>" if rel.startswith(os.pardir) else rel


def run_provenance(extra=None):
    """Return a JSON-serializable provenance dict.

    Git facts are taken from the current working directory (where the run is
    launched), so they describe the actual checkout/worktree in use. ``extra``
    is merged in for run-specific identity (dataset size, seed, scoring tol).
    """
    import datetime
    import platform

    cwd = os.getcwd()
    root = _repo_root(cwd)
    prov = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"], cwd),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "git_dirty": bool(_git(["status", "--porcelain"], cwd)),
        "argv": list(sys.argv),
        # Paths are repo-relative and the hostname is opt-in: these manifests are
        # committed and published, so they must not carry the operator's username,
        # home-directory layout, or machine name.
        "cwd": _relativize(cwd, root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node() if os.environ.get("TSQA_PROVENANCE_HOSTNAME") else None,
        "packages": _pkg_versions(),
        # The engine code that actually ran — catches a stale editable install
        # resolving to a different checkout than the one on disk here.
        "tsqa_package_path": _tsqa_path(root),
    }
    if extra:
        prov.update(extra)
    return prov
