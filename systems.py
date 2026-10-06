"""
Named RHS systems: drafts for the session, permanent ones on disk.

  save_system(name, source)   write a draft, check it (describe), return the result
  keep_system(name)           move a passing draft into SYSTEMS_DIR (permanent)
  list_systems()              both, each with its contract status

Drafts live in STUDIO_ROOT/drafts/ (~/.cache/bif-studio/drafts): they survive
an MCP reconnect (which restarts this server) and are removed by
studio.cleanup() once untouched for its threshold (24 h from the /bif pane).
A draft may shadow a permanent system of the same name; lookups prefer the
draft (it is the newer work) until it is kept or expires.

Writing through these tools works for any MCP client, with no file access of
its own.
"""

import os
import re
from pathlib import Path

import contract
import studio

HERE = Path(__file__).parent
SYSTEMS_DIR = Path(os.environ.get("BIF_SYSTEMS_DIR", HERE / "systems"))
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_SOURCE = 200_000


def drafts_dir() -> Path:
    d = studio.STUDIO_ROOT / studio.DRAFTS
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError(f"system name must match {NAME_RE.pattern} "
                         f"(lowercase, digits, underscore; e.g. 'chua' or 'lif_adaptive'), got {name!r}")
    return name


def _brief(desc: dict) -> dict:
    """The parts of describe() worth returning to a model."""
    out = {"ok": desc["ok"], "errors": desc["errors"], "warnings": desc["warnings"]}
    if desc["ok"]:
        out["kind"] = desc["system"]["kind"]
        out["params"] = [f"{q['name']}={q['default']:g} [{q['min']:g}, {q['max']:g}]"
                         + (" sweepable" if q["sweepable"] else "")
                         if q["default"] is not None else q["name"] for q in desc["params"]]
        out["state"] = [s["name"] for s in desc["state"]]
    return out


def save_system(name: str, source: str) -> dict:
    name = _check_name(name)
    if len(source) > MAX_SOURCE:
        raise ValueError(f"source is {len(source)} characters; the limit is {MAX_SOURCE}")
    path = drafts_dir() / f"{name}.py"
    tmp = path.with_suffix(".py.tmp")
    tmp.write_text(source)
    tmp.replace(path)
    desc = contract.describe(str(path))
    permanent = SYSTEMS_DIR / f"{name}.py"
    out = {"name": name, "path": str(path), "draft": True, **_brief(desc),
           "shadowsPermanent": permanent.exists()}
    out["next"] = (f"Fix the errors and call save_system again with the same name."
                   if not desc["ok"] else
                   f"The user can open it with /bif {name}. Call keep_system('{name}') when the "
                   f"user wants it kept permanently"
                   + (" (it would replace the permanent one: pass overwrite=true only if they agree)."
                      if out["shadowsPermanent"] else "."))
    return out


def keep_system(name: str, overwrite: bool = False) -> dict:
    name = _check_name(name)
    draft = drafts_dir() / f"{name}.py"
    if not draft.exists():
        raise ValueError(f"no draft named {name!r}; save it with save_system first")
    desc = contract.describe(str(draft))
    if not desc["ok"]:
        raise ValueError(f"draft {name!r} breaks the contract; fix it first:\n" + "\n".join(desc["errors"]))
    target = SYSTEMS_DIR / f"{name}.py"
    replaced = target.exists()
    if replaced and not overwrite:
        raise ValueError(f"a permanent system {name!r} already exists ({target}); pass overwrite=true "
                         f"to replace it (ask the user first), or save under another name")
    SYSTEMS_DIR.mkdir(parents=True, exist_ok=True)
    draft.replace(target)
    return {"name": name, "path": str(target), "replaced": replaced,
            "next": f"Kept. Open it with /bif {name}."}


def list_systems() -> dict:
    found = []
    for where, d in (("draft", drafts_dir()), ("permanent", SYSTEMS_DIR)):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            if not NAME_RE.match(f.stem):
                continue
            desc = contract.describe(str(f))
            entry = {"name": f.stem, "where": where, "path": str(f), "ok": desc["ok"]}
            if desc["ok"]:
                entry["kind"] = desc["system"]["kind"]
                entry["params"] = [q["name"] for q in desc["params"]]
            else:
                entry["errors"] = len(desc["errors"])
            found.append(entry)
    # Name -> the file /bif opens: the draft when both exist.
    resolve = {}
    for e in found:
        resolve.setdefault(e["name"], e["path"])
    return {"systems": found, "resolve": resolve,
            "systemsDir": str(SYSTEMS_DIR), "draftsDir": str(drafts_dir())}
