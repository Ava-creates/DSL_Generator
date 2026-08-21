#!/usr/bin/env python3
"""Generate program-form classification appendix (LaTeX) for a pipeline run."""

from __future__ import annotations

import argparse
import json
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROFILES = {
    12: {
        "locating": {
            "PICK",
            "USE_TOOL",
            "CRAFT",
            "NAVIGATE_TO_PRIMITIVE",
            "NAVIGATE_TO_WORKSHOP",
            "NAVIGATE_TO_OBSTACLE",
            "GET",
        },
        "explicit": {"MOVE", "TURN"},
        "section_label": "app:program-form",
        "decision_rule": [
            "A statement is a \\emph{locating} terminal if it determines the cell it acts on",
            "at execution time rather than from the program text. In the Run~12 DSL~1",
            "grammar every terminal except \\texttt{MOVE} is locating:",
            "\\texttt{NAVIGATE\\_TO\\_PRIMITIVE}, \\texttt{NAVIGATE\\_TO\\_WORKSHOP},",
            "\\texttt{NAVIGATE\\_TO\\_OBSTACLE}, \\texttt{PICK}, \\texttt{CRAFT},",
            "\\texttt{USE\\_TOOL} and \\texttt{GET} all search or plan from the current grid",
            "(Table~\\ref{tab:app-run-terminals}).",
            "A statement is an \\emph{explicit movement} terminal if it is \\texttt{MOVE};",
            "it names a step count and encodes one layout.",
            "This run's grammar has no \\texttt{TURN} terminal.",
        ],
        "terminal_caption_extra": (
            "Rows where the CFG description and the implemented behaviour disagree "
            "are the last three."
        ),
        "terminal_label": "tab:app-run-terminals",
        "form_counts_label": "tab:app-form-counts",
        "form_by_task_label": "tab:app-form-by-task",
        "mixed_label": "tab:app-mixed-programs",
        "locating_label": "tab:app-locating-programs",
        "explicit_label": "tab:app-explicit-programs",
        "include_explicit_column": False,
        "terminal_order": [
            "NAVIGATE_TO_PRIMITIVE",
            "NAVIGATE_TO_WORKSHOP",
            "NAVIGATE_TO_OBSTACLE",
            "GET",
            "MOVE",
            "PICK",
            "CRAFT",
            "USE_TOOL",
        ],
        "impl_notes": {
            "NAVIGATE_TO_PRIMITIVE": "As described.",
            "NAVIGATE_TO_WORKSHOP": "As described.",
            "NAVIGATE_TO_OBSTACLE": "As described.",
            "GET": (
                "Fixed exploration pattern over primitives and workshops; does not "
                "perform the described planning. No synthesized program in this run "
                "uses \\texttt{GET}."
            ),
            "MOVE": "As described. One action per step.",
            "PICK": (
                "Breadth-first search to the nearest instance of the named resource, "
                "face it, then \\texttt{USE}."
            ),
            "CRAFT": (
                "Plans a path to a free cell adjacent to the named workshop, faces it, "
                "then \\texttt{USE}."
            ),
            "USE_TOOL": (
                "Breadth-first search to a free cell adjacent to any obstacle of that "
                "type, turn, \\texttt{USE}."
            ),
        },
        "locating_check": {
            "NAVIGATE_TO_PRIMITIVE": True,
            "NAVIGATE_TO_WORKSHOP": True,
            "NAVIGATE_TO_OBSTACLE": True,
            "GET": True,
            "MOVE": False,
            "PICK": True,
            "CRAFT": True,
            "USE_TOOL": True,
        },
        "midrule_before": "PICK",
    },
    13: {
        "locating": {
            "PICKUP",
            "NAVIGATE_TO_RESOURCE",
            "CRAFT_ITEM",
            "USE_TOOL_ON_OBSTACLE",
            "BUILD",
        },
        "explicit": {"MOVE", "TURN"},
        "section_label": "app:program-form-run13",
        "decision_rule": [
            "A statement is a \\emph{locating} terminal if it determines the cell it acts on",
            "at execution time rather than from the program text. In the Run~13 DSL~1",
            "grammar the locating terminals are",
            "\\texttt{NAVIGATE\\_TO\\_RESOURCE}, \\texttt{CRAFT\\_ITEM},",
            "\\texttt{USE\\_TOOL\\_ON\\_OBSTACLE}, \\texttt{BUILD} and \\texttt{PICKUP}",
            "(Table~\\ref{tab:app-run13-terminals}).",
            "\\texttt{CRAFT} and \\texttt{USE\\_TOOL} do not search: the former only inspects",
            "the four adjacent cells, and the latter emits a single \\texttt{USE}.",
            "A statement is an \\emph{explicit movement} terminal if it is \\texttt{MOVE}",
            "or \\texttt{TURN}; both name a cardinal direction and encode one layout.",
        ],
        "terminal_caption_extra": (
            "Rows where the CFG description and the implemented behaviour disagree "
            "are marked in the rightmost column."
        ),
        "terminal_label": "tab:app-run13-terminals",
        "form_counts_label": "tab:app-form-counts-run13",
        "form_by_task_label": "tab:app-form-by-task-run13",
        "mixed_label": "tab:app-mixed-programs-run13",
        "locating_label": "tab:app-locating-programs-run13",
        "explicit_label": "tab:app-explicit-programs-run13",
        "include_explicit_column": True,
        "terminal_order": [
            "NAVIGATE_TO_RESOURCE",
            "CRAFT_ITEM",
            "USE_TOOL_ON_OBSTACLE",
            "BUILD",
            "MOVE",
            "TURN",
            "PICKUP",
            "CRAFT",
            "USE_TOOL",
        ],
        "impl_notes": {
            "NAVIGATE_TO_RESOURCE": (
                "Breadth-first search to a free cell \\emph{adjacent} to the named "
                "resource, then face it. Does not step onto the resource cell."
            ),
            "CRAFT_ITEM": (
                "Breadth-first search to a free cell adjacent to the workshop required "
                "by the named item, face it, then \\texttt{USE}."
            ),
            "USE_TOOL_ON_OBSTACLE": (
                "Requires the named tool in inventory. Breadth-first search to a free "
                "neighbour of the nearest named obstacle, face it, then \\texttt{USE}."
            ),
            "BUILD": (
                "Recursive gather-and-craft planner: collects missing primitives, "
                "navigates, and crafts intermediates at workshops. Not a place-here "
                "primitive."
            ),
            "MOVE": "As described. One cell in the named cardinal direction.",
            "TURN": (
                "Faces the named direction. Because a directional action also steps "
                "forward, the implementation may emit a reverse step to stay put."
            ),
            "PICKUP": (
                "Greedy Manhattan walk to the nearest named resource, step to an "
                "adjacent free cell, face it, then \\texttt{USE}. Ignores the current cell."
            ),
            "CRAFT": (
                "If the recipe workshop is in the four-neighbourhood, face it and "
                "\\texttt{USE}; otherwise emit \\texttt{USE} in place. Does not pathfind."
            ),
            "USE_TOOL": (
                "Ignores both arguments and emits a single \\texttt{USE}. The caller "
                "must already be facing the obstacle."
            ),
        },
        "locating_check": {
            "NAVIGATE_TO_RESOURCE": True,
            "CRAFT_ITEM": True,
            "USE_TOOL_ON_OBSTACLE": True,
            "BUILD": True,
            "MOVE": False,
            "TURN": False,
            "PICKUP": True,
            "CRAFT": False,
            "USE_TOOL": False,
        },
        "midrule_before": "PICKUP",
    },
}


def classify(prog: str, locating: set[str], explicit: set[str]) -> str:
    terms = {stmt_terminal(s) for s in prog.split(";") if s.strip()}
    has_loc = bool(terms & locating)
    has_exp = bool(terms & explicit)
    if has_loc and not has_exp:
        return "locating"
    if has_exp and not has_loc:
        return "explicit"
    if has_loc and has_exp:
        return "mixed"
    return "other"


def norm_prog(prog: str) -> str:
    return re.sub(r"\s*;\s*", ";", prog.strip())


def stmt_terminal(stmt: str) -> str:
    return stmt.split("(")[0].strip().upper()


def program_length(prog: str) -> int:
    return len([s for s in prog.split(";") if s.strip()])


def latex_task(task: str) -> str:
    return f"\\texttt{{{task}}}"


def latex_program_inline(prog: str) -> str:
    parts = [p.strip() for p in prog.split(";") if p.strip()]
    body = "\\allowbreak ".join(latex_stmt(s) for s in parts)
    return f"\\texttt{{{body}}}"


def latex_program_block(prog: str) -> str:
    parts = [p.strip() for p in prog.split(";") if p.strip()]
    body = ";\n".join(latex_stmt(s) for s in parts)
    return f"\\raggedright\\texttt{{{body}}}\\arraybackslash"


def latex_stmt(stmt: str) -> str:
    name, rest = stmt.split("(", 1)
    rest = rest[:-1] if rest.endswith(")") else rest
    latex_name = name.replace("_", r"\_")
    return f"{latex_name}({rest})"


def fmt_mean(values: list[float] | None) -> str:
    if not values:
        return "---"
    return f"{sum(values) / len(values):.2f}"


def fmt_seeds(seeds: list[int]) -> str:
    if len(seeds) == 1:
        return str(seeds[0])
    groups: list[str] = []
    start = prev = seeds[0]
    for s in seeds[1:]:
        if s == prev + 5:
            prev = s
            continue
        groups.append(f"{start}" if start == prev else f"{start}, {prev}")
        start = prev = s
    groups.append(f"{start}" if start == prev else f"{start}, {prev}")
    return ", ".join(groups)


def load_cross_seed_g(coverage_path: Path, label: str) -> dict[tuple[str, str], int]:
    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    run = next(r for r in payload["runs"] if r["label"] == label)
    out: dict[tuple[str, str], int] = {}
    for task, info in run["tasks"].items():
        for entry in info["programs"]:
            out[(task, entry["program"])] = entry["coverage"]
    return out


def lookup_g(prog_g: dict[tuple[str, str], int], task: str, prog: str) -> int:
    prog = prog.strip()
    if (task, prog) in prog_g:
        return prog_g[(task, prog)]
    target = norm_prog(prog)
    for (t, p), g in prog_g.items():
        if t == task and norm_prog(p) == target:
            return g
    raise KeyError(f"no cross-seed coverage for {task}: {prog[:80]!r}")


def load_instances(
    experiment_dir: Path, locating: set[str], explicit: set[str]
) -> list[dict]:
    instances: list[dict] = []
    base = experiment_dir / "results_tracking" / "dsl1" / "tasks"
    for task_dir in sorted(glob.glob(str(base / "*"))):
        path = os.path.join(task_dir, "program_synthesis_seed_outcomes.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                entry = json.loads(line)
                if not entry.get("solved") or not entry.get("solved_program"):
                    continue
                prog = entry["solved_program"].strip()
                instances.append(
                    {
                        "task": entry["task"],
                        "seed": int(entry["seed"]),
                        "program": prog,
                        "form": classify(prog, locating, explicit),
                        "L": program_length(prog),
                    }
                )
    return instances


def load_task_order(experiment_dir: Path) -> list[str]:
    state_path = experiment_dir / "pipeline_state.txt"
    for line in state_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("tasks="):
            return json.loads(line.split("=", 1)[1])
    raise FileNotFoundError(f"tasks= not found in {state_path}")


def aggregate_unique(
    instances: list[dict],
    prog_g: dict[tuple[str, str], int],
    locating: set[str],
    explicit: set[str],
) -> list[dict]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    canonical: dict[tuple[str, str], str] = {}
    for inst in instances:
        key = (inst["task"], norm_prog(inst["program"]))
        grouped[key].append(inst["seed"])
        canonical[key] = inst["program"]
    rows: list[dict] = []
    for (task, nprog), seeds in grouped.items():
        prog = canonical[(task, nprog)]
        rows.append(
            {
                "task": task,
                "program": prog,
                "seeds": sorted(seeds),
                "n": len(seeds),
                "form": classify(prog, locating, explicit),
                "L": program_length(prog),
                "g": lookup_g(prog_g, task, prog),
            }
        )
    return rows


def render_longtable(
    caption: str,
    label: str,
    rows: list[dict],
    *,
    include_seeds: bool = False,
) -> str:
    if not rows:
        return ""
    if include_seeds:
        header = "Task & Seeds & Program & $L$ & $g$ & $n$ \\\\\n"
        colspec = "@{}l >{\\raggedright\\arraybackslash}p{0.10\\linewidth} >{\\raggedright\\arraybackslash}p{0.47\\linewidth} r r r@{}"
        body_lines = []
        for row in rows:
            seeds = fmt_seeds(row["seeds"])
            body_lines.append(
                f"{latex_task(row['task'])} & {seeds} & {latex_program_inline(row['program'])} & "
                f"{row['L']} & {row['g']} & {row['n']} \\\\"
            )
    else:
        header = "Task & Program & $L$ & $g$ & $n$ \\\\\n"
        colspec = "@{}l >{\\raggedright\\arraybackslash}p{0.62\\linewidth} r r r@{}"
        body_lines = []
        for row in rows:
            body_lines.append(
                f"{latex_task(row['task'])} & {latex_program_block(row['program'])} & "
                f"{row['L']} & {row['g']} & {row['n']} \\\\"
            )

    lines = [
        "{\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{longtable}}{{{colspec}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}\\\\",
        "\\toprule",
        header + "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{6}{c}{\\emph{Continued from previous page}} \\\\",
        "\\toprule",
        header + "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{6}{r}{\\emph{Continued on next page}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for i, body in enumerate(body_lines):
        if i:
            lines.append("\\addlinespace")
        lines.append(body)
    lines.extend(["\\end{longtable}", "}"])
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def weighted_mean(rows: list[dict], field: str) -> str:
    if not rows:
        return "---"
    total_n = sum(r["n"] for r in rows)
    if total_n == 0:
        return "---"
    return f"{sum(r[field] * r['n'] for r in rows) / total_n:.2f}"


def generate_appendix(
    *,
    pipeline_run: int,
    experiment_dir: Path,
    coverage_path: Path,
    coverage_label: str,
    output_path: Path,
) -> dict:
    if pipeline_run not in PROFILES:
        raise KeyError(f"no program-form profile for pipeline run {pipeline_run}")
    profile = PROFILES[pipeline_run]
    locating = profile["locating"]
    explicit = profile["explicit"]

    prog_g = load_cross_seed_g(coverage_path, coverage_label)
    instances = load_instances(experiment_dir, locating, explicit)
    for inst in instances:
        inst["g"] = lookup_g(prog_g, inst["task"], inst["program"])

    unique_rows = aggregate_unique(instances, prog_g, locating, explicit)
    tasks = load_task_order(experiment_dir)

    inst_by_form = Counter(i["form"] for i in instances)
    uniq_by_form = Counter(r["form"] for r in unique_rows)

    per_task_rows = []
    for task in tasks:
        task_inst = [i for i in instances if i["task"] == task]
        loc = [i for i in task_inst if i["form"] == "locating"]
        mix = [i for i in task_inst if i["form"] == "mixed"]
        exp = [i for i in task_inst if i["form"] == "explicit"]
        per_task_rows.append(
            {
                "task": task,
                "loc_n": len(loc),
                "mix_n": len(mix),
                "exp_n": len(exp),
                "loc_g": fmt_mean([i["g"] for i in loc]),
                "mix_g": fmt_mean([i["g"] for i in mix]),
                "exp_g": fmt_mean([i["g"] for i in exp]),
            }
        )

    total_loc = [i for i in instances if i["form"] == "locating"]
    total_mix = [i for i in instances if i["form"] == "mixed"]
    total_exp = [i for i in instances if i["form"] == "explicit"]

    mixed_rows = sorted(
        [r for r in unique_rows if r["form"] == "mixed"],
        key=lambda r: (r["task"], -r["g"], r["program"]),
    )
    locating_rows = sorted(
        [r for r in unique_rows if r["form"] == "locating"],
        key=lambda r: (r["task"], -r["n"], -r["g"], r["program"]),
    )
    explicit_rows = sorted(
        [r for r in unique_rows if r["form"] == "explicit"],
        key=lambda r: (r["task"], -r["n"], -r["g"], r["program"]),
    )

    mixed_inst = [i for i in instances if i["form"] == "mixed"]
    cfg = json.loads((experiment_dir / "cfg" / "cfg_output_1.json").read_text(encoding="utf-8"))

    n_loc = inst_by_form.get("locating", 0)
    n_mix = inst_by_form.get("mixed", 0)
    n_exp = inst_by_form.get("explicit", 0)
    class_sentence = (
        f"Of the {len(instances)} stored programs, {n_loc} are locating-only, "
        f"{n_mix} are mixed, and {n_exp} are explicit-only."
    )

    lines: list[str] = [
        f"% Appendix: program-form classification, Pipeline Run {pipeline_run}, DSL 1.",
        "% Requires: booktabs, longtable, array.",
        "",
        f"\\section{{Program-form classification (Pipeline Run~{pipeline_run}, DSL~1)}}",
        f"\\label{{{profile['section_label']}}}",
        "",
        "This appendix records the classification used in",
        f"Table~\\ref{{tab:generalization-by-form-run{pipeline_run}}}.",
        f"The grammar is the Run~{pipeline_run} DSL~1 CFG.",
        "Each observation is one synthesized program $p_{t,s}$ stored for a",
        "$(t,s)$ pair; identical strings that won several seeds are listed once",
        "with multiplicity $n$.",
        "Length $L$ is the number of semicolon-separated statements.",
        "Coverage $g$ is the number of the ten test seeds $\\{0,5,\\ldots,45\\}$ on",
        "which that string succeeds when executed unchanged.",
        "",
        "\\subsection{Decision rule}",
        "",
        *profile["decision_rule"],
        "",
        "Let $N$ be the set of locating terminals used in the program",
        "and $M$ the set of explicit movement terminals.",
        "\\begin{itemize}",
        "  \\item \\textbf{Locating} if $N\\neq\\emptyset$ and $M=\\emptyset$.",
        "  \\item \\textbf{Explicit} if $M\\neq\\emptyset$ and $N=\\emptyset$.",
        "  \\item \\textbf{Mixed} if $N\\neq\\emptyset$ and $M\\neq\\emptyset$.",
        "\\end{itemize}",
        "",
        class_sentence,
        "",
        f"\\subsection{{Terminals in the Run~{pipeline_run} DSL~1 grammar}}",
        "",
        "Roles follow the implementations FunSearch produced, not only the descriptions the",
        "CFG supplies.",
        profile["terminal_caption_extra"],
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption[Run~{pipeline_run} DSL~1 terminals: description against implementation]"
        f"{{DSL~1 terminals for Pipeline Run~{pipeline_run}. "
        "\\emph{Loc.} marks a terminal that determines its target at execution time. "
        + profile["terminal_caption_extra"]
        + "}",
        f"\\label{{{profile['terminal_label']}}}",
        "\\begin{tabular}{@{}l c >{\\raggedright\\arraybackslash}p{0.27\\linewidth} >{\\raggedright\\arraybackslash}p{0.40\\linewidth}@{}}",
        "\\toprule",
        "Terminal & Loc. & CFG description & As implemented \\\\",
        "\\midrule",
    ]

    for name in profile["terminal_order"]:
        if name == profile.get("midrule_before"):
            lines.append("\\midrule")
        loc_mark = "\\checkmark" if profile["locating_check"][name] else ""
        desc = latex_escape(str(cfg["terminals"][name]))
        impl = profile["impl_notes"][name]
        latex_name = name.replace("_", r"\_")
        lines.append(
            "\\texttt{" + latex_name + "} & " + loc_mark + " & " + desc + " & " + impl + " \\\\"
        )
        if name != profile["terminal_order"][-1] and name != profile.get("midrule_before"):
            nxt = profile["terminal_order"][profile["terminal_order"].index(name) + 1]
            if nxt != profile.get("midrule_before"):
                lines.append("\\addlinespace")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\subsection{Instance counts}",
            "",
            f"The main-text table uses $(t,s)$ instances ($n={len(instances)}$), not unique strings ",
            f"($n={len(unique_rows)}$).",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Program form, Pipeline Run~{pipeline_run}, DSL~1. Instances weight each stored "
            "$(t,s)$ program once; unique strings collapse identical text within a task.}",
            f"\\label{{{profile['form_counts_label']}}}",
            "\\begin{tabular}{@{}lcc@{}}",
            "\\toprule",
            "Program form & Unique strings & $(t,s)$ instances \\\\",
            "\\midrule",
            f"Locating only & {uniq_by_form.get('locating', 0)} & {inst_by_form.get('locating', 0)} \\\\",
            f"Mixed & {uniq_by_form.get('mixed', 0)} & {inst_by_form.get('mixed', 0)} \\\\",
            f"Explicit only & {uniq_by_form.get('explicit', 0)} & {inst_by_form.get('explicit', 0)} \\\\",
            "\\midrule",
            f"Total & {len(unique_rows)} & {len(instances)} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Generalization by program form, Pipeline Run~{pipeline_run}, DSL~1. "
            "Mean $L$ and mean $g$ are instance-weighted.}",
            f"\\label{{tab:generalization-by-form-run{pipeline_run}}}",
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Program form & Unique strings & Instances & Mean $L$ & Mean $g$ \\\\",
            "\\midrule",
        ]
    )
    for form_name, key in (
        ("Locating only", "locating"),
        ("Mixed", "mixed"),
        ("Explicit only", "explicit"),
    ):
        form_rows = [r for r in unique_rows if r["form"] == key]
        form_inst = [i for i in instances if i["form"] == key]
        lines.append(
            f"{form_name} & {len(form_rows)} & {len(form_inst)} & "
            f"{fmt_mean([i['L'] for i in form_inst])} & "
            f"{fmt_mean([i['g'] for i in form_inst])} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            f"Total & {len(unique_rows)} & {len(instances)} & "
            f"{fmt_mean([i['L'] for i in instances])} & "
            f"{fmt_mean([i['g'] for i in instances])} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\subsection{Per-task breakdown}",
            "",
            "Counts below are $(t,s)$ instances.",
            "A blank mean is a form that did not occur on that task.",
            "",
        ]
    )

    if profile["include_explicit_column"]:
        lines.extend(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\small",
                "\\setlength{\\tabcolsep}{3pt}",
                f"\\caption{{Per-task form counts and instance-weighted mean $g$, Pipeline "
                f"Run~{pipeline_run}, DSL~1. $g$ is the number of the ten test seeds the program solves when "
                "executed unchanged.}",
                f"\\label{{{profile['form_by_task_label']}}}",
                "\\begin{tabular}{@{}l rrr rrr@{}}",
                "\\toprule",
                " & \\multicolumn{3}{c}{Instances} & \\multicolumn{3}{c}{Mean $g$} \\\\",
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                "Task & Loc. & Mix. & Exp. & Loc. & Mix. & Exp. \\\\",
                "\\midrule",
            ]
        )
        for row in per_task_rows:
            lines.append(
                f"{latex_task(row['task'])} & {row['loc_n']} & {row['mix_n']} & {row['exp_n']} & "
                f"{row['loc_g']} & {row['mix_g']} & {row['exp_g']} \\\\"
            )
        lines.extend(
            [
                "\\midrule",
                f"Total & {len(total_loc)} & {len(total_mix)} & {len(total_exp)} & "
                f"{fmt_mean([i['g'] for i in total_loc])} & "
                f"{fmt_mean([i['g'] for i in total_mix])} & "
                f"{fmt_mean([i['g'] for i in total_exp])} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\small",
                f"\\caption{{Per-task form counts and instance-weighted mean $g$, Pipeline "
                f"Run~{pipeline_run}, DSL~1. $g$ is the number of the ten test seeds the program solves when "
                "executed unchanged.}",
                f"\\label{{{profile['form_by_task_label']}}}",
                "\\begin{tabular}{@{}l rr rr@{}}",
                "\\toprule",
                " & \\multicolumn{2}{c}{Instances} & \\multicolumn{2}{c}{Mean $g$} \\\\",
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
                "Task & Loc. & Mix. & Loc. & Mix. \\\\",
                "\\midrule",
            ]
        )
        for row in per_task_rows:
            lines.append(
                f"{latex_task(row['task'])} & {row['loc_n']} & {row['mix_n']} & "
                f"{row['loc_g']} & {row['mix_g']} \\\\"
            )
        lines.extend(
            [
                "\\midrule",
                f"Total & {len(total_loc)} & {len(total_mix)} & "
                f"{fmt_mean([i['g'] for i in total_loc])} & "
                f"{fmt_mean([i['g'] for i in total_mix])} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ]
        )

    if mixed_inst:
        mix_tasks = Counter(i["task"] for i in mixed_inst)
        top_tasks = ", ".join(
            f"\\texttt{{{t}}}" for t, _ in mix_tasks.most_common(3)
        )
        lines.extend(
            [
                f"Mixed programs concentrate on {top_tasks} "
                f"({sum(c for _, c in mix_tasks.most_common(3))} of {len(mixed_inst)} mixed instances).",
                "",
            ]
        )

    if mixed_inst:
        lines.extend(
            [
                "\\subsection{Mixed programs}",
                "",
                "Every program containing both a locating terminal and an explicit movement terminal.",
                f"Coverage within this class ranges from $g = {min(i['g'] for i in mixed_inst)}$ "
                f"to $g = {max(i['g'] for i in mixed_inst)}$.",
                "",
                render_longtable(
                    f"Every mixed program, Pipeline Run~{pipeline_run}, DSL~1.",
                    profile["mixed_label"],
                    mixed_rows,
                    include_seeds=True,
                ),
                "",
            ]
        )

    if locating_rows:
        lines.extend(
            [
                "\\subsection{Locating programs}",
                "",
                "Unique strings only; $n$ is how many of the ten test seeds synthesis stored",
                "this string for. Mean length and mean $g$ in the main-text table weight by $n$.",
                "",
                render_longtable(
                    f"Every unique locating program, Pipeline Run~{pipeline_run}, DSL~1.",
                    profile["locating_label"],
                    locating_rows,
                    include_seeds=False,
                ),
                "",
            ]
        )

    if explicit_rows:
        lines.extend(
            [
                "\\subsection{Explicit programs}",
                "",
                "Unique strings that use \\texttt{MOVE}/\\texttt{TURN} and no locating terminal.",
                "",
                render_longtable(
                    f"Every unique explicit program, Pipeline Run~{pipeline_run}, DSL~1.",
                    profile["explicit_label"],
                    explicit_rows,
                    include_seeds=True,
                ),
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "instances": len(instances),
        "unique": len(unique_rows),
        "inst_by_form": dict(inst_by_form),
        "uniq_by_form": dict(uniq_by_form),
        "mean_g_by_form": {
            "locating": fmt_mean([i["g"] for i in total_loc]),
            "mixed": fmt_mean([i["g"] for i in total_mix]),
            "explicit": fmt_mean([i["g"] for i in total_exp]),
        },
        "mean_L_by_form": {
            "locating": fmt_mean([i["L"] for i in total_loc]),
            "mixed": fmt_mean([i["L"] for i in total_mix]),
            "explicit": fmt_mean([i["L"] for i in total_exp]),
        },
        "per_task": per_task_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-run", type=int, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dsl1_cross_seed_coverage.json",
    )
    parser.add_argument("--coverage-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    summary = generate_appendix(
        pipeline_run=args.pipeline_run,
        experiment_dir=args.experiment_dir,
        coverage_path=args.coverage_json,
        coverage_label=args.coverage_label,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2))
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
