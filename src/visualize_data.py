"""
Run this to generate a "system card" PNG for every preprocessed person.

For each person under data/preprocessed/, we render one figure with:
  i)   4 stacked biosignal panels (BVP, EDA, HR, TEMP) on a continuous, phase-stitched time axis
  ii)  alternating phase shading + phase labels along the top
  iii) a 14-row heatmap summarizing self-report per phase:
        difficulty / PANAS pos Σ / PANAS neg Σ / 5 positive items / 6 negative items

Output: report/figures/person_{N}_data_card.png
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = REPO_ROOT / "data" / "preprocessed"
BIOSIGNAL_DIR = PREPROCESSED_DIR / "biosignals"
RESPONSES_DIR = PREPROCESSED_DIR / "responses"
FIGURES_DIR = REPO_ROOT / "report" / "figures"

SIGNALS = [
    ("BVP",  "BVP (ppg)", "tab:red"),
    ("EDA",  "EDA (µS)",  "tab:blue"),
    ("HR",   "HR (bpm)",  "tab:green"),
    ("TEMP", "TEMP (°C)", "tab:orange"),
]

PANAS_POS = ["alert", "inspired", "determined", "attentive", "active"]
PANAS_NEG = ["upset", "hostile", "ashamed", "nervous", "afraid", "frustrated"]

ROWS = [
    ("difficulty", "difficulty", "Purples", 1,  5),
    ("PA",         "pos Σ",      "Greens",  5, 25),
    ("NA",         "neg Σ",      "Reds",    6, 30),
    ("alert",      "alert",      "Greens",  1,  5),
    ("inspired",   "inspired",   "Greens",  1,  5),
    ("determined", "determined", "Greens",  1,  5),
    ("attentive",  "attentive",  "Greens",  1,  5),
    ("active",     "active",     "Greens",  1,  5),
    ("upset",      "upset",      "Reds",    1,  5),
    ("hostile",    "hostile",    "Reds",    1,  5),
    ("ashamed",    "ashamed",    "Reds",    1,  5),
    ("nervous",    "nervous",    "Reds",    1,  5),
    ("afraid",     "afraid",     "Reds",    1,  5),
    ("frustrated", "frustrated", "Reds",    1,  5),
]

GROUP_BREAKS = [1, 3, 8]


def load_person(biosignal_path: Path, response_path: Path):
    df = pd.read_csv(biosignal_path)
    df["time"] = pd.to_datetime(df["time"])
    df["phase_key"] = df["round"] + "/" + df["phase"]
    phase_order = df["phase_key"].drop_duplicates().tolist()

    elapsed_parts = []
    phase_bounds = {}
    running_offset = 0.0
    for key in phase_order:
        chunk = df.loc[df["phase_key"] == key, "time"]
        secs = (chunk - chunk.iloc[0]).dt.total_seconds()
        start = running_offset
        end = running_offset + secs.iloc[-1]
        phase_bounds[key] = (start, end)
        elapsed_parts.append(secs + running_offset)
        running_offset = end
    df["t"] = pd.concat(elapsed_parts).sort_index()

    resp = pd.read_csv(response_path)
    resp["phase_key"] = resp["round"] + "/" + resp["phase"]
    resp["PA"] = resp[PANAS_POS].sum(axis=1, min_count=len(PANAS_POS))
    resp["NA"] = resp[PANAS_NEG].sum(axis=1, min_count=len(PANAS_NEG))
    resp = resp.set_index("phase_key")

    return df, resp, phase_order, phase_bounds


def plot_system_card(person_key: str, df, resp, phase_order, phase_bounds):
    fig, axes = plt.subplots(
        len(SIGNALS) + 1, 1,
        sharex=True,
        figsize=(14, 16),
        gridspec_kw={"height_ratios": [3, 3, 3, 3, 7]},
    )
    fig.suptitle(f"{person_key} biosignals + self-report across rounds & phases", fontsize=14)

    sig_axes = axes[:4]
    for ax, (col, label, color) in zip(sig_axes, SIGNALS):
        ax.plot(df["t"], df[col], color=color, linewidth=0.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    for i, key in enumerate(phase_order):
        start, end = phase_bounds[key]
        if i % 2 == 0:
            for ax in sig_axes:
                ax.axvspan(start, end, color="grey", alpha=0.06, zorder=0)
        for ax in sig_axes:
            ax.axvline(start, color="black", alpha=0.25, linewidth=0.5)

    top = sig_axes[0]
    for key in phase_order:
        start, end = phase_bounds[key]
        top.text(
            (start + end) / 2, top.get_ylim()[1], key,
            ha="center", va="bottom", fontsize=8,
        )

    heat_ax = axes[-1]
    edges_x = [phase_bounds[k][0] for k in phase_order] + [phase_bounds[phase_order[-1]][1]]

    for row_idx, (col, label, cmap, vmin, vmax) in enumerate(ROWS):
        values = np.array([[resp[col].get(k, np.nan) for k in phase_order]], dtype=float)
        heat_ax.pcolormesh(
            edges_x,
            [row_idx, row_idx + 1],
            values,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="flat",
            edgecolors="white", linewidth=0.5,
        )
        for j, key in enumerate(phase_order):
            v = values[0, j]
            cx = (edges_x[j] + edges_x[j + 1]) / 2
            cy = row_idx + 0.5
            text = "—" if np.isnan(v) else (f"{int(v)}" if float(v).is_integer() else f"{v:.1f}")
            heat_ax.text(cx, cy, text, ha="center", va="center", fontsize=8, color="black")

    for y in GROUP_BREAKS:
        heat_ax.axhline(y, color="black", linewidth=1.0)

    heat_ax.set_yticks([i + 0.5 for i in range(len(ROWS))])
    heat_ax.set_yticklabels([r[1] for r in ROWS])
    heat_ax.set_ylim(len(ROWS), 0)
    heat_ax.set_xlabel("seconds from session start")

    for key in phase_order:
        heat_ax.axvline(phase_bounds[key][0], color="black", alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    return fig


PERSON_NUMBER_RE = re.compile(r"^Person(\d+)_")


def render_all(
    biosignal_dir: Path = BIOSIGNAL_DIR,
    responses_dir: Path = RESPONSES_DIR,
    out_dir: Path = FIGURES_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    biosignal_files = sorted(biosignal_dir.glob("Person*.csv"))
    rendered = 0
    for bio_path in biosignal_files:
        person_key = bio_path.stem
        match = PERSON_NUMBER_RE.match(person_key)
        if match is None:
            print(f"Skipping {person_key} (unexpected filename)")
            continue
        n = int(match.group(1))

        resp_path = responses_dir / f"{person_key}_responses.csv"
        if not resp_path.exists():
            print(f"Skipping {person_key} (no response file at {resp_path})")
            continue

        df, resp, phase_order, phase_bounds = load_person(bio_path, resp_path)
        fig = plot_system_card(person_key, df, resp, phase_order, phase_bounds)
        out_path = out_dir / f"person_{n}_data_card.png"
        fig.savefig(out_path)
        plt.close(fig)
        rendered += 1
        print(f"Rendered {person_key} -> {out_path.relative_to(REPO_ROOT)}")

    print(f"Done! Wrote {rendered} system card figures")


if __name__ == "__main__":
    render_all()
