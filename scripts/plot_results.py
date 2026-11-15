"""Turn results/benchmark.csv into the graphs for the report.

    python scripts/plot_results.py
    python scripts/plot_results.py --input results/benchmark.csv --outdir plots --physical-cores 4

Produces four PNG files:
    execution_time.png  time vs threads, largest image, one line per filter
    speedup.png         speedup vs threads, with the ideal (linear) line
    efficiency.png      efficiency vs threads
    scalability.png     speedup vs threads for every image size, one panel per filter
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt

# Colour and marker stay attached to the same filter / size in every graph.
# Marker shapes differ too, so the lines can still be told apart when the
# report is printed in black and white.
SERIES_STYLES = [
    {"color": "#2a78d6", "marker": "o"},  # blue
    {"color": "#eb6834", "marker": "s"},  # orange
    {"color": "#1baf7a", "marker": "^"},  # aqua
    {"color": "#eda100", "marker": "D"},  # yellow
    {"color": "#e87ba4", "marker": "v"},  # magenta
]
FILTER_ORDER = ["edge", "blur", "sharpen"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8983"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": GRID,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "legend.frameon": False,
})


def load(path, schedule):
    """Returns (rows, sequential) where
    rows[(filter, megapixels)]       = sorted list of (threads, best_ms, speedup, efficiency)
    sequential[(filter, megapixels)] = best_ms of the sequential baseline
    """
    rows = defaultdict(list)
    sequential = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["filter"], float(r["megapixels"]))
            if r["mode"] == "sequential":
                sequential[key] = float(r["best_ms"])
            elif r["schedule"] == schedule:
                rows[key].append((int(r["threads"]), float(r["best_ms"]),
                                  float(r["speedup"]), float(r["efficiency"])))
    for series in rows.values():
        series.sort()
    return rows, sequential


def filter_label(name):
    return {"edge": "Edge detection (3x3)", "blur": "Gaussian blur (5x5)",
            "sharpen": "Sharpen (3x3)"}.get(name, name)


def size_label(megapixels):
    return f"{megapixels:g} MP"


def setup_thread_axis(ax, threads, physical_cores):
    ax.set_xticks(threads)
    ax.set_xlabel("OpenMP threads")
    ax.set_xlim(min(threads) - 0.4, max(threads) + 0.4)
    ax.grid(axis="x", visible=False)
    if physical_cores and min(threads) < physical_cores < max(threads):
        # Past this point extra threads share a physical core (hyper-threading),
        # which is where the curves are expected to bend.
        ax.axvline(physical_cores, color=REFERENCE, linewidth=1, linestyle=":")
        ax.annotate(f"{physical_cores} physical cores", xy=(physical_cores, 1),
                    xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", va="top", ha="left",
                    fontsize=8.5, color=INK_SECONDARY)


def titled(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=INK_SECONDARY, va="bottom")


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="results/benchmark.csv")
    parser.add_argument("--outdir", default="plots")
    parser.add_argument("--schedule", default="static")
    parser.add_argument("--physical-cores", type=int, default=0,
                        help="mark this thread count on the x axis")
    parser.add_argument("--machine", default="",
                        help="short machine description added to the subtitles")
    args = parser.parse_args()

    rows, sequential = load(args.input, args.schedule)
    if not rows:
        raise SystemExit(f"no '{args.schedule}' rows found in {args.input}")
    os.makedirs(args.outdir, exist_ok=True)

    filters = [f for f in FILTER_ORDER if any(k[0] == f for k in rows)]
    filters += sorted({k[0] for k in rows} - set(filters))
    sizes = sorted({k[1] for k in rows})
    largest = sizes[-1]
    threads = [t for t, *_ in rows[(filters[0], largest)]]
    style = {name: SERIES_STYLES[i] for i, name in enumerate(filters)}

    context = f"{size_label(largest)} image, schedule({args.schedule})"
    if args.machine:
        context += f", {args.machine}"

    # ---- 1. execution time --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name in filters:
        series = rows[(name, largest)]
        ax.plot([t for t, *_ in series], [ms / 1000 for _, ms, *_ in series],
                label=filter_label(name), **style[name])
    ax.set_ylabel("Execution time (seconds)")
    ax.set_ylim(bottom=0)
    setup_thread_axis(ax, threads, args.physical_cores)
    titled(ax, "Execution time falls as threads are added", context)
    ax.legend(loc="center right")
    save(fig, args.outdir, "execution_time.png")

    # ---- 2. speedup ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(threads, threads, color=REFERENCE, linewidth=1.3, linestyle="--",
            label="Ideal (linear) speedup")
    for name in filters:
        series = rows[(name, largest)]
        ax.plot([t for t, *_ in series], [s for _, _, s, _ in series],
                label=filter_label(name), **style[name])
    ax.set_ylabel("Speedup  (sequential time / parallel time)")
    ax.set_ylim(0, max(threads) + 0.5)
    setup_thread_axis(ax, threads, args.physical_cores)
    titled(ax, "Speedup over the sequential version", context)
    ax.legend(loc="upper left")
    save(fig, args.outdir, "speedup.png")

    # ---- 3. efficiency ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(100, color=REFERENCE, linewidth=1.3, linestyle="--",
               label="Ideal (100%)")
    top = 100
    for name in filters:
        series = rows[(name, largest)]
        values = [100 * e for *_, e in series]
        top = max(top, max(values))
        ax.plot([t for t, *_ in series], values, label=filter_label(name), **style[name])
    ax.set_ylabel("Parallel efficiency  (speedup / threads)")
    ax.set_ylim(0, top * 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    setup_thread_axis(ax, threads, args.physical_cores)
    titled(ax, "Each added thread contributes less", context)
    ax.legend(loc="lower left")
    save(fig, args.outdir, "efficiency.png")

    # ---- 4. scalability: one panel per filter, one line per image size --------
    size_style = {mp: SERIES_STYLES[i] for i, mp in enumerate(sizes)}
    fig, axes = plt.subplots(1, len(filters), figsize=(4.1 * len(filters), 4.2),
                             sharey=True, squeeze=False)
    for ax, name in zip(axes[0], filters):
        ax.plot(threads, threads, color=REFERENCE, linewidth=1.3, linestyle="--",
                label="Ideal")
        for mp in sizes:
            series = rows.get((name, mp), [])
            ax.plot([t for t, *_ in series], [s for _, _, s, _ in series],
                    label=size_label(mp), **size_style[mp])
        ax.set_title(filter_label(name), loc="left", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, max(threads) + 0.5)
        setup_thread_axis(ax, threads, args.physical_cores)
    axes[0][0].set_ylabel("Speedup")
    axes[0][0].legend(loc="upper left", title="Image size", alignment="left")
    subtitle = f"schedule({args.schedule})" + (f", {args.machine}" if args.machine else "")
    fig.suptitle("Speedup by image size", x=0.01, ha="left", fontsize=12.5,
                 fontweight="bold")
    fig.text(0.01, 0.905, subtitle, fontsize=9.5, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args.outdir, "scalability.png")


if __name__ == "__main__":
    main()
