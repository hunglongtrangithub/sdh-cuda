from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14

plot_dir = Path(__file__).parent / "plots"
plot_dir.mkdir(exist_ok=True, parents=True)
print(f"Saving plots to: {plot_dir.absolute()}")


def load_data(filepath):
    """Load and preprocess the experiment data using Polars"""
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pl.read_csv(filepath)

    # Convert block_size to integer and sort it numerically
    df = df.with_columns(pl.col("block_size").cast(pl.Int32).sort())

    # Ensure other numerical columns are properly typed
    df = df.with_columns(
        pl.col("num_atoms").cast(pl.Int32),
        pl.col("resolution").cast(pl.Float64),
        pl.col("time_ms").cast(pl.Float64),
    )

    return df.sort(["num_atoms", "resolution", "block_size"])


def plot_time_vs_atoms(df):
    """Plot execution time vs number of atoms for each algorithm"""
    plt.figure(figsize=(14, 8))

    # Get algorithms in consistent order
    algorithms = sorted(df["algorithm"].unique().to_list())
    markers = ["o", "s", "D", "^"]

    for alg, marker in zip(algorithms, markers):
        alg_df = df.filter(pl.col("algorithm") == alg)

        # Group and sort by num_atoms
        grouped = (
            alg_df.group_by("num_atoms").agg(pl.col("time_ms").mean()).sort("num_atoms")
        )

        plt.plot(
            grouped["num_atoms"].to_list(),
            grouped["time_ms"].to_list(),
            marker=marker,
            markersize=8,
            linewidth=2,
            label=f"{alg}",
        )

    plt.title("Execution Time vs Number of Atoms")
    plt.xlabel("Number of Atoms")
    plt.ylabel("Execution Time (ms)")
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(sorted(df["num_atoms"].unique().to_list()))
    plt.legend(title="Algorithm")
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.savefig(plot_dir / "time_vs_atoms.png", dpi=300, bbox_inches="tight")
    plt.close()  # Use close() instead of show() for scripted execution


def plot_time_vs_block_size(df):
    """Plot execution time vs block size for GPU algorithms"""
    plt.figure(figsize=(14, 8))

    # Only plot GPU algorithms in consistent order
    gpu_algorithms = sorted(["GRID_2D", "SHARED_MEM", "OUTPUT_PRIV"])
    markers = ["s", "D", "^"]

    for alg, marker in zip(gpu_algorithms, markers):
        alg_df = df.filter(pl.col("algorithm") == alg)

        # Group and sort by block_size numerically
        grouped = (
            alg_df.group_by("block_size")
            .agg(pl.col("time_ms").mean())
            .sort("block_size")
        )

        plt.plot(
            grouped["block_size"].to_list(),
            grouped["time_ms"].to_list(),
            marker=marker,
            markersize=10,
            linewidth=2,
            label=f"{alg}",
        )

    plt.title("GPU Execution Time vs Block Size")
    plt.xlabel("Block Size")
    plt.ylabel("Execution Time (ms)")
    plt.yscale("log")
    plt.xticks(sorted(df["block_size"].unique().to_list()))
    plt.legend(title="GPU Algorithm")
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.savefig(plot_dir / "time_vs_block_size.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_vs_resolution(df):
    """Plot execution time vs resolution for each algorithm"""
    plt.figure(figsize=(14, 8))

    algorithms = sorted(df["algorithm"].unique().to_list())
    markers = ["o", "s", "D", "^"]

    for alg, marker in zip(algorithms, markers):
        alg_df = df.filter(pl.col("algorithm") == alg)

        # Group and sort by resolution numerically
        grouped = (
            alg_df.group_by("resolution")
            .agg(pl.col("time_ms").mean())
            .sort("resolution")
        )

        plt.plot(
            grouped["resolution"].to_list(),
            grouped["time_ms"].to_list(),
            marker=marker,
            markersize=8,
            linewidth=2,
            label=f"{alg}",
        )

    plt.title("Execution Time vs Resolution")
    plt.xlabel("Resolution")
    plt.ylabel("Execution Time (ms)")
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(sorted(df["resolution"].unique().to_list()))
    plt.legend(title="Algorithm")
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.savefig(plot_dir / "time_vs_resolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_vs_atoms_by_resolution(df):
    """Plot execution time vs atoms, faceted by resolution"""
    # Ensure resolutions are sorted
    resolutions = sorted(df["resolution"].unique())

    g = sns.FacetGrid(
        df,
        col="resolution",
        hue="algorithm",
        col_wrap=3,
        height=5,
        aspect=1.2,
        sharey=False,
        col_order=resolutions,
    )
    g.map(sns.lineplot, "num_atoms", "time_ms", marker="o", errorbar="sd")
    g.add_legend(title="Algorithm")
    g.set_axis_labels("Number of Atoms", "Execution Time (ms)")
    g.set_titles("Resolution = {col_name}")
    g.set(yscale="log", xscale="log")

    # Set consistent x-ticks
    for ax in g.axes.flat:
        ax.set_xticks(sorted(df["num_atoms"].unique()))

    plt.tight_layout()
    plt.savefig(
        plot_dir / "time_vs_atoms_by_resolution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    try:
        df = load_data("experiment_results.csv")  # Fixed typo in filename

        # Generate all plots
        plot_time_vs_atoms(df)
        plot_time_vs_block_size(df)
        plot_time_vs_resolution(df)
        plot_time_vs_atoms_by_resolution(df)

        print("All plots generated successfully!")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
