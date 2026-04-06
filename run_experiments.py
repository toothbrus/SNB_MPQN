from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List

from quantum_simulator import (
    Circuit,
    SimulationParams,
    build_circuit_from_qasm,
    generate_ghz_family_circuit,
    run_execution_vs_insertion_loss,
    run_execution_vs_n,
    write_csv,
)


def parse_int_list(value: str) -> List[int]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in items]


def maybe_plot_relation_1(csv_summary: Path, png_out: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return False

    df = pd.read_csv(csv_summary)
    plt.figure(figsize=(8, 5))
    for network in sorted(df["network"].unique()):
        part = df[df["network"] == network].sort_values("n_qpus")
        plt.errorbar(
            part["n_qpus"],
            part["mean_execution_time"],
            yerr=part["std_execution_time"],
            marker="o",
            capsize=3,
            label=network,
        )

    plt.xlabel("Number of QPUs (N)")
    plt.ylabel("Execution time (s)")
    plt.title("Execution Time vs N")
    plt.grid(True, alpha=0.3)
    plt.legend()
    png_out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_out, dpi=180)
    plt.close()
    return True


def maybe_plot_relation_2(csv_summary: Path, png_out: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return False

    df = pd.read_csv(csv_summary)
    plt.figure(figsize=(8, 5))
    for network in sorted(df["network"].unique()):
        part = df[df["network"] == network].sort_values("loss_stage_db")
        plt.errorbar(
            part["loss_stage_db"],
            part["mean_execution_time"],
            yerr=part["std_execution_time"],
            marker="o",
            capsize=3,
            label=network,
        )

    plt.xlabel("Insertion loss per stage (dB)")
    plt.ylabel("Execution time (s)")
    plt.title("Execution Time vs Insertion Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    png_out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_out, dpi=180)
    plt.close()
    return True


def _require_qasm_file(args: argparse.Namespace) -> Path:
    if not args.qasm_file:
        raise ValueError("--qasm-file is required when --circuit-mode=qasm")
    qasm_file = Path(args.qasm_file)
    if not qasm_file.exists():
        raise FileNotFoundError(f"QASM file not found: {qasm_file}")
    return qasm_file


def make_relation1_builder(args: argparse.Namespace) -> Callable[[int], Circuit]:
    if args.circuit_mode == "ghz_family":
        return generate_ghz_family_circuit

    qasm_file = _require_qasm_file(args)

    def _builder(n_qpus: int) -> Circuit:
        return build_circuit_from_qasm(
            qasm_path=qasm_file,
            n_qpus=n_qpus,
            mapping=args.qasm_mapping,
        )

    return _builder


def make_relation2_builder(args: argparse.Namespace) -> Callable[[int], Circuit]:
    if args.circuit_mode == "ghz_family":
        return generate_ghz_family_circuit

    qasm_file = _require_qasm_file(args)

    def _builder(n_qpus: int) -> Circuit:
        return build_circuit_from_qasm(
            qasm_path=qasm_file,
            n_qpus=n_qpus,
            mapping=args.qasm_mapping,
        )

    return _builder


def run_relation_1(
    args: argparse.Namespace,
    params: SimulationParams,
    outdir: Path,
    circuit_builder: Callable[[int], Circuit],
) -> None:
    seeds = list(range(args.seeds))
    n_values = parse_int_list(args.n_values)

    result = run_execution_vs_n(
        n_values=n_values,
        seeds=seeds,
        base_params=params,
        circuit_builder=circuit_builder,
    )

    raw_path = outdir / "relation1_raw.csv"
    summary_path = outdir / "relation1_summary.csv"
    write_csv(raw_path, result.rows)
    write_csv(summary_path, result.summary_rows)

    plotted = maybe_plot_relation_1(summary_path, outdir / "relation1_plot.png")
    if plotted:
        print(f"[relation1] Plot saved to: {outdir / 'relation1_plot.png'}")
    else:
        print("[relation1] Plot skipped (matplotlib/pandas unavailable).")

    print(f"[relation1] Raw CSV: {raw_path}")
    print(f"[relation1] Summary CSV: {summary_path}")


def run_relation_2(
    args: argparse.Namespace,
    params: SimulationParams,
    outdir: Path,
    circuit_builder: Callable[[int], Circuit],
) -> None:
    seeds = list(range(args.seeds))

    if args.loss_points < 2:
        raise ValueError("loss_points must be >= 2")

    step = (args.loss_max - args.loss_min) / (args.loss_points - 1)
    losses = [args.loss_min + i * step for i in range(args.loss_points)]

    result = run_execution_vs_insertion_loss(
        n_qpus_fixed=args.fixed_n,
        loss_stage_db_values=losses,
        seeds=seeds,
        base_params=params,
        circuit_builder=circuit_builder,
    )

    raw_path = outdir / "relation2_raw.csv"
    summary_path = outdir / "relation2_summary.csv"
    write_csv(raw_path, result.rows)
    write_csv(summary_path, result.summary_rows)

    plotted = maybe_plot_relation_2(summary_path, outdir / "relation2_plot.png")
    if plotted:
        print(f"[relation2] Plot saved to: {outdir / 'relation2_plot.png'}")
    else:
        print("[relation2] Plot skipped (matplotlib/pandas unavailable).")

    print(f"[relation2] Raw CSV: {raw_path}")
    print(f"[relation2] Summary CSV: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict QMSN vs MPQN simulation experiments")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--seeds", type=int, default=30)

    parser.add_argument(
        "--circuit-mode",
        choices=["qasm", "ghz_family"],
        default="qasm",
        help="relation-1 circuit source; relation-2 uses the same source mode",
    )
    parser.add_argument("--qasm-file", default=None)
    parser.add_argument(
        "--qasm-mapping",
        choices=["round_robin", "block", "one_to_one"],
        default="round_robin",
    )

    parser.add_argument("--n-values", default="2,4,8,16")

    parser.add_argument("--fixed-n", type=int, default=16)
    parser.add_argument("--loss-min", type=float, default=0.01)
    parser.add_argument("--loss-max", type=float, default=0.5)
    parser.add_argument("--loss-points", type=int, default=10)

    parser.add_argument("--p-e", type=float, default=0.5)
    parser.add_argument("--p-t", type=float, default=0.95)

    parser.add_argument("--p-d", type=float, default=0.9)
    parser.add_argument("--t-retry", type=float, default=1e-6)

    parser.add_argument("--p-net", type=float, default=0.98)
    parser.add_argument("--p-ent", type=float, default=2.18e-4)
    parser.add_argument("--t-decoherence", type=float, default=1.0)
    parser.add_argument("--t-local", type=float, default=100e-9)
    parser.add_argument("--t-remote", type=float, default=1.6e-6)
    parser.add_argument("--qmsn-period", type=float, default=100e-6)
    parser.add_argument("--max-retries", type=int, default=100000)
    parser.add_argument("--phase-error-mzi", type=float, default=0.01)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    params = SimulationParams(
        p_e=args.p_e,
        p_t=args.p_t,
        p_net=args.p_net,
        p_d=args.p_d,
        t_retry=args.t_retry,
        t_decoherence=args.t_decoherence,
        t_local=args.t_local,
        t_remote=args.t_remote,
        qmsn_period=args.qmsn_period,
        max_retries=args.max_retries,
        phase_error_mzi=args.phase_error_mzi,
    )

    relation1_builder = make_relation1_builder(args)
    relation2_builder = make_relation2_builder(args)

    run_relation_1(args, params, outdir, relation1_builder)
    run_relation_2(args, params, outdir, relation2_builder)


if __name__ == "__main__":
    main()
