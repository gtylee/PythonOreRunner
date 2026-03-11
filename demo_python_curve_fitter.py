from __future__ import annotations

import argparse
from pathlib import Path

from ore_curve_fit_parity import compare_python_vs_ore, trace_curve


DEFAULT_ORE_XML = Path(
    "/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/"
    "multiccy_benchmark_final/cases/flat_USD_5Y_B/Input/ore.xml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean demo of the Python diagnostic curve fitter against ORE native nodes."
    )
    parser.add_argument(
        "--ore-xml",
        default=str(DEFAULT_ORE_XML),
        help="Path to an ORE case file.",
    )
    parser.add_argument(
        "--currency",
        default="USD",
        help="Discounting currency to trace and compare.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=5,
        help="Number of comparison points to print.",
    )
    parser.add_argument(
        "--quotes-per-segment",
        type=int,
        default=4,
        help="Number of input quotes to print per instrument segment.",
    )
    return parser


def _fmt_number(value: object) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.12f}"
    except Exception:
        return str(value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ore_xml = Path(args.ore_xml).resolve()

    trace = trace_curve(ore_xml, currency=args.currency)
    comparison = compare_python_vs_ore(ore_xml, currency=args.currency)

    payload = trace.payload
    config = payload["curve_config"]
    native_nodes = payload["native_curve_nodes"]
    segment_alignment = payload["segment_alignment"]

    print("Python curve fitter demo")
    print(f"ore_xml: {ore_xml}")
    print(f"curve_handle: {trace.curve_handle}")
    print(f"curve_name: {trace.curve_name}")
    print(f"curve_id: {trace.curve_id}")
    print(f"interpolation: {config['interpolation_variable']}/{config['interpolation_method']}")
    print(f"discount_dependency: {config['discount_curve']}")
    print(f"pillar_count: {len(payload['ore_calibration_trace']['pillars'])}")
    print(f"native_node_count: {len(native_nodes['times'])}")
    print(f"comparison_status: {comparison.status}")
    print(f"max_abs_error: {comparison.max_abs_error}")
    print(f"max_rel_error: {comparison.max_rel_error}")

    print("\nInput instruments")
    for segment in segment_alignment:
        print(
            f"  {segment['type']} | conventions={segment['conventions']} "
            f"| quotes={segment['quote_count']}"
        )
        for quote in segment["quotes"][: args.quotes_per_segment]:
            pillar = quote.get("ore_pillar") or {}
            print(
                "    "
                f"quote={quote['quote_key']} "
                f"market={_fmt_number(quote.get('quote_value'))} "
                f"time={_fmt_number(pillar.get('time'))} "
                f"df={_fmt_number(pillar.get('discountFactor'))} "
                f"zero={_fmt_number(pillar.get('zeroRate'))}"
            )

    print("\nFirst native ORE nodes")
    for time_value, df in list(zip(native_nodes["times"], native_nodes["discount_factors"]))[: args.points]:
        print(f"  t={time_value:.8f} df={df:.12f}")

    print("\nFitted outputs on ORE report grid")
    for point in list(comparison.points)[: args.points]:
        print(
            "  "
            f"t={point.time:.8f} "
            f"ore_df={point.ore_value:.12f} "
            f"python_df={point.engine_value:.12f}"
        )

    print("\nSample ORE vs Python grid comparison")
    for point in list(comparison.points)[: args.points]:
        print(
            "  "
            f"t={point.time:.8f} "
            f"ore={point.ore_value:.12f} "
            f"python={point.engine_value:.12f} "
            f"abs_err={point.abs_error:.3e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
