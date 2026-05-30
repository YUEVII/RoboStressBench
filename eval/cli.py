from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from typing import Any, TextIO

from .config import ConfigError, load_config
from .dataset import load_evaluation_samples
from .runtime import configure_runtime_noise
from .scheduler import run_model_inference


LOGGER = logging.getLogger("robostressbench")

_STRESS_PRIMARY_DISPLAY = {"G": "Geometry", "M": "Material", "V": "Viewpoint", "L": "Lighting"}


class _StdoutTee:
    """Duplicate stdout writes to terminal and a log file (verbatim)."""

    __slots__ = ("_terminal", "_log_file")

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self._terminal = terminal
        self._log_file = log_file

    def write(self, data: Any) -> int:
        text = data if isinstance(data, str) else str(data)
        self._terminal.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()


def _stress_total_acc_display(stats: dict | None) -> str:
    if not isinstance(stats, dict):
        return "N/A"
    if stats.get("stress_bucket_size", 0) == 0:
        return "N/A"
    ta = stats.get("total_acc")
    if ta is None:
        return "N/A"
    return f"{float(ta):.4f}"


def _parse_csv_or_space_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed or None


def _parse_gpu_ids(raw_value: str | None) -> list[int] | None:
    if raw_value is None or not raw_value.strip():
        return None
    gpu_ids = []
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        gpu_ids.append(int(item))
    return gpu_ids


def _detect_available_gpu_ids() -> list[int]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        entries = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if not entries:
            return []
        if all(entry.isdigit() for entry in entries):
            return list(range(len(entries)))

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    gpu_ids = []
    for line in result.stdout.splitlines():
        item = line.strip()
        if item.isdigit():
            gpu_ids.append(int(item))
    return gpu_ids


def _render_progress(model_name: str, completed: int, total: int) -> None:
    progress_text = f"\r[{model_name}] progress: {completed}/{total}"
    sys.stdout.write(progress_text)
    sys.stdout.flush()


def _print_model_summary(model_name: str, summary: dict) -> None:
    overall_stats = summary["overall_statistics"]
    timing = summary.get("timing", {})
    profiling = summary.get("profiling", {})
    time_to_first_result_seconds = timing.get("time_to_first_result_seconds", 0.0)
    total_wall_time_seconds = timing.get("total_wall_time_seconds", 0.0)
    unparsed_by = overall_stats.get("unparsed_by_gt_type") or {}
    unparsed_ratio_by = overall_stats.get("unparsed_ratio_by_gt_type") or {}
    mcq_u = unparsed_by.get("mcq", 0)
    mask_u = unparsed_by.get("mask", 0)
    bbox_u = unparsed_by.get("bbox", 0)
    mcq_up_r = unparsed_ratio_by.get("mcq", 0.0)
    mask_up_r = unparsed_ratio_by.get("mask", 0.0)
    bbox_up_r = unparsed_ratio_by.get("bbox", 0.0)

    per_l1 = summary.get("per_1st_level_stress") or {}
    per_l2 = summary.get("per_2nd_level_stress") or {}
    l1_labels = [f"L1 {_STRESS_PRIMARY_DISPLAY[c]}" for c in ("G", "M", "V", "L")]
    l2_labels = [f"L2 stress {k}" for k in per_l2.keys()]

    labels = [
        "Load Time",
        "Task Time",
        "MCQ Unparsed",
        "Mask Unparsed",
        "BBox Unparsed",
        "MCQ ACC",
        "Mask ACC",
        "BBox Mean IoU",
        "BBox IoU≥0.5",
        "BBox IoU≥0.95",
        *l1_labels,
        *l2_labels,
    ]
    label_width = max(len(label) for label in labels)

    print(f"[{model_name}]")
    print(f"  {'Load Time'.ljust(label_width)}  = {time_to_first_result_seconds:.2f}s")
    print(f"  {'Task Time'.ljust(label_width)}  = {total_wall_time_seconds:.2f}s")
    print(
        f"  {'MCQ Unparsed'.ljust(label_width)}  = "
        f"{mcq_u} ({mcq_up_r * 100:.2f}% of mcq)"
    )
    print(
        f"  {'Mask Unparsed'.ljust(label_width)}  = "
        f"{mask_u} ({mask_up_r * 100:.2f}% of mask)"
    )
    print(
        f"  {'BBox Unparsed'.ljust(label_width)}  = "
        f"{bbox_u} ({bbox_up_r * 100:.2f}% of bbox)"
    )
    print()
    mcq_acc = overall_stats.get("mcq_accuracy")
    mask_acc = overall_stats.get("mask_accuracy")
    bbox_mean = overall_stats.get("bbox_mean_iou")
    bbox_ge = overall_stats.get("bbox_iou_ge_0_5_ratio")
    bbox_ge095 = overall_stats.get("bbox_iou_ge_0_95_ratio")
    print(f"  {'MCQ ACC'.ljust(label_width)}  = {mcq_acc:.4f}" if mcq_acc is not None else f"  {'MCQ ACC'.ljust(label_width)}  = n/a")
    print(f"  {'Mask ACC'.ljust(label_width)}  = {mask_acc:.4f}" if mask_acc is not None else f"  {'Mask ACC'.ljust(label_width)}  = n/a")
    print(
        f"  {'BBox Mean IoU'.ljust(label_width)}  = {bbox_mean:.4f}"
        if bbox_mean is not None
        else f"  {'BBox Mean IoU'.ljust(label_width)}  = n/a"
    )
    print(
        f"  {'BBox IoU≥0.5'.ljust(label_width)}  = {bbox_ge:.4f}"
        if bbox_ge is not None
        else f"  {'BBox IoU≥0.5'.ljust(label_width)}  = n/a"
    )
    print(
        f"  {'BBox IoU≥0.95'.ljust(label_width)}  = {bbox_ge095:.4f}"
        if bbox_ge095 is not None
        else f"  {'BBox IoU≥0.95'.ljust(label_width)}  = n/a"
    )
    print()
    print("  Total accuracy — stress breakdown:")
    print(
        "    (per L1/L2: mean over samples in that bucket; bbox counts 1 iff IoU ≥ 0.95)"
    )
    print()
    for code in ("G", "M", "V", "L"):
        lbl = f"L1 {_STRESS_PRIMARY_DISPLAY[code]}"
        stats = per_l1.get(code) or {}
        disp = _stress_total_acc_display(stats)
        print(f"  {lbl.ljust(label_width)}  = {disp}")
    for composite_key, stats in per_l2.items():
        lbl = f"L2 stress {composite_key}"
        disp = _stress_total_acc_display(stats if isinstance(stats, dict) else None)
        print(f"  {lbl.ljust(label_width)}  = {disp}")
    profile_total = sum(
        profiling.get(key, 0.0)
        for key in (
            "load_images_seconds",
            "prepare_inputs_seconds",
            "generate_seconds",
            "decode_seconds",
        )
    )
    if profile_total > 0:
        batch_count = int(profiling.get("batch_count", 0.0))
        sample_count = int(profiling.get("sample_count", 0.0))
        print("[Profiling]")
        print(f"  batches          = {batch_count}")
        print(f"  samples          = {sample_count}")
        for label, key in (
            ("load_images", "load_images_seconds"),
            ("prepare_inputs", "prepare_inputs_seconds"),
            ("generate", "generate_seconds"),
            ("decode", "decode_seconds"),
        ):
            seconds = profiling.get(key, 0.0)
            ratio = (seconds / profile_total * 100.0) if profile_total else 0.0
            print(f"  {label.ljust(16)} = {seconds:.2f}s ({ratio:.1f}%)")
    print()


def _print_run_overview(
    gpu_ids: list[int],
    selected_datasets: str | list[str],
    selected_gt_types: str | list[str],
    selected_models: list[str],
) -> None:
    gpu_id_text = ", ".join(str(gpu_id) for gpu_id in gpu_ids) if gpu_ids else "none"
    datasets_text = selected_datasets if selected_datasets == "all" else ", ".join(selected_datasets)
    gt_types_text = selected_gt_types if selected_gt_types == "all" else ", ".join(selected_gt_types)
    models_text = ", ".join(selected_models)

    print("[Run Overview]")
    print(f"  GPU Count          = {len(gpu_ids)}")
    print(f"  GPU IDs            = {gpu_id_text}")
    print(f"  Datasets           = {datasets_text}")
    print(f"  GT Types           = {gt_types_text}")
    print(f"  Models             = {models_text}")
    print()


def _merge_run_profiling_totals(
    total_profiling: dict[str, float],
    summary: dict,
) -> dict[str, float]:
    profiling = summary.get("profiling", {})
    for key in ("load_images_seconds", "prepare_inputs_seconds", "generate_seconds"):
        total_profiling[key] = total_profiling.get(key, 0.0) + float(profiling.get(key, 0.0))
    return total_profiling


def _print_total_timing(
    total_load_time_seconds: float,
    total_task_time_seconds: float,
    total_profiling: dict[str, float] | None = None,
) -> None:
    print("[Run Timing]")
    print(f"  Total load time    = {total_load_time_seconds:.2f}s")
    print(f"  Total task time    = {total_task_time_seconds:.2f}s")
    if total_profiling:
        load_seconds = total_profiling.get("load_images_seconds", 0.0)
        prepare_seconds = total_profiling.get("prepare_inputs_seconds", 0.0)
        generate_seconds = total_profiling.get("generate_seconds", 0.0)
        profile_total = load_seconds + prepare_seconds + generate_seconds
        print("  Total profile:")
        for label, seconds in (
            ("Load Time", load_seconds),
            ("Prepare Time", prepare_seconds),
            ("Generate Time", generate_seconds),
        ):
            ratio = (seconds / profile_total * 100.0) if profile_total else 0.0
            print(f"    {label.ljust(13)} = {seconds:.2f}s ({ratio:.1f}%)")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RoboStressBench VLM evaluation.")
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML",
        help="Path to experiment YAML (e.g. configs/local_test.yaml); required, no default.",
    )
    parser.add_argument("--models", nargs="*", help="Override models_to_run with one or more model names.")
    parser.add_argument("--datasets", nargs="*", help="Override datasets_to_run with one or more dataset names.")
    parser.add_argument("--gt-types", nargs="*", help="Override gt_types_to_run with mcq, bbox, and/or mask.")
    parser.add_argument("--gpu-ids", help="Comma-separated GPU ids, for example: 0,1,2")
    parser.add_argument("--limit", type=int, help="Optional max number of questions to run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline without loading real models. Useful for scheduler validation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_runtime_noise()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        app_config = load_config(args.config)
        selected_models = app_config.get_selected_model_names(_parse_csv_or_space_list(args.models))
        app_config.validate_selected_models(selected_models, dry_run=args.dry_run)
        selected_datasets = app_config.get_selected_datasets(_parse_csv_or_space_list(args.datasets))
        selected_gt_types = app_config.get_selected_gt_types(_parse_csv_or_space_list(args.gt_types))
        sample_limit = args.limit if args.limit is not None else app_config.run_settings.run_limit
        rs = app_config.run_settings
        samples = load_evaluation_samples(
            app_config,
            selected_datasets=selected_datasets,
            selected_gt_types=selected_gt_types,
            limit=sample_limit,
            stratified=rs.stratified_balanced_sample,
            sample_seed=rs.sample_seed,
        )

        output_root = app_config.run_settings.output_dir
        output_root.mkdir(parents=True, exist_ok=True)
        gpu_cli = args.gpu_ids
        gpu_ids_explicit = _parse_gpu_ids(gpu_cli)
        if gpu_ids_explicit is None:
            detected_gpu_ids = _detect_available_gpu_ids()
            max_workers = app_config.run_settings.max_workers
            if len(detected_gpu_ids) > max_workers:
                LOGGER.info(
                    "Detected %s GPUs; using first %s (run_settings.max_workers=%s)",
                    len(detected_gpu_ids),
                    max_workers,
                    max_workers,
                )
                gpu_ids = detected_gpu_ids[:max_workers]
            else:
                gpu_ids = detected_gpu_ids
        else:
            gpu_ids = gpu_ids_explicit

        statistics_log_path = output_root / "run_statistics.log"
        original_stdout = sys.stdout
        statistics_log_handle = statistics_log_path.open("w", encoding="utf-8")
        sys.stdout = _StdoutTee(original_stdout, statistics_log_handle)
        try:
            _print_run_overview(gpu_ids, selected_datasets, selected_gt_types, selected_models)
            total_load_time_seconds = 0.0
            total_task_time_seconds = 0.0
            total_profiling: dict[str, float] = {}

            for model_name in selected_models:
                model_config = app_config.model_registry[model_name]
                LOGGER.info("Running model %s on %s questions", model_name, len(samples))
                print(f"[{model_name}] loading model...")

                progress_started = False

                def progress_callback(completed: int, total: int, _record) -> None:
                    nonlocal progress_started
                    progress_started = True
                    _render_progress(model_name, completed, total)

                _, summary = run_model_inference(
                    model_config=model_config,
                    run_settings=app_config.run_settings,
                    samples=samples,
                    output_root=output_root,
                    requested_gpu_ids=gpu_ids,
                    dry_run=args.dry_run,
                    progress_callback=progress_callback,
                )
                if progress_started:
                    print()
                timing = summary.get("timing", {})
                total_load_time_seconds += timing.get("time_to_first_result_seconds", 0.0)
                total_task_time_seconds += timing.get("total_wall_time_seconds", 0.0)
                total_profiling = _merge_run_profiling_totals(total_profiling, summary)
                _print_model_summary(model_name, summary)
            _print_total_timing(
                total_load_time_seconds,
                total_task_time_seconds,
                total_profiling=total_profiling,
            )
        finally:
            sys.stdout = original_stdout
            statistics_log_handle.close()

        return 0
    except ConfigError as exc:
        parser.error(str(exc))
    except Exception as exc:  # pragma: no cover - CLI protection path
        sys.stderr.write("\n")
        sys.stderr.flush()
        LOGGER.exception("Run failed")
        parser.exit(status=1, message=f"Run failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
