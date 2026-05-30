from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path
from typing import Any

from .adapters import create_adapter
from .config import ModelConfig, RunSettings
from .dataset import QuestionSample
from .metrics import compute_metrics
from .parsing import parse_answer
from .prompting import get_model_prompt
from .results import (
    PredictionRecord,
    ensure_model_output_dir,
    summarize_predictions,
    write_predictions,
    write_submission,
    write_summary,
)
from .runtime import configure_runtime_noise


def _get_worker_batch_size(model_config: ModelConfig, run_settings: RunSettings) -> int:
    if model_config.mode == "api":
        return max(1, run_settings.max_concurrent_api_requests)
    if model_config.batch_size is not None:
        return model_config.batch_size
    return 1


def _drain_task_batch(task_queue: mp.Queue, first_task: dict[str, Any], batch_size: int) -> tuple[list[dict[str, Any]], bool]:
    task_batch = [first_task]
    saw_stop_signal = False
    while len(task_batch) < batch_size:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break
        if task is None:
            saw_stop_signal = True
            break
        task_batch.append(task)
    return task_batch, saw_stop_signal


def _build_prediction_record(
    sample: QuestionSample,
    prompt: str,
    raw_response: str | None,
    model_name: str,
    error: str | None = None,
) -> PredictionRecord:
    parsed_answer = parse_answer(sample, raw_response)
    metrics = compute_metrics(sample, parsed_answer)
    return PredictionRecord(
        question_id=sample.question_id,
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        subcategory=sample.subcategory,
        task=sample.task,
        gt_type=sample.gt_type,
        stress=sample.stress,
        image_path=sample.image_rel_path,
        record_path=sample.record_relpath,
        prompt=prompt,
        raw_response=raw_response,
        parsed_answer=parsed_answer,
        ground_truth=sample.ground_truth,
        metrics=metrics,
        model_name=model_name,
        error=error,
    )


def _merge_profiling_stats(
    total_stats: dict[str, float],
    worker_stats: dict[str, float] | None,
) -> dict[str, float]:
    if not worker_stats:
        return total_stats
    for key, value in worker_stats.items():
        total_stats[key] = total_stats.get(key, 0.0) + float(value)
    return total_stats


def _worker_loop(
    worker_id: int,
    gpu_id: int | None,
    model_config: ModelConfig,
    run_settings: RunSettings,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    dry_run: bool,
) -> None:
    configure_runtime_noise()
    adapter = None
    try:
        adapter = create_adapter(
            model_config,
            run_settings,
            gpu_id=gpu_id,
            dry_run=dry_run,
        )
    except Exception as exc:  # pragma: no cover - subprocess safety path
        result_queue.put(
            {
                "kind": "fatal_error",
                "worker_id": worker_id,
                "model_name": model_config.name,
                "stage": "initialization",
                "message": (
                    f"Worker {worker_id} failed to initialize model "
                    f"'{model_config.name}': {exc}"
                ),
            }
        )
        return

    batch_size = _get_worker_batch_size(model_config, run_settings)
    while True:
        first_task = task_queue.get()
        if first_task is None:
            break

        task_batch, saw_stop_signal = _drain_task_batch(task_queue, first_task, batch_size)
        sample_batch = [QuestionSample(**task) for task in task_batch]
        prompt_batch = [get_model_prompt(sample) for sample in sample_batch]

        try:
            generation_results = adapter.generate_batch_results(sample_batch)
            if len(generation_results) != len(sample_batch):
                raise RuntimeError(
                    f"Expected {len(sample_batch)} generation results but received {len(generation_results)}."
                )
            for sample, prompt, generation_result in zip(sample_batch, prompt_batch, generation_results):
                result_queue.put(
                    _build_prediction_record(
                        sample=sample,
                        prompt=prompt,
                        raw_response=generation_result.raw_response,
                        model_name=model_config.name,
                        error=generation_result.error,
                    )
                )
        except Exception as exc:  # pragma: no cover - subprocess safety path
            result_queue.put(
                {
                    "kind": "fatal_error",
                    "worker_id": worker_id,
                    "model_name": model_config.name,
                    "stage": "inference",
                    "question_ids": [sample.question_id for sample in sample_batch],
                    "datasets": [sample.dataset for sample in sample_batch],
                    "subcategories": [sample.subcategory for sample in sample_batch],
                    "image_paths": [sample.image_rel_path for sample in sample_batch],
                    "prompts": prompt_batch,
                    "batch_size": len(sample_batch),
                    "message": (
                        f"Worker {worker_id} inference failed for model "
                        f"'{model_config.name}', batch_question_ids="
                        f"{[sample.question_id for sample in sample_batch]}, "
                        f"batch_datasets={[sample.dataset for sample in sample_batch]}, "
                        f"batch_subcategories={[sample.subcategory for sample in sample_batch]}, "
                        f"batch_image_paths={[sample.image_rel_path for sample in sample_batch]}: "
                        f"{''.join(traceback.format_exception_only(type(exc), exc)).strip()}"
                    ),
                }
            )
            return

        if saw_stop_signal:
            break

    if adapter is not None and hasattr(adapter, "shutdown"):
        adapter.shutdown()

    result_queue.put(
        {
            "kind": "worker_done",
            "worker_id": worker_id,
            "profiling": adapter.get_profiling_stats(),
        }
    )


def _terminate_workers(workers: list[mp.Process]) -> None:
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=5)


def _prewarm_trust_remote_code(model_config: ModelConfig, worker_gpus: list[int | None]) -> None:
    if (
        model_config.mode != "local"
        or not model_config.trust_remote_code
        or model_config.ckpt_path is None
        or len(worker_gpus) <= 1
    ):
        return

    configure_runtime_noise()

    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    ckpt_path = model_config.ckpt_path
    AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)

    config_path = ckpt_path / "config.json"
    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)

    auto_map = config_payload.get("auto_map", {})
    for key in ("AutoConfig", "AutoModelForImageTextToText"):
        class_reference = auto_map.get(key)
        if class_reference:
            get_class_from_dynamic_module(class_reference, str(ckpt_path))


def determine_worker_gpus(
    model_config: ModelConfig,
    requested_gpu_ids: list[int],
    dry_run: bool,
) -> list[int | None]:
    if model_config.mode == "api":
        return [None]
    if dry_run and not requested_gpu_ids:
        return [None]
    if model_config.worker_strategy == "single_instance":
        return [requested_gpu_ids[0] if requested_gpu_ids else None]
    return requested_gpu_ids or [None]


def run_model_inference(
    model_config: ModelConfig,
    run_settings: RunSettings,
    samples: list[QuestionSample],
    output_root: Path,
    requested_gpu_ids: list[int],
    dry_run: bool = False,
    progress_callback: Any | None = None,
) -> tuple[list[PredictionRecord], dict[str, Any]]:
    start_time = time.perf_counter()
    worker_gpus = determine_worker_gpus(model_config, requested_gpu_ids, dry_run)
    output_dir = ensure_model_output_dir(output_root, model_config.name)
    predictions_path = output_dir / "predictions.json"
    legacy_predictions_path = output_dir / "predictions.jsonl"
    submission_path = output_dir / "submission.json"
    summary_path = output_dir / "summary.json"
    if predictions_path.exists():
        predictions_path.unlink()
    if legacy_predictions_path.exists():
        legacy_predictions_path.unlink()
    if submission_path.exists():
        submission_path.unlink()

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    workers: list[mp.Process] = []

    _prewarm_trust_remote_code(model_config, worker_gpus)

    for worker_id, gpu_id in enumerate(worker_gpus):
        process = ctx.Process(
            target=_worker_loop,
            args=(
                worker_id,
                gpu_id,
                model_config,
                run_settings,
                task_queue,
                result_queue,
                dry_run,
            ),
        )
        process.start()
        workers.append(process)

    for sample in samples:
        task_queue.put(sample.to_payload())
    for _ in workers:
        task_queue.put(None)

    records: list[PredictionRecord] = []
    completed = 0
    completed_workers = 0
    time_to_first_result_seconds: float | None = None
    profiling_totals: dict[str, float] = {}
    while completed < len(samples) or completed_workers < len(workers):
        try:
            item = result_queue.get(timeout=1.0)
        except queue.Empty:
            dead_workers = [
                worker
                for worker in workers
                if not worker.is_alive() and worker.exitcode not in (0, None)
            ]
            if dead_workers:
                exit_codes = [worker.exitcode for worker in dead_workers]
                _terminate_workers(workers)
                sigkill_hint = ""
                if any(code == -9 for code in exit_codes):
                    sigkill_hint = (
                        " This includes exit code -9 (SIGKILL), which usually means the worker was "
                        "killed by the OS or scheduler, commonly due to CPU RAM / GPU memory pressure. "
                        "Try lowering model_registry.<model>.batch_size, reducing --gpu-ids / "
                        "run_settings.max_workers, or running a smaller --limit first."
                    )
                raise RuntimeError(
                    f"Worker process exited before reporting completion for model "
                    f"{model_config.name}: exit_codes={exit_codes}.{sigkill_hint}"
                )
            continue
        if isinstance(item, dict):
            if item.get("kind") == "worker_done":
                profiling_totals = _merge_profiling_stats(profiling_totals, item.get("profiling"))
                completed_workers += 1
                continue
            if item.get("kind") == "fatal_error":
                if records:
                    write_predictions(predictions_path, records)
                _terminate_workers(workers)
                raise RuntimeError(item["message"])
        if isinstance(item, PredictionRecord):
            records.append(item)
            completed += 1
            if time_to_first_result_seconds is None:
                time_to_first_result_seconds = time.perf_counter() - start_time
            if progress_callback is not None:
                progress_callback(completed, len(samples), item)

    for worker in workers:
        worker.join()
        if worker.exitcode not in (0, None):
            raise RuntimeError(f"Worker process exited with code {worker.exitcode} for model {model_config.name}.")

    summary = summarize_predictions(records)
    summary["model_name"] = model_config.name
    summary["worker_gpus"] = worker_gpus
    summary["dry_run"] = dry_run
    summary["timing"] = {
        "time_to_first_result_seconds": (
            time_to_first_result_seconds if time_to_first_result_seconds is not None else 0.0
        ),
        "total_wall_time_seconds": time.perf_counter() - start_time,
    }
    summary["profiling"] = profiling_totals
    write_predictions(predictions_path, records)
    write_submission(submission_path, records, model_config.name)
    write_summary(summary_path, summary)
    return records, summary
