from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when config.yaml is invalid."""


@dataclass(frozen=True)
class RunSettings:
    models_to_run: str | list[str]
    exclude_molmo2_family: bool
    dataset_root: Path
    manifest_path: Path
    datasets_to_run: str | list[str]
    gt_types_to_run: str | list[str]
    output_dir: Path
    max_concurrent_api_requests: int
    max_new_tokens: int
    temperature: float
    top_p: float
    run_limit: int | None = None
    max_workers: int = 8
    stratified_balanced_sample: bool = False
    sample_seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    name: str
    mode: str
    family: str | None = None
    ckpt_path: Path | None = None
    cache_path: Path | None = None
    torch_dtype: str | None = None
    device_map: Any = "auto"
    trust_remote_code: bool = False
    provider: str | None = None
    api_model_name: str | None = None
    api_url: str | None = None
    api_key_env_var: str | None = None
    worker_strategy: str = "per_gpu"
    batch_size: int | None = None


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    run_settings: RunSettings
    model_registry: dict[str, ModelConfig]

    def _get_family_to_models(self) -> dict[str, list[str]]:
        family_to_models: dict[str, list[str]] = {}
        for model_name, model_config in self.model_registry.items():
            if model_config.family:
                family_to_models.setdefault(model_config.family, []).append(model_name)
        for model_names in family_to_models.values():
            model_names.sort()
        return family_to_models

    def _apply_exclude_rules(self, model_names: list[str], requested: str | list[str]) -> list[str]:
        if requested in {"all", "all_local_VLMs"} and self.run_settings.exclude_molmo2_family:
            model_names = [
                model_name
                for model_name in model_names
                if self.model_registry[model_name].family != "molmo2"
            ]
        return model_names

    def get_selected_model_names(self, override: list[str] | None = None) -> list[str]:
        requested = override if override is not None else self.run_settings.models_to_run
        family_to_models = self._get_family_to_models()

        if requested == "all":
            return self._apply_exclude_rules(list(self.model_registry.keys()), requested)
        if requested == "all_local_VLMs":
            local_model_names = [
                model_name
                for model_name, model_config in self.model_registry.items()
                if model_config.mode == "local"
            ]
            return self._apply_exclude_rules(local_model_names, requested)

        if isinstance(requested, str):
            if requested in family_to_models:
                return family_to_models[requested]
            raise ConfigError(
                "run_settings.models_to_run must be 'all', 'all_local_VLMs', "
                "a known family name, or a non-empty list of model/family names."
            )

        if not isinstance(requested, list) or not requested:
            raise ConfigError(
                "run_settings.models_to_run must be 'all', 'all_local_VLMs', "
                "or a non-empty list of model/family names."
            )

        selected: list[str] = []
        missing: list[str] = []
        for item in requested:
            if item in self.model_registry:
                selected.append(item)
            elif item in family_to_models:
                selected.extend(family_to_models[item])
            else:
                missing.append(item)

        if missing:
            available_families = sorted(family_to_models.keys())
            raise ConfigError(
                f"Unknown models_to_run entries: {missing}. "
                f"Known families: {available_families}"
            )

        deduped_selected = list(dict.fromkeys(selected))
        return deduped_selected

    def get_selected_datasets(self, override: list[str] | None = None) -> str | list[str]:
        requested = override if override is not None else self.run_settings.datasets_to_run
        if requested == "all":
            return "all"
        if not isinstance(requested, list) or not requested:
            raise ConfigError("run_settings.datasets_to_run must be 'all' or a non-empty list.")
        return requested

    def get_selected_gt_types(self, override: list[str] | None = None) -> str | list[str]:
        requested = override if override is not None else self.run_settings.gt_types_to_run
        if requested == "all":
            return "all"
        if not isinstance(requested, list) or not requested:
            raise ConfigError("run_settings.gt_types_to_run must be 'all' or a non-empty list.")
        invalid = sorted(set(requested) - {"mcq", "bbox", "mask"})
        if invalid:
            raise ConfigError(f"Unknown gt_types_to_run entries: {invalid}")
        return requested

    def validate_selected_models(self, model_names: list[str], *, dry_run: bool = False) -> None:
        if dry_run:
            return
        for model_name in model_names:
            model_config = self.model_registry[model_name]
            if model_config.mode == "local":
                if model_config.ckpt_path is None or not model_config.ckpt_path.exists():
                    raise ConfigError(
                        f"Checkpoint path not found for local model '{model_name}': {model_config.ckpt_path}"
                    )


def _require_mapping(data: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{field_name} must be a mapping.")
    return data


def _require_list_or_all(value: Any, field_name: str) -> str | list[str]:
    if value == "all":
        return "all"
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{field_name} must be 'all' or a non-empty list of strings.")
    return value


def _require_model_selector(value: Any, field_name: str) -> str | list[str]:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    raise ConfigError(
        f"{field_name} must be a non-empty string selector or a non-empty list of strings."
    )


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string.")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise ConfigError(f"{field_name} must be an integer.")
    return value


def _require_int_at_least(value: Any, field_name: str, minimum: int) -> int:
    parsed = _require_int(value, field_name)
    if parsed < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum}.")
    return parsed


def _require_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, field_name)


def _require_optional_int_at_least(value: Any, field_name: str, minimum: int) -> int | None:
    if value is None:
        return None
    return _require_int_at_least(value, field_name, minimum)


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a boolean.")
    return value


def _require_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be a number.")
    return float(value)


def _resolve_local_cache_path(
    project_root: Path,
    cache_dir: Path,
    family_name: str,
    model_name: str,
    use_family_cache_for_internvl: bool,
    use_family_cache_for_molmo2: bool,
) -> Path:
    use_family_cache = (
        (family_name == "internvl" and use_family_cache_for_internvl)
        or (family_name == "molmo2" and use_family_cache_for_molmo2)
    )
    cache_leaf = family_name if use_family_cache else model_name
    return (project_root / cache_dir / cache_leaf).resolve()


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _infer_project_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _load_yaml_mapping(path: Path, field_name: str) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return _require_mapping(payload, field_name)


def _load_registry_payload(root: dict[str, Any], project_root: Path, config_path: Path) -> dict[str, Any]:
    registry_payload = root.get("model_registry")
    registry_path_value = root.get("model_registry_path")
    if registry_payload is not None and registry_path_value is not None:
        raise ConfigError("Specify either model_registry or model_registry_path, not both.")
    if registry_payload is not None:
        return _require_mapping(registry_payload, "model_registry")
    if registry_path_value is None:
        raise ConfigError("config must contain model_registry or model_registry_path.")

    registry_path = Path(_require_str(registry_path_value, "model_registry_path")).expanduser()
    if not registry_path.is_absolute():
        config_relative_path = (config_path.parent / registry_path).resolve()
        registry_path = config_relative_path if config_relative_path.exists() else _resolve_path(project_root, registry_path)
    payload = _load_yaml_mapping(registry_path, "model_registry file")
    if "model_registry" in payload:
        return _require_mapping(payload["model_registry"], "model_registry")
    return payload


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).expanduser().resolve()
    root = _load_yaml_mapping(config_path, "config")
    run_payload = _require_mapping(root.get("run_settings"), "run_settings")
    project_root = _infer_project_root(config_path)
    registry_payload = _load_registry_payload(root, project_root, config_path)
    cache_dir = Path(_require_str(root.get("cache_dir", "crossCache"), "cache_dir"))
    use_family_cache_for_internvl = _require_bool(
        root.get("use_family_cache_for_internvl", False),
        "use_family_cache_for_internvl",
    )
    use_family_cache_for_molmo2 = _require_bool(
        root.get("use_family_cache_for_molmo2", False),
        "use_family_cache_for_molmo2",
    )

    dataset_root = _resolve_path(
        project_root,
        _require_str(run_payload.get("dataset_root"), "run_settings.dataset_root"),
    )
    manifest_path = Path(_require_str(run_payload.get("manifest_path", "manifest.jsonl"), "run_settings.manifest_path"))

    run_settings = RunSettings(
        models_to_run=_require_model_selector(
            run_payload.get("models_to_run"),
            "run_settings.models_to_run",
        ),
        exclude_molmo2_family=_require_bool(
            run_payload.get("exclude_molmo2_family", False),
            "run_settings.exclude_molmo2_family",
        ),
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        datasets_to_run=_require_list_or_all(
            run_payload.get("datasets_to_run", "all"),
            "run_settings.datasets_to_run",
        ),
        gt_types_to_run=_require_list_or_all(
            run_payload.get("gt_types_to_run", "all"),
            "run_settings.gt_types_to_run",
        ),
        output_dir=_resolve_path(
            project_root,
            _require_str(run_payload.get("output_dir"), "run_settings.output_dir"),
        ),
        max_concurrent_api_requests=_require_int(
            run_payload.get("max_concurrent_API_requests", 20),
            "run_settings.max_concurrent_API_requests",
        ),
        max_new_tokens=_require_int(run_payload.get("max_new_tokens", 16), "run_settings.max_new_tokens"),
        temperature=_require_float(run_payload.get("temperature", 0.0), "run_settings.temperature"),
        top_p=_require_float(run_payload.get("top_p", 1.0), "run_settings.top_p"),
        run_limit=_require_optional_int_at_least(
            run_payload.get("run_limit"),
            "run_settings.run_limit",
            1,
        ),
        max_workers=_require_int_at_least(
            run_payload.get("max_workers", 8),
            "run_settings.max_workers",
            1,
        ),
        stratified_balanced_sample=_require_bool(
            run_payload.get("stratified_balanced_sample", False),
            "run_settings.stratified_balanced_sample",
        ),
        sample_seed=_require_int(run_payload.get("sample_seed", 42), "run_settings.sample_seed"),
    )

    model_registry: dict[str, ModelConfig] = {}
    for model_name, entry in registry_payload.items():
        item = _require_mapping(entry, f"model_registry.{model_name}")
        mode = _require_str(item.get("mode"), f"model_registry.{model_name}.mode")
        if mode == "local":
            ckpt_value = _require_str(item.get("ckpt_path"), f"model_registry.{model_name}.ckpt_path")
            family_name = _require_str(item.get("family"), f"model_registry.{model_name}.family")
            resolved_cache_path = _resolve_local_cache_path(
                project_root=project_root,
                cache_dir=cache_dir,
                family_name=family_name,
                model_name=model_name,
                use_family_cache_for_internvl=use_family_cache_for_internvl,
                use_family_cache_for_molmo2=use_family_cache_for_molmo2,
            )
            model_registry[model_name] = ModelConfig(
                name=model_name,
                mode=mode,
                family=family_name,
                ckpt_path=(project_root / ckpt_value).resolve(),
                cache_path=resolved_cache_path,
                torch_dtype=_require_str(
                    item.get("torch_dtype", "bfloat16"),
                    f"model_registry.{model_name}.torch_dtype",
                ),
                device_map=item.get("device_map", "auto"),
                trust_remote_code=bool(item.get("trust_remote_code", False)),
                worker_strategy=_require_str(
                    item.get("worker_strategy", "per_gpu"),
                    f"model_registry.{model_name}.worker_strategy",
                ),
                batch_size=_require_optional_int_at_least(
                    item.get("batch_size"),
                    f"model_registry.{model_name}.batch_size",
                    1,
                ),
            )
        elif mode == "api":
            model_registry[model_name] = ModelConfig(
                name=model_name,
                mode=mode,
                provider=_require_str(item.get("provider"), f"model_registry.{model_name}.provider"),
                api_model_name=_require_str(
                    item.get("api_model_name"),
                    f"model_registry.{model_name}.api_model_name",
                ),
                api_url=_require_str(item.get("api_url"), f"model_registry.{model_name}.api_url"),
                api_key_env_var=_require_str(
                    item.get("api_key_env_var"),
                    f"model_registry.{model_name}.api_key_env_var",
                ),
            )
        else:
            raise ConfigError(f"model_registry.{model_name}.mode must be 'local' or 'api'.")

    return AppConfig(project_root=project_root, run_settings=run_settings, model_registry=model_registry)
