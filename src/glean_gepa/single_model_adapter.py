"""Adapter for optimizing prompts in single-model iterations."""

from __future__ import annotations

from typing import Any, cast

from glean_gepa.adapter_types import (
    ALDataInst,
    SingleModelALDataInst,
    SingleModelALRolloutOutput,
    SingleModelALTrajectory,
)
from glean_gepa.al_adapter import (
    ALRunner,
    GleanAdapterBase,
    Thresholds,
)
from glean_gepa.batch import EvalRunIds, GleanEvaluationBatch
from glean_gepa.focused_evalset import SESSION_BUCKET_TYPE, ensure_focused_eval_set, resolve_eval_run_target
from glean_gepa.objectives import SingleModelObjective
from glean_gepa.objectives.shell import ShellSuccessObjective, ShellToolTelemetryPendingError
from glean_gepa.prompt import compile_encoded_prompt
from glean_gepa.prompt_constants import WRITING_CODE_KEY


class SingleModelAdapter(GleanAdapterBase):
    """Optimize prompts for a single student model using an injected telemetry objective."""

    supports_high_signal_eval = True

    def high_signal_batch(self, eval_batch: GleanEvaluationBatch) -> list[ALDataInst]:
        """Keep the parent eval ID required to map trace-side UUIDs to source entries."""
        prepared = super().high_signal_batch(eval_batch)
        source_run_ids: dict[tuple[str, str, tuple[str, ...]], str] = {}
        for trajectory in eval_batch.trajectories or []:
            if trajectory["score"] >= 1.0:
                continue
            data = trajectory["data"]
            output = trajectory["output"]
            source_eval_run_id = data.get("eval_run_id") or output.get("student_eval_run_id")
            if source_eval_run_id:
                key = (data["eval_set_name"], data["eval_set_version"], tuple(data["deployment_ids"]))
                source_run_ids.setdefault(key, source_eval_run_id)

        enriched: list[ALDataInst] = []
        for data in prepared:
            key = (data["eval_set_name"], data["eval_set_version"], tuple(data["deployment_ids"]))
            source_eval_run_id = source_run_ids.get(key)
            enriched.append({**data, "source_eval_run_id": source_eval_run_id} if source_eval_run_id else data)
        return enriched

    def prepare_high_signal_batch(self, batch: list[ALDataInst]) -> list[ALDataInst] | None:
        """Upload/reuse focused eval sets once, before concurrent child screening."""
        if self.objective.focused_bucket_type != SESSION_BUCKET_TYPE:
            return super().prepare_high_signal_batch(batch)
        prepared: list[ALDataInst] = []
        for data in batch:
            entry_ids = None if data.get("validation_only") else data.get("eval_entry_ids")
            if not entry_ids:
                prepared.append(data)
                continue

            source_eval_run_id = data.get("source_eval_run_id")
            if not source_eval_run_id:
                print("[Focused eval set] Missing the parent eval run ID needed to resolve source entries")
                return None
            source_entries = self.objective.prepare_focused_source_entries(
                eval_set_name=data["eval_set_name"],
                eval_set_version=data["eval_set_version"],
                eval_run_id=source_eval_run_id,
                entry_ids=entry_ids,
                deployment_ids=data["deployment_ids"],
            )
            if source_entries is None:
                return None
            resolved_entry_ids = sorted({str(entry["id"]) for entry in source_entries})
            missing_entry_ids = sorted(set(entry_ids) - set(resolved_entry_ids))
            if missing_entry_ids:
                print(
                    f"[Focused eval set] Skipping {len(missing_entry_ids)} entries without resolved stt, runId, "
                    f"and traceId: {', '.join(missing_entry_ids)}"
                )
            if not resolved_entry_ids:
                print("[Focused eval set] None of the requested entries could be resolved")
                return None
            focused = ensure_focused_eval_set(
                self.runner.evalcli,
                base_eval_set_name=data["eval_set_name"],
                base_eval_set_version=data["eval_set_version"],
                deployment_ids=data["deployment_ids"],
                entry_ids=resolved_entry_ids,
                source_entries=source_entries,
                bucket_type=self.objective.focused_bucket_type,
            )
            if focused is None:
                return None
            prepared.append(
                {
                    **data,
                    "eval_entry_ids": resolved_entry_ids,
                    "eval_set_name": focused.name,
                    "eval_set_version": focused.version,
                    "focused_eval_set_name": focused.name,
                    "focused_eval_set_version": focused.version,
                }
            )
        return prepared

    def __init__(
        self,
        runner: ALRunner,
        thresholds: Thresholds,
        student_model: str,
        *,
        bigquery_client: Any | None = None,
        agentspan_lookback_days: int = 1,
        editable_modules: list[str] | None = None,
        cache_file: str | None = None,
        primary_objective: str | None = None,
        default_frontier_type: str = "objective",
        composite_weights: dict[str, float] | None = None,
        constant_scores: dict[str, float] | None = None,
        objective: SingleModelObjective | None = None,
    ):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.agentspan_lookback_days = agentspan_lookback_days
        self.objective = objective or ShellSuccessObjective(
            bigquery_client=bigquery_client,
            lookback_days=agentspan_lookback_days,
        )
        self.telemetry_dimensions = self.objective.telemetry_dimensions
        super().__init__(
            runner=runner,
            thresholds=thresholds,
            student_model=student_model,
            evaluate_fn=self._evaluate_single_model,
            failure_pattern_fn=self.objective.failure_pattern,
            reflective_example_fn=self.objective.build_reflective_example,
            reflection_prompt_fn=self.objective.reflection_prompt,
            reflective_metrics_fn=self.objective.format_reflective_metrics,
            failure_label=self.objective.failure_label,
            primary_objective=primary_objective or self.objective.name,
            default_frontier_type=default_frontier_type,
            editable_modules=list(editable_modules) if editable_modules else [WRITING_CODE_KEY],
            composite_weights=dict({self.objective.name: 1.0} if composite_weights is None else composite_weights),
            constant_scores=dict(constant_scores or {}),
            cache_file=cache_file,
        )

    @property
    def _eval_analysis_cache(self) -> dict[str, Any]:
        return getattr(self.objective, "_eval_analysis_cache", {})

    def _extra_cache_payload(self) -> dict[str, Any]:
        return {"eval_analysis_cache": self.objective.cache_payload()}

    def _load_extra_cache(self, data: dict[str, Any]) -> None:
        self.objective.load_cache(data.get("eval_analysis_cache", {}))

    def _get_or_fetch_analysis(
        self,
        eval_id: str,
        *,
        include_error_examples: bool = True,
        include_per_entry: bool = True,
    ):
        analysis = self.objective.analyze(
            eval_id,
            include_error_examples=include_error_examples,
            include_per_entry=include_per_entry,
            evalcli=self.runner.evalcli,
        )
        if eval_id in self._eval_analysis_cache:
            self._save_cache()
        return analysis

    def _evaluate_single_model(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> GleanEvaluationBatch:
        result = self._evaluate_student_objective(
            cast(list[SingleModelALDataInst], batch), candidate, capture_traces=capture_traces
        )
        if capture_traces:
            return result
        return GleanEvaluationBatch(
            outputs=result.outputs,
            scores=result.scores,
            trajectories=None,
            objective_scores=result.objective_scores,
            summary=result.summary,
            eval_run_ids=result.eval_run_ids,
        )

    def _evaluate_student_objective(
        self,
        batch: list[SingleModelALDataInst],
        candidate: dict[str, str],
        *,
        capture_traces: bool,
    ) -> GleanEvaluationBatch[SingleModelALTrajectory, SingleModelALRolloutOutput]:
        if not batch:
            return GleanEvaluationBatch(
                outputs=[],
                scores=[],
                trajectories=None,
                objective_scores=[],
                summary=None,
            )

        system_prompt = compile_encoded_prompt(candidate)
        all_outputs: list[SingleModelALRolloutOutput] = []
        all_scores: list[float] = []
        all_trajectories: list[SingleModelALTrajectory] = []
        all_objective_scores: list[dict[str, float]] = []
        all_eval_run_ids: list[EvalRunIds] = []
        summary_rates: list[float] = []
        total_high_signal_entries = 0

        for al_data_inst in batch:
            deployment_ids = al_data_inst.get("deployment_ids", [])
            requested_entry_ids = al_data_inst.get("eval_entry_ids")
            target = resolve_eval_run_target(
                self.runner.evalcli,
                al_data_inst,
                bigquery_client=self.bigquery_client,
                bucket_type=self.objective.focused_bucket_type,
            )
            if target is None:
                summary_rates.append(0.0)
                continue
            eval_set_name = target.eval_set_name
            eval_set_version = target.eval_set_version
            run_label = target.run_label
            is_focused_eval = target.is_focused

            student_eval_id = al_data_inst.get("cached_student_eval_run_id")
            if student_eval_id:
                print(f"[Child cache HIT] Using cached student eval_id: {student_eval_id} ({run_label})")
            else:
                student_eval_id = self._get_or_run_student_eval(
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                    system_prompt=system_prompt,
                    run_label=run_label,
                )
            all_eval_run_ids.append(
                {
                    "eval_set_name": eval_set_name,
                    "eval_set_version": eval_set_version,
                    "student_eval_run_id": student_eval_id,
                }
            )
            if not is_focused_eval:
                evaluation_kind = "Trace evaluation" if capture_traces else "Validation"
                label = self.objective.pending_telemetry_label or self.objective.name
                print(
                    f"[{evaluation_kind}] Reading "
                    f"{'eval-set' if capture_traces else 'full-validation'} {label} results for "
                    f"{eval_set_name} {eval_set_version}: {student_eval_id}"
                )
            analysis = self._get_or_fetch_analysis(
                student_eval_id,
                include_error_examples=capture_traces and not is_focused_eval,
                include_per_entry=is_focused_eval or capture_traces,
            )
            if self.objective.is_pending(analysis):
                label = self.objective.pending_telemetry_label or self.objective.name
                raise self.objective.pending_error_type(
                    f"No {label} telemetry is available yet for eval {student_eval_id}; "
                    "refusing to score 0/0 as success"
                )
            if not is_focused_eval:
                self.objective.log_analysis(analysis)

            high_signal_entry_ids = self.objective.entry_ids_to_score(analysis, requested_entry_ids)
            if is_focused_eval:
                assert requested_entry_ids is not None
                summary_rates.append(self.objective.focused_pass_rate(analysis, requested_entry_ids))
                total_high_signal_entries += len(requested_entry_ids)
            else:
                summary_rates.append(self.objective.aggregate_score(analysis))
                total_high_signal_entries += len(high_signal_entry_ids)

            scored_rows = self.objective.scored_rows(
                analysis,
                al_data_inst=al_data_inst,
                student_eval_id=student_eval_id,
                eval_set_name=eval_set_name,
                eval_set_version=eval_set_version,
                deployment_ids=deployment_ids,
                requested_entry_ids=requested_entry_ids,
                is_focused_eval=is_focused_eval,
                capture_traces=capture_traces,
            )
            for row in scored_rows:
                output = cast(SingleModelALRolloutOutput, dict(row.output))
                objective_score = {**self.constant_scores, **row.dimension_scores}
                score = self.composite_score(objective_score)
                all_outputs.append(output)
                all_scores.append(score)
                all_objective_scores.append(objective_score)
                if is_focused_eval or capture_traces:
                    entry_data = {**al_data_inst, **row.data_overrides}
                    all_trajectories.append(
                        {
                            "data": cast(SingleModelALDataInst, entry_data),
                            "output": output,
                            "score": score,
                            "objective_scores": objective_score,
                        }
                    )

        summary = None
        if summary_rates:
            summary = {
                self.objective.name: sum(summary_rates) / len(summary_rates),
                "high_signal_entry_count": float(total_high_signal_entries),
            }

        return GleanEvaluationBatch(
            outputs=all_outputs,
            scores=all_scores,
            trajectories=all_trajectories,
            objective_scores=all_objective_scores,
            summary=summary,
            eval_run_ids=all_eval_run_ids,
        )


__all__ = [
    "SingleModelALDataInst",
    "SingleModelALRolloutOutput",
    "SingleModelALTrajectory",
    "SingleModelAdapter",
    "ShellToolTelemetryPendingError",
]
