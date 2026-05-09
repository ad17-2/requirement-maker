from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from requirement_maker.generate import GenerationConfig, generate_structured_json


PRODUCT_SIGNAL_CATEGORIES = (
    "decisions",
    "functional_requirements",
    "non_functional_requirements",
    "risks",
    "constraints",
    "open_questions",
    "out_of_scope",
    "task_candidates",
)


@dataclass(frozen=True)
class SourceRef:
    chunk_id: str
    snippet: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], valid_chunk_ids: set[str]) -> SourceRef:
        chunk_id = _required_str(data, "chunk_id")
        if chunk_id not in valid_chunk_ids:
            raise ValueError(f"source_refs references unknown chunk_id: {chunk_id}")
        snippet = _required_str(data, "snippet")
        return cls(chunk_id=chunk_id, snippet=snippet)

    def to_dict(self) -> dict[str, str]:
        return {"chunk_id": self.chunk_id, "snippet": self.snippet}


@dataclass(frozen=True)
class TranscriptChunk:
    id: str
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "id": self.id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class PlanUnit:
    id: str
    focus: str
    source_chunk_ids: list[str]
    status: str = "pending"
    skip_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], valid_chunk_ids: set[str]) -> PlanUnit:
        unit_id = _required_str(data, "id")
        focus = _required_str(data, "focus")
        source_chunk_ids = _required_str_list(data, "source_chunk_ids")
        unknown = set(source_chunk_ids) - valid_chunk_ids
        if unknown:
            raise ValueError(f"plan unit {unit_id} references unknown chunks: {sorted(unknown)}")
        return cls(id=unit_id, focus=focus, source_chunk_ids=source_chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "focus": self.focus,
            "source_chunk_ids": self.source_chunk_ids,
            "status": self.status,
        }
        if self.skip_reason:
            data["skip_reason"] = self.skip_reason
        return data


@dataclass
class WorkflowPlan:
    units: list[PlanUnit]

    @classmethod
    def from_dict(cls, data: dict[str, Any], valid_chunk_ids: set[str]) -> WorkflowPlan:
        raw_units = data.get("units")
        if not isinstance(raw_units, list) or not raw_units:
            raise ValueError("planner response must include a non-empty units list")
        return cls(
            units=[
                PlanUnit.from_dict(_dict_item(unit, "planner unit"), valid_chunk_ids)
                for unit in raw_units
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {"units": [unit.to_dict() for unit in self.units]}


@dataclass
class StructuredItem:
    id: str
    category: str
    title: str
    summary: str
    source_refs: list[SourceRef]
    source_unit_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        item_id: str,
        category: str,
        unit_id: str,
        valid_chunk_ids: set[str],
    ) -> StructuredItem:
        title = _required_str(data, "title")
        summary = _required_str(data, "summary")
        raw_refs = data.get("source_refs")
        if not isinstance(raw_refs, list) or not raw_refs:
            raise ValueError(f"{category} item {title!r} must include source_refs")
        source_refs = [
            SourceRef.from_dict(_dict_item(ref, "source ref"), valid_chunk_ids)
            for ref in raw_refs
        ]
        return cls(
            id=item_id,
            category=category,
            title=title,
            summary=summary,
            source_refs=source_refs,
            source_unit_ids=[unit_id],
        )

    def merge_sources_from(self, other: StructuredItem) -> None:
        existing_refs = {(ref.chunk_id, ref.snippet) for ref in self.source_refs}
        for ref in other.source_refs:
            key = (ref.chunk_id, ref.snippet)
            if key not in existing_refs:
                self.source_refs.append(ref)
                existing_refs.add(key)
        for unit_id in other.source_unit_ids:
            if unit_id not in self.source_unit_ids:
                self.source_unit_ids.append(unit_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "summary": self.summary,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "source_unit_ids": self.source_unit_ids,
        }


@dataclass
class ExtractionState:
    items: list[StructuredItem]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit_id: str,
        valid_chunk_ids: set[str],
        id_allocator: StableIdAllocator,
    ) -> ExtractionState:
        items: list[StructuredItem] = []
        for category in PRODUCT_SIGNAL_CATEGORIES:
            raw_items = data.get(category)
            if raw_items is None:
                raise ValueError(f"extraction response missing required category: {category}")
            if not isinstance(raw_items, list):
                raise ValueError(f"extraction category {category} must be a list")
            for raw_item in raw_items:
                item_id = id_allocator.next(category)
                items.append(
                    StructuredItem.from_dict(
                        _dict_item(raw_item, category),
                        item_id=item_id,
                        category=category,
                        unit_id=unit_id,
                        valid_chunk_ids=valid_chunk_ids,
                    )
                )
        return cls(items=items)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in PRODUCT_SIGNAL_CATEGORIES}
        for item in self.items:
            grouped[item.category].append(item.to_dict())
        return grouped


@dataclass
class CriticFinding:
    id: str
    severity: str
    target_id: str
    message: str
    action: str
    resolution: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriticFinding:
        return cls(
            id=_required_str(data, "id"),
            severity=_required_str(data, "severity"),
            target_id=_required_str(data, "target_id"),
            message=_required_str(data, "message"),
            action=_required_str(data, "action"),
            resolution=_required_str(data, "resolution"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "severity": self.severity,
            "target_id": self.target_id,
            "message": self.message,
            "action": self.action,
            "resolution": self.resolution,
        }


@dataclass
class CriticReview:
    findings: list[CriticFinding]
    remove_item_ids: list[str]
    add_open_questions: list[StructuredItem]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        valid_chunk_ids: set[str],
        id_allocator: StableIdAllocator,
    ) -> CriticReview:
        raw_findings = data.get("findings", [])
        if not isinstance(raw_findings, list):
            raise ValueError("critic findings must be a list")
        raw_remove = data.get("remove_item_ids", [])
        if not isinstance(raw_remove, list) or not all(isinstance(item, str) for item in raw_remove):
            raise ValueError("critic remove_item_ids must be a string list")
        raw_open_questions = data.get("add_open_questions", [])
        if not isinstance(raw_open_questions, list):
            raise ValueError("critic add_open_questions must be a list")
        add_open_questions = [
            StructuredItem.from_dict(
                _dict_item(item, "critic open question"),
                item_id=id_allocator.next("open_questions"),
                category="open_questions",
                unit_id="critic",
                valid_chunk_ids=valid_chunk_ids,
            )
            for item in raw_open_questions
        ]
        return cls(
            findings=[CriticFinding.from_dict(_dict_item(item, "critic finding")) for item in raw_findings],
            remove_item_ids=list(raw_remove),
            add_open_questions=add_open_questions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "remove_item_ids": self.remove_item_ids,
            "add_open_questions": [item.to_dict() for item in self.add_open_questions],
        }


@dataclass
class FinalWorkflowState:
    plan: WorkflowPlan
    chunks: list[TranscriptChunk]
    decisions: list[StructuredItem]
    requirements: list[StructuredItem]
    non_functional_requirements: list[StructuredItem]
    risks: list[StructuredItem]
    constraints: list[StructuredItem]
    open_questions: list[StructuredItem]
    out_of_scope: list[StructuredItem]
    task_candidates: list[StructuredItem]
    critic_findings: list[CriticFinding]
    conflicts: list[StructuredItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "workflow-state-v1",
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "plan": self.plan.to_dict(),
            "decisions": [item.to_dict() for item in self.decisions],
            "requirements": [item.to_dict() for item in self.requirements],
            "non_functional_requirements": [
                item.to_dict() for item in self.non_functional_requirements
            ],
            "risks": [item.to_dict() for item in self.risks],
            "constraints": [item.to_dict() for item in self.constraints],
            "open_questions": [item.to_dict() for item in self.open_questions],
            "out_of_scope": [item.to_dict() for item in self.out_of_scope],
            "task_candidates": [item.to_dict() for item in self.task_candidates],
            "critic_findings": [finding.to_dict() for finding in self.critic_findings],
            "conflicts": [item.to_dict() for item in self.conflicts],
        }


@dataclass(frozen=True)
class TraceEntry:
    id: str
    stage: str
    role: str
    status: str
    input_artifact_ids: list[str]
    output_artifact_ids: list[str]
    provider: dict[str, str | int]
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "role": self.role,
            "status": self.status,
            "input_artifact_ids": self.input_artifact_ids,
            "output_artifact_ids": self.output_artifact_ids,
            "provider": self.provider,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkflowTrace:
    entries: list[TraceEntry] = field(default_factory=list)
    final_artifacts: dict[str, str] = field(default_factory=dict)

    def add(
        self,
        *,
        stage: str,
        role: str,
        input_artifact_ids: list[str],
        output_artifact_ids: list[str],
        provider: dict[str, str | int],
        status: str = "success",
        retry_count: int = 0,
    ) -> None:
        self.entries.append(
            TraceEntry(
                id=f"trace-{len(self.entries) + 1:04d}",
                stage=stage,
                role=role,
                status=status,
                input_artifact_ids=input_artifact_ids,
                output_artifact_ids=output_artifact_ids,
                provider=provider,
                retry_count=retry_count,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "workflow-trace-v1",
            "entries": [entry.to_dict() for entry in self.entries],
            "final_artifacts": self.final_artifacts,
        }


@dataclass(frozen=True)
class WorkflowConfig:
    model: str
    timeout: float = 60.0
    retries: int = 2


@dataclass(frozen=True)
class WorkflowProviderRequest:
    stage: str
    payload: str
    model: str
    call_id: str
    unit_id: str | None = None


class WorkflowProvider(Protocol):
    def complete_json(self, request: WorkflowProviderRequest) -> dict[str, Any]:
        ...

    def write_markdown(self, request: WorkflowProviderRequest) -> str:
        ...


@dataclass
class WorkflowResult:
    markdown: str
    final_state: FinalWorkflowState
    trace: WorkflowTrace


class AnthropicWorkflowProvider:
    def __init__(self, api_key: str, config: WorkflowConfig) -> None:
        self.api_key = api_key
        self.config = config

    def complete_json(self, request: WorkflowProviderRequest) -> dict[str, Any]:
        return generate_structured_json(
            request.payload,
            self.api_key,
            GenerationConfig(
                model=self.config.model,
                timeout=self.config.timeout,
                retries=self.config.retries,
            ),
        )

    def write_markdown(self, request: WorkflowProviderRequest) -> str:
        state = json.loads(request.payload)
        return generate_structured_markdown(
            state,
            self.api_key,
            GenerationConfig(
                model=self.config.model,
                timeout=self.config.timeout,
                retries=self.config.retries,
            ),
        )


class StableIdAllocator:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def next(self, category: str) -> str:
        prefixes = {
            "decisions": "dec",
            "functional_requirements": "req",
            "non_functional_requirements": "nfr",
            "risks": "risk",
            "constraints": "con",
            "open_questions": "q",
            "out_of_scope": "scope",
            "task_candidates": "task",
        }
        prefix = prefixes[category]
        self._counts[category] = self._counts.get(category, 0) + 1
        return f"{prefix}-{self._counts[category]:04d}"


class AgenticWorkflow:
    def __init__(self, provider: WorkflowProvider, config: WorkflowConfig) -> None:
        self.provider = provider
        self.config = config
        self.trace = WorkflowTrace()
        self.id_allocator = StableIdAllocator()
        self._call_count = 0

    def run(self, transcript: str) -> WorkflowResult:
        chunks = chunk_transcript(transcript)
        valid_chunk_ids = {chunk.id for chunk in chunks}
        plan = self._plan(chunks, valid_chunk_ids)
        extractions = [
            self._extract(unit, chunks, valid_chunk_ids)
            for unit in plan.units
            if self._should_process(unit)
        ]
        merged_items, conflicts = self._merge(extractions)
        review = self._critic(plan, chunks, merged_items, conflicts, valid_chunk_ids)
        final_items = self._apply_review(merged_items, conflicts, review)
        final_state = self._build_final_state(plan, chunks, final_items, conflicts, review.findings)
        self._assert_final_constraints(final_state)
        markdown = self._write(final_state)
        return WorkflowResult(markdown=markdown, final_state=final_state, trace=self.trace)

    def _next_call_id(self, stage: str) -> str:
        self._call_count += 1
        return f"mockable-{stage}-{self._call_count:04d}"

    def _provider_meta(self, call_id: str) -> dict[str, str | int]:
        return {"provider": "Anthropic", "model": self.config.model, "call_id": call_id}

    def _request(self, *, stage: str, payload: dict[str, Any], unit_id: str | None = None) -> WorkflowProviderRequest:
        call_id = self._next_call_id(stage)
        return WorkflowProviderRequest(
            stage=stage,
            payload=json.dumps(payload, sort_keys=True),
            model=self.config.model,
            call_id=call_id,
            unit_id=unit_id,
        )

    def _plan(self, chunks: list[TranscriptChunk], valid_chunk_ids: set[str]) -> WorkflowPlan:
        request = self._request(
            stage="planner",
            payload={
                "instruction": "Create extraction units before any extraction. Return JSON with units.",
                "required_unit_fields": ["id", "focus", "source_chunk_ids"],
                "chunks": [chunk.to_dict() for chunk in chunks],
            },
        )
        data = self.provider.complete_json(request)
        plan = WorkflowPlan.from_dict(data, valid_chunk_ids)
        self.trace.add(
            stage="planning",
            role="planner",
            input_artifact_ids=[chunk.id for chunk in chunks],
            output_artifact_ids=[unit.id for unit in plan.units],
            provider=self._provider_meta(request.call_id),
        )
        return plan

    def _should_process(self, unit: PlanUnit) -> bool:
        if not unit.source_chunk_ids:
            unit.status = "skipped"
            unit.skip_reason = "Planner unit did not reference any source chunks."
            self.trace.add(
                stage="extraction",
                role="extractor",
                input_artifact_ids=[unit.id],
                output_artifact_ids=[],
                provider={"provider": "local", "model": "none", "call_id": f"skip-{unit.id}"},
                status="skipped",
            )
            return False
        return True

    def _extract(
        self,
        unit: PlanUnit,
        chunks: list[TranscriptChunk],
        valid_chunk_ids: set[str],
    ) -> ExtractionState:
        selected_chunks = [chunk for chunk in chunks if chunk.id in set(unit.source_chunk_ids)]
        request = self._request(
            stage="extractor",
            unit_id=unit.id,
            payload={
                "instruction": (
                    "Extract all required product signals with source_refs. "
                    "Return every category even when empty."
                ),
                "required_categories": list(PRODUCT_SIGNAL_CATEGORIES),
                "unit": unit.to_dict(),
                "chunks": [chunk.to_dict() for chunk in selected_chunks],
            },
        )
        data = self.provider.complete_json(request)
        extraction = ExtractionState.from_dict(
            data,
            unit_id=unit.id,
            valid_chunk_ids=valid_chunk_ids,
            id_allocator=self.id_allocator,
        )
        unit.status = "processed"
        self.trace.add(
            stage="extraction",
            role="extractor",
            input_artifact_ids=[unit.id, *unit.source_chunk_ids],
            output_artifact_ids=[item.id for item in extraction.items],
            provider=self._provider_meta(request.call_id),
        )
        return extraction

    def _merge(self, extractions: list[ExtractionState]) -> tuple[list[StructuredItem], list[StructuredItem]]:
        merged: list[StructuredItem] = []
        conflicts: list[StructuredItem] = []
        by_key: dict[tuple[str, str], StructuredItem] = {}
        for extraction in extractions:
            for item in extraction.items:
                key = (item.category, _normalise_title(item.title))
                existing = by_key.get(key)
                if existing is None:
                    by_key[key] = item
                    merged.append(item)
                    continue
                if _normalise_text(existing.summary) == _normalise_text(item.summary):
                    existing.merge_sources_from(item)
                else:
                    conflict = StructuredItem(
                        id=self.id_allocator.next("open_questions"),
                        category="open_questions",
                        title=f"Conflict: {existing.title}",
                        summary=(
                            "Conflicting transcript evidence was found: "
                            f"{existing.summary} / {item.summary}"
                        ),
                        source_refs=[*existing.source_refs],
                        source_unit_ids=[*existing.source_unit_ids],
                    )
                    conflict.merge_sources_from(item)
                    conflicts.append(conflict)
        self.trace.add(
            stage="merge_dedupe",
            role="local",
            input_artifact_ids=[item.id for extraction in extractions for item in extraction.items],
            output_artifact_ids=[item.id for item in merged] + [item.id for item in conflicts],
            provider={"provider": "local", "model": "deterministic", "call_id": "merge-dedupe-0001"},
        )
        return merged, conflicts

    def _critic(
        self,
        plan: WorkflowPlan,
        chunks: list[TranscriptChunk],
        merged_items: list[StructuredItem],
        conflicts: list[StructuredItem],
        valid_chunk_ids: set[str],
    ) -> CriticReview:
        request = self._request(
            stage="critic",
            payload={
                "instruction": (
                    "Review merged structured state for missing traceability, unsupported claims, "
                    "duplicates, contradictions, and weak acceptance criteria. Return corrections."
                ),
                "plan": plan.to_dict(),
                "chunks": [chunk.to_dict() for chunk in chunks],
                "items": [item.to_dict() for item in merged_items],
                "conflicts": [item.to_dict() for item in conflicts],
            },
        )
        data = self.provider.complete_json(request)
        review = CriticReview.from_dict(
            data,
            valid_chunk_ids=valid_chunk_ids,
            id_allocator=self.id_allocator,
        )
        self.trace.add(
            stage="critic",
            role="critic",
            input_artifact_ids=[item.id for item in merged_items] + [item.id for item in conflicts],
            output_artifact_ids=[finding.id for finding in review.findings]
            + [item.id for item in review.add_open_questions],
            provider=self._provider_meta(request.call_id),
        )
        return review

    def _apply_review(
        self,
        merged_items: list[StructuredItem],
        conflicts: list[StructuredItem],
        review: CriticReview,
    ) -> list[StructuredItem]:
        removed = set(review.remove_item_ids)
        final_items = [item for item in merged_items if item.id not in removed]
        final_items.extend(conflicts)
        final_items.extend(review.add_open_questions)
        return final_items

    def _build_final_state(
        self,
        plan: WorkflowPlan,
        chunks: list[TranscriptChunk],
        items: list[StructuredItem],
        conflicts: list[StructuredItem],
        findings: list[CriticFinding],
    ) -> FinalWorkflowState:
        return FinalWorkflowState(
            plan=plan,
            chunks=chunks,
            decisions=_items_by_category(items, "decisions"),
            requirements=_items_by_category(items, "functional_requirements"),
            non_functional_requirements=_items_by_category(items, "non_functional_requirements"),
            risks=_items_by_category(items, "risks"),
            constraints=_items_by_category(items, "constraints"),
            open_questions=_items_by_category(items, "open_questions"),
            out_of_scope=_items_by_category(items, "out_of_scope"),
            task_candidates=_items_by_category(items, "task_candidates"),
            critic_findings=findings,
            conflicts=conflicts,
        )

    def _assert_final_constraints(self, state: FinalWorkflowState) -> None:
        all_items = (
            state.decisions
            + state.requirements
            + state.non_functional_requirements
            + state.risks
            + state.constraints
            + state.open_questions
            + state.out_of_scope
            + state.task_candidates
        )
        ids = [item.id for item in all_items]
        if len(ids) != len(set(ids)):
            raise ValueError("final state contains duplicate stable IDs")
        for unit in state.plan.units:
            if unit.status not in {"processed", "skipped"}:
                raise ValueError(f"planner unit {unit.id} was neither processed nor skipped")
            if unit.status == "skipped" and not unit.skip_reason:
                raise ValueError(f"planner unit {unit.id} skipped without a reason")
        for item in all_items:
            if not item.source_refs:
                raise ValueError(f"final item {item.id} lacks source traceability")

    def _write(self, final_state: FinalWorkflowState) -> str:
        request = self._request(
            stage="writer",
            payload={
                "instruction": (
                    "Write final Markdown only from this reviewed structured state. "
                    "Do not introduce unsupported requirement IDs or claims."
                ),
                **final_state.to_dict(),
            },
        )
        markdown = self.provider.write_markdown(request)
        allowed_ids = _final_item_ids(final_state)
        unknown_ids = {
            token
            for token in _extract_id_tokens(markdown)
            if token not in allowed_ids and token not in {"workflow-state-v1", "workflow-trace-v1"}
        }
        if unknown_ids:
            raise ValueError(f"writer produced unsupported stable IDs: {sorted(unknown_ids)}")
        self.trace.add(
            stage="writer",
            role="writer",
            input_artifact_ids=sorted(allowed_ids),
            output_artifact_ids=["markdown"],
            provider=self._provider_meta(request.call_id),
        )
        return markdown


def make_workflow_provider(api_key: str, config: WorkflowConfig) -> WorkflowProvider:
    return AnthropicWorkflowProvider(api_key, config)


def run_agentic_workflow(
    transcript: str,
    provider: WorkflowProvider,
    config: WorkflowConfig,
) -> WorkflowResult:
    return AgenticWorkflow(provider, config).run(transcript)


def chunk_transcript(transcript: str) -> list[TranscriptChunk]:
    text = transcript.strip()
    if not text:
        raise ValueError("transcript is empty")
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not parts:
        parts = [text]
    chunks: list[TranscriptChunk] = []
    cursor = 0
    for index, part in enumerate(parts, start=1):
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        chunks.append(
            TranscriptChunk(
                id=f"chunk-{index:04d}",
                text=part,
                start_char=start,
                end_char=end,
            )
        )
        cursor = end
    return chunks


def generate_structured_markdown(
    state: dict[str, Any],
    api_key: str,
    config: GenerationConfig,
) -> str:
    prompt = (
        "Write a polished requirements document from this reviewed structured JSON state. "
        "Preserve stable IDs next to each requirement, decision, question, and task. "
        "Do not invent unsupported claims.\n\n"
        f"{json.dumps(state, sort_keys=True)}"
    )
    from requirement_maker.generate import generate_requirements

    return generate_requirements(prompt, api_key, config)


def _items_by_category(items: list[StructuredItem], category: str) -> list[StructuredItem]:
    return [item for item in items if item.category == category]


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid string field: {key}")
    return value


def _required_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"missing or invalid string list field: {key}")
    return value


def _dict_item(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _normalise_title(value: str) -> str:
    return " ".join(value.lower().split())


def _normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def _final_item_ids(state: FinalWorkflowState) -> set[str]:
    items = (
        state.decisions
        + state.requirements
        + state.non_functional_requirements
        + state.risks
        + state.constraints
        + state.open_questions
        + state.out_of_scope
        + state.task_candidates
    )
    return {item.id for item in items} | {finding.id for finding in state.critic_findings}


def _extract_id_tokens(markdown: str) -> set[str]:
    prefixes = ("dec", "req", "nfr", "risk", "con", "q", "scope", "task", "finding")
    tokens: set[str] = set()
    for raw in markdown.replace("(", " ").replace(")", " ").replace(",", " ").split():
        token = raw.strip("[]:;,.`*_")
        if "-" not in token:
            continue
        prefix, suffix = token.rsplit("-", 1)
        if prefix in prefixes and suffix.isdigit():
            tokens.add(token)
    return tokens
