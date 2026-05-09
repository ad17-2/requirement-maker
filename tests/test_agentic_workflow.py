import json
from pathlib import Path

from click.testing import CliRunner

from requirement_maker import cli
from requirement_maker.workflow import (
    AgenticWorkflow,
    WorkflowConfig,
    WorkflowProvider,
    WorkflowProviderRequest,
    chunk_transcript,
    run_agentic_workflow,
)
from requirement_maker.provider_errors import ProviderError, ProviderErrorKind, ProviderStage


class FixtureProvider(WorkflowProvider):
    def __init__(self) -> None:
        self.requests: list[WorkflowProviderRequest] = []

    def complete_json(self, request: WorkflowProviderRequest) -> dict:
        self.requests.append(request)
        if request.stage == "planner":
            return {
                "units": [
                    {
                        "id": "unit-auth",
                        "focus": "Authentication",
                        "source_chunk_ids": ["chunk-0001"],
                    },
                    {
                        "id": "unit-billing",
                        "focus": "Billing",
                        "source_chunk_ids": ["chunk-0002"],
                    },
                ]
            }
        if request.stage == "extractor" and request.unit_id == "unit-auth":
            return {
                "decisions": [
                    {
                        "title": "Use email login",
                        "summary": "Users sign in with email links.",
                        "source_refs": [{"chunk_id": "chunk-0001", "snippet": "email links"}],
                    }
                ],
                "functional_requirements": [
                    {
                        "title": "Passwordless login",
                        "summary": "Send a secure email link when a user requests sign-in.",
                        "source_refs": [{"chunk_id": "chunk-0001", "snippet": "secure email link"}],
                    }
                ],
                "non_functional_requirements": [
                    {
                        "title": "Login link expiry",
                        "summary": "Login links expire within 15 minutes.",
                        "source_refs": [{"chunk_id": "chunk-0001", "snippet": "15 minutes"}],
                    }
                ],
                "risks": [
                    {
                        "title": "Email deliverability",
                        "summary": "Users may not receive login links.",
                        "source_refs": [{"chunk_id": "chunk-0001", "snippet": "deliverability"}],
                    }
                ],
                "constraints": [
                    {
                        "title": "No passwords",
                        "summary": "The first version must avoid password storage.",
                        "source_refs": [{"chunk_id": "chunk-0001", "snippet": "avoid passwords"}],
                    }
                ],
                "open_questions": [
                    {
                        "title": "Admin login",
                        "summary": "Whether admins use the same login method is unresolved.",
                        "source_refs": [{"chunk_id": "chunk-0001", "snippet": "admins"}],
                    }
                ],
                "out_of_scope": [
                    {
                        "title": "Social login",
                        "summary": "OAuth providers are deferred.",
                        "source_refs": [{"chunk_id": "chunk-0001", "snippet": "deferred"}],
                    }
                ],
                "task_candidates": [
                    {
                        "title": "Implement email link login",
                        "summary": "Build link generation, delivery, and verification.",
                        "source_refs": [{"chunk_id": "chunk-0001", "snippet": "email link"}],
                    }
                ],
            }
        if request.stage == "extractor" and request.unit_id == "unit-billing":
            return {
                "decisions": [],
                "functional_requirements": [
                    {
                        "title": "Passwordless login",
                        "summary": "Send a secure email link when a user requests sign-in.",
                        "source_refs": [{"chunk_id": "chunk-0002", "snippet": "same login"}],
                    },
                    {
                        "title": "Billing export",
                        "summary": "Finance can export invoice CSV files.",
                        "source_refs": [{"chunk_id": "chunk-0002", "snippet": "invoice CSV"}],
                    },
                    {
                        "title": "Passwordless login",
                        "summary": "Require passwords for returning users.",
                        "source_refs": [{"chunk_id": "chunk-0002", "snippet": "require passwords"}],
                    },
                ],
                "non_functional_requirements": [],
                "risks": [],
                "constraints": [],
                "open_questions": [],
                "out_of_scope": [],
                "task_candidates": [],
            }
        if request.stage == "critic":
            return {
                "findings": [
                    {
                        "id": "finding-001",
                        "severity": "high",
                        "target_id": "req-0001",
                        "message": "Duplicate login requirements were merged.",
                        "action": "merge_duplicate",
                        "resolution": "applied",
                    }
                ],
                "remove_item_ids": ["req-0004"],
                "add_open_questions": [
                    {
                        "title": "Resolve login contradiction",
                        "summary": "Transcript conflicts on passwordless versus password-required login.",
                        "source_refs": [
                            {"chunk_id": "chunk-0001", "snippet": "secure email link"},
                            {"chunk_id": "chunk-0002", "snippet": "require passwords"},
                        ],
                    }
                ],
            }
        raise AssertionError(f"unexpected request {request}")

    def write_markdown(self, request: WorkflowProviderRequest) -> str:
        self.requests.append(request)
        payload = json.loads(request.payload)
        ids = [item["id"] for item in payload["requirements"]]
        return "# Requirements\n\n" + "\n".join(f"- {item_id}" for item_id in ids)


def test_workflow_enforces_stage_order_traceability_and_critic_corrections() -> None:
    provider = FixtureProvider()

    result = run_agentic_workflow(
        "Authentication discussion.\n\nBilling discussion.",
        provider,
        WorkflowConfig(model="claude-test"),
    )

    stages = [request.stage for request in provider.requests]
    assert stages == ["planner", "extractor", "extractor", "critic", "writer"]
    assert [request.unit_id for request in provider.requests if request.stage == "extractor"] == [
        "unit-auth",
        "unit-billing",
    ]
    assert [entry.stage for entry in result.trace.entries] == [
        "planning",
        "extraction",
        "extraction",
        "merge_dedupe",
        "critic",
        "writer",
    ]
    assert all(entry.status == "success" for entry in result.trace.entries)
    assert all(unit.status == "processed" for unit in result.final_state.plan.units)

    requirement_ids = {item.id for item in result.final_state.requirements}
    assert requirement_ids == {"req-0001", "req-0003"}
    login = next(item for item in result.final_state.requirements if item.id == "req-0001")
    assert {ref.chunk_id for ref in login.source_refs} == {"chunk-0001", "chunk-0002"}
    assert all(item.source_refs for item in result.final_state.decisions)
    assert all(item.source_refs for item in result.final_state.requirements)
    assert any("contradiction" in item.title.lower() for item in result.final_state.open_questions)
    assert "req-0004" not in result.markdown

    trace = result.trace.to_dict()
    assert trace["entries"][0]["provider"]["model"] == "claude-test"
    assert trace["entries"][-1]["output_artifact_ids"] == ["markdown"]


def test_invalid_extraction_schema_fails_without_final_artifact() -> None:
    class BadProvider(FixtureProvider):
        def complete_json(self, request: WorkflowProviderRequest) -> dict:
            if request.stage == "planner":
                return {"units": [{"id": "unit-1", "focus": "A", "source_chunk_ids": ["chunk-0001"]}]}
            return {
                "decisions": [],
                "functional_requirements": [{"title": "Missing source refs", "summary": "bad"}],
                "non_functional_requirements": [],
                "risks": [],
                "constraints": [],
                "open_questions": [],
                "out_of_scope": [],
                "task_candidates": [],
            }

    workflow = AgenticWorkflow(BadProvider(), WorkflowConfig(model="claude-test"))

    try:
        workflow.run("Transcript")
    except ValueError as exc:
        assert "source_refs" in str(exc)
    else:
        raise AssertionError("invalid extraction schema should fail")


def test_cli_exposes_agentic_workflow_and_writes_trace(monkeypatch) -> None:
    runner = CliRunner()
    provider = FixtureProvider()

    async def fake_prepare_audio(input_file: Path, temp_dir: Path) -> list[Path]:
        return [input_file]

    async def fake_transcribe_chunks(audio_paths, openai_key, on_chunk_done, config):  # noqa: ANN001, ANN202
        on_chunk_done(1, 1)
        return "Authentication discussion.\n\nBilling discussion."

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "prepare_audio", fake_prepare_audio)
    monkeypatch.setattr(cli, "transcribe_chunks", fake_transcribe_chunks)
    monkeypatch.setattr(cli, "make_workflow_provider", lambda api_key, config: provider)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")

        result = runner.invoke(cli.main, ["meeting.mp3", "--verbose"])

        assert result.exit_code == 0
        assert Path("meeting_requirements.md").exists()
        trace_path = Path("meeting_requirements.trace.json")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))

    assert "Planning extraction units..." in result.output
    assert "Running structured extraction..." in result.output
    assert "Merging and deduplicating structured state..." in result.output
    assert "Running critic review..." in result.output
    assert "Writing final requirements..." in result.output
    assert "Trace:" in result.output
    assert [entry["stage"] for entry in trace["entries"]] == [
        "planning",
        "extraction",
        "extraction",
        "merge_dedupe",
        "critic",
        "writer",
    ]
    assert all("call_id" in entry["provider"] for entry in trace["entries"] if entry["role"] != "local")


def test_transcript_chunking_is_deterministic_complete_and_budgeted() -> None:
    transcript = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

    first = chunk_transcript(transcript, max_chunk_chars=18)
    second = chunk_transcript(transcript, max_chunk_chars=18)

    assert [chunk.to_dict() for chunk in first] == [chunk.to_dict() for chunk in second]
    assert [chunk.id for chunk in first] == ["chunk-0001", "chunk-0002", "chunk-0003", "chunk-0004"]
    assert all(len(chunk.text) <= 18 for chunk in first)
    assert "".join(transcript[chunk.start_char:chunk.end_char] for chunk in first) == transcript
    assert [(chunk.start_char, chunk.end_char) for chunk in first] == [
        (0, 17),
        (17, 31),
        (31, 46),
        (46, 63),
    ]


def test_long_transcript_requests_respect_configured_budget_and_cover_every_chunk() -> None:
    class BudgetProvider(WorkflowProvider):
        def __init__(self) -> None:
            self.requests: list[WorkflowProviderRequest] = []

        def complete_json(self, request: WorkflowProviderRequest) -> dict:
            self.requests.append(request)
            assert len(request.payload) <= 4000
            if request.stage == "planner":
                chunks = json.loads(request.payload)["chunks"]
                return {
                    "units": [
                        {"id": f"unit-{chunk['id']}", "focus": chunk["id"], "source_chunk_ids": [chunk["id"]]}
                        for chunk in chunks
                    ]
                }
            if request.stage == "extractor":
                chunk_id = json.loads(request.payload)["chunks"][0]["id"]
                return {
                    "decisions": [],
                    "functional_requirements": [
                        {
                            "title": f"Requirement for {chunk_id}",
                            "summary": f"Keep content from {chunk_id}.",
                            "source_refs": [{"chunk_id": chunk_id, "snippet": chunk_id}],
                        }
                    ],
                    "non_functional_requirements": [],
                    "risks": [],
                    "constraints": [],
                    "open_questions": [],
                    "out_of_scope": [],
                    "task_candidates": [],
                }
            if request.stage == "critic":
                return {"findings": [], "remove_item_ids": [], "add_open_questions": []}
            raise AssertionError(request.stage)

        def write_markdown(self, request: WorkflowProviderRequest) -> str:
            self.requests.append(request)
            assert len(request.payload) <= 4000
            payload = json.loads(request.payload)
            return "\n".join(item["id"] for item in payload["requirements"])

    provider = BudgetProvider()
    transcript = "\n\n".join(f"Topic {index} requires ordered feature {index}." for index in range(1, 7))

    result = run_agentic_workflow(
        transcript,
        provider,
        WorkflowConfig(model="claude-test", transcript_chunk_chars=60, request_budget_chars=4000),
    )

    assert [chunk.id for chunk in result.final_state.chunks] == [f"chunk-{index:04d}" for index in range(1, 7)]
    assert {ref.chunk_id for item in result.final_state.requirements for ref in item.source_refs} == {
        chunk.id for chunk in result.final_state.chunks
    }
    assert [request.stage for request in provider.requests].count("extractor") == 6


def test_transient_provider_failures_retry_and_permanent_failures_do_not_retry() -> None:
    class RetryProvider(FixtureProvider):
        def __init__(self) -> None:
            super().__init__()
            self.planner_attempts = 0

        def complete_json(self, request: WorkflowProviderRequest) -> dict:
            if request.stage == "planner":
                self.requests.append(request)
                self.planner_attempts += 1
                if self.planner_attempts == 1:
                    raise ProviderError(
                        stage=ProviderStage.REQUIREMENT_GENERATION,
                        kind=ProviderErrorKind.RATE_LIMIT,
                        provider="Anthropic",
                        detail="rate limit",
                        transient=True,
                    )
                return {"units": [{"id": "unit-auth", "focus": "Only", "source_chunk_ids": ["chunk-0001"]}]}
            if request.stage == "critic":
                self.requests.append(request)
                return {"findings": [], "remove_item_ids": [], "add_open_questions": []}
            return super().complete_json(request)

    provider = RetryProvider()
    result = run_agentic_workflow("Authentication discussion.", provider, WorkflowConfig(model="claude-test", retries=1))

    assert result.trace.entries[0].stage == "planning"
    assert result.trace.entries[0].retry_count == 1
    assert provider.planner_attempts == 2

    class PermanentProvider(RetryProvider):
        def complete_json(self, request: WorkflowProviderRequest) -> dict:
            self.requests.append(request)
            raise ProviderError(
                stage=ProviderStage.REQUIREMENT_GENERATION,
                kind=ProviderErrorKind.INVALID_REQUEST,
                provider="Anthropic",
                detail="bad model",
                transient=False,
            )

    permanent = PermanentProvider()
    try:
        run_agentic_workflow("Authentication discussion.", permanent, WorkflowConfig(model="claude-test", retries=3))
    except ProviderError as exc:
        assert exc.kind is ProviderErrorKind.INVALID_REQUEST
    else:
        raise AssertionError("permanent failures must fail")
    assert len(permanent.requests) == 1


def test_malformed_responses_repair_with_trace_and_fail_after_limit() -> None:
    class RepairProvider(FixtureProvider):
        def __init__(self, always_bad: bool = False) -> None:
            super().__init__()
            self.always_bad = always_bad
            self.extractor_attempts = 0

        def complete_json(self, request: WorkflowProviderRequest) -> dict:
            if request.stage == "planner":
                self.requests.append(request)
                return {"units": [{"id": "unit-auth", "focus": "Authentication", "source_chunk_ids": ["chunk-0001"]}]}
            if request.stage == "extractor":
                self.requests.append(request)
                self.extractor_attempts += 1
                if self.always_bad or self.extractor_attempts == 1:
                    return {"decisions": []}
                return FixtureProvider.complete_json(self, request)
            if request.stage == "critic":
                self.requests.append(request)
                return {"findings": [], "remove_item_ids": [], "add_open_questions": []}
            return FixtureProvider.complete_json(self, request)

    repaired = RepairProvider()
    result = run_agentic_workflow(
        "Authentication discussion.",
        repaired,
        WorkflowConfig(model="claude-test", repair_attempts=1),
    )

    assert repaired.extractor_attempts == 2
    assert any(entry.status == "repair" and entry.stage == "extraction" for entry in result.trace.entries)

    broken = RepairProvider(always_bad=True)
    try:
        run_agentic_workflow(
            "Authentication discussion.",
            broken,
            WorkflowConfig(model="claude-test", repair_attempts=1),
        )
    except ValueError as exc:
        assert "exhausted repair attempts" in str(exc)
    else:
        raise AssertionError("malformed responses must fail after repair limit")
    assert broken.extractor_attempts == 2
