import json
from pathlib import Path

from click.testing import CliRunner

from requirement_maker import cli
from requirement_maker.workflow import (
    AgenticWorkflow,
    WorkflowConfig,
    WorkflowProvider,
    WorkflowProviderRequest,
    run_agentic_workflow,
)


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
