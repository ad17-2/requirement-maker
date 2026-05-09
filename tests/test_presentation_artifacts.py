import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from requirement_maker import cli
from requirement_maker.workflow import (
    CriticFinding,
    FinalWorkflowState,
    PlanUnit,
    SourceRef,
    StructuredItem,
    WorkflowPlan,
    WorkflowTrace,
    render_json_export,
    render_markdown_document,
    render_task_export,
)


def sample_state() -> FinalWorkflowState:
    ref = SourceRef(chunk_id="chunk-0001", snippet="/Users/alice/project secret sk-test open issue")
    return FinalWorkflowState(
        plan=WorkflowPlan([PlanUnit(id="unit-1", focus="Checkout", source_chunk_ids=["chunk-0001"], status="processed")]),
        chunks=[],
        decisions=[StructuredItem("dec-0001", "decisions", "Use Stripe", "Use Stripe for payments.", [ref])],
        requirements=[
            StructuredItem("req-0001", "functional_requirements", "Checkout payment", "Shopper can pay and receive confirmation.", [ref])
        ],
        non_functional_requirements=[
            StructuredItem("nfr-0001", "non_functional_requirements", "Payment latency", "Payment confirmation should complete quickly.", [ref])
        ],
        risks=[StructuredItem("risk-0001", "risks", "Payment failure", "Provider downtime can block checkout.", [ref])],
        constraints=[StructuredItem("con-0001", "constraints", "PCI scope", "Do not store card details.", [ref])],
        open_questions=[StructuredItem("q-0001", "open_questions", "Refund owner", "Refund workflow ownership remains unresolved.", [ref])],
        out_of_scope=[StructuredItem("scope-0001", "out_of_scope", "Crypto", "Crypto payments are deferred.", [ref])],
        task_candidates=[
            StructuredItem("task-0001", "task_candidates", "Build checkout", "Implement payment creation and confirmation handling.", [ref])
        ],
        critic_findings=[CriticFinding("finding-001", "medium", "req-0001", "Needs acceptance criteria.", "add_validation", "applied")],
        conflicts=[],
    )


def test_deterministic_markdown_json_and_task_exports_are_safe_and_source_linked() -> None:
    state = sample_state()

    markdown = render_markdown_document(state)
    json_export = render_json_export(state)
    task_export = render_task_export(state)

    required_headings = [
        "Executive Summary",
        "Background & Context",
        "Goals & Objectives",
        "Functional Requirements",
        "Non-Functional Requirements",
        "User Flows & Scenarios",
        "Data Requirements",
        "Dependencies & Constraints",
        "Out of Scope",
        "Open Questions & Ambiguities",
        "Participants & Decisions",
        "Action & Task Candidates",
    ]
    assert all(f"## {heading}" in markdown for heading in required_headings)
    assert "req-0001" in markdown
    assert "Acceptance criteria" in markdown
    assert "/Users/" not in markdown
    assert "sk-test" not in markdown

    parsed = json.loads(json_export)
    assert parsed["schema_version"] == "requirements-export-v1"
    assert parsed["requirements"][0]["id"] == "req-0001"
    assert parsed["requirements"][0]["source_refs"][0]["chunk_id"] == "chunk-0001"
    assert "task_candidates" in parsed
    assert "/Users/" not in json_export
    assert "sk-test" not in json_export

    tasks = json.loads(task_export)
    assert tasks["schema_version"] == "task-handoff-v1"
    assert tasks["tasks"][0]["id"] == "task-0001"
    assert tasks["tasks"][0]["acceptance_criteria"]
    assert tasks["tasks"][0]["source_refs"][0]["chunk_id"] == "chunk-0001"


def test_cli_writes_requested_exports_manifest_and_honors_force(monkeypatch) -> None:
    runner = CliRunner()

    async def fake_prepare_audio(input_file: Path, temp_dir: Path) -> list[Path]:
        return [input_file]

    async def fake_transcribe_chunks(audio_paths, openai_key, on_chunk_done, config):  # noqa: ANN001, ANN202
        return "Transcript"

    def fake_run_agentic_workflow(transcript: str, provider, config):  # noqa: ANN001
        return SimpleNamespace(markdown="provider markdown", final_state=sample_state(), trace=WorkflowTrace())

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "prepare_audio", fake_prepare_audio)
    monkeypatch.setattr(cli, "transcribe_chunks", fake_transcribe_chunks)
    monkeypatch.setattr(cli, "run_agentic_workflow", fake_run_agentic_workflow)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")
        result = runner.invoke(cli.main, ["meeting.mp3", "--json", "--tasks"])

        assert result.exit_code == 0
        expected = {
            "meeting_requirements.md",
            "meeting_requirements.json",
            "meeting_requirements.tasks.json",
            "meeting_requirements.trace.json",
            "meeting_requirements.manifest.json",
        }
        assert expected == {path.name for path in Path(".").glob("meeting_requirements*")}
        manifest = json.loads(Path("meeting_requirements.manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "artifact-manifest-v1"
        listed = {artifact["kind"]: artifact["path"] for artifact in manifest["artifacts"]}
        assert set(listed) == {"markdown", "json", "tasks", "trace", "manifest"}
        assert all(Path(path).exists() for path in listed.values())
        assert all(not Path(path).is_absolute() for path in listed.values())
        assert "Manifest:" in result.output

        protected = runner.invoke(cli.main, ["meeting.mp3", "--json", "--tasks"])
        assert protected.exit_code != 0
        assert "Output already exists" in protected.output

        forced = runner.invoke(cli.main, ["meeting.mp3", "--json", "--tasks", "--force"])
        assert forced.exit_code == 0
