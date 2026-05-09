.DEFAULT_GOAL := help

VENV_BIN := .venv/bin
PYTHON := $(VENV_BIN)/python
CLI := $(VENV_BIN)/requirement-maker

OUTPUT_ARG := $(if $(OUTPUT),--output "$(OUTPUT)",)

.PHONY: help req req-json req-tasks req-demo doctor test lint typecheck check

define require_cli
	@if [ ! -x "$(CLI)" ]; then \
		echo "error: $(CLI) not found or not executable."; \
		echo "Run: python3 -m venv .venv && .venv/bin/python -m pip install -e ."; \
		exit 1; \
	fi
endef

define require_python
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "error: $(PYTHON) not found or not executable."; \
		echo "Run: python3 -m venv .venv && .venv/bin/python -m pip install -e ."; \
		exit 1; \
	fi
endef

define require_input
	@if [ -z "$(INPUT)" ]; then \
		echo "error: INPUT is required. Example: make $@ INPUT=meeting.mp4"; \
		exit 1; \
	fi
endef

help:
	@printf "%s\n" \
		"requirement-maker targets:" \
		"  make req INPUT=meeting.mp4 [OUTPUT=spec.md]       Generate Markdown requirements" \
		"  make req-json INPUT=meeting.mp4 [OUTPUT=spec.md]  Generate Markdown plus JSON export" \
		"  make req-tasks INPUT=meeting.mp4 [OUTPUT=spec.md] Generate Markdown plus task handoff export" \
		"  make req-demo                                    Generate a tiny demo.wav and demo artifacts" \
		"  make doctor                                      Run local diagnostics" \
		"  make test                                        Run pytest" \
		"  make lint                                        Run ruff checks" \
		"  make typecheck                                   Run mypy over src" \
		"  make check                                       Run test, lint, and typecheck"

req:
	$(call require_cli)
	$(call require_input)
	"$(CLI)" "$(INPUT)" $(OUTPUT_ARG)

req-json:
	$(call require_cli)
	$(call require_input)
	"$(CLI)" "$(INPUT)" $(OUTPUT_ARG) --json

req-tasks:
	$(call require_cli)
	$(call require_input)
	"$(CLI)" "$(INPUT)" $(OUTPUT_ARG) --tasks

req-demo:
	$(call require_cli)
	ffmpeg -hide_banner -loglevel error -f lavfi -i "sine=frequency=1000:duration=1" -ar 16000 -ac 1 demo.wav -y
	"$(CLI)" demo.wav --output demo_requirements.md --json --tasks --verbose --model claude-haiku-4-5-20251001 --concurrency 1 --timeout 60 --retries 1 --force

doctor:
	$(call require_cli)
	"$(CLI)" --doctor

test:
	$(call require_python)
	"$(PYTHON)" -m pytest -q

lint:
	$(call require_python)
	"$(PYTHON)" -m ruff check .

typecheck:
	$(call require_python)
	"$(PYTHON)" -m mypy src

check: test lint typecheck
