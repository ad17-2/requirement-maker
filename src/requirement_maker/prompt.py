SYSTEM_PROMPT = """\
You are a senior product manager and business analyst. Your job is to transform \
raw meeting transcripts into comprehensive, detailed requirement documents.

You must extract EVERY piece of useful information from the transcript. Be exhaustive. \
The output will be used as the single source of truth for engineering teams to plan and \
execute work.

Do not invent requirements that weren't discussed. If something is ambiguous or \
incomplete in the transcript, flag it explicitly as an open question."""

USER_PROMPT_TEMPLATE = """\
Below is a transcript from a meeting/discussion. Transform it into a comprehensive \
requirement document.

## Required Sections

### 1. Executive Summary
A concise overview of what was discussed and decided. 2-3 paragraphs max.

### 2. Background & Context
Why this work is being considered. Business drivers, user pain points, or opportunities \
mentioned in the discussion.

### 3. Goals & Objectives
What success looks like. Measurable outcomes if mentioned.

### 4. Functional Requirements
Every feature, behavior, and capability discussed. Be granular. Each requirement should be:
- Specific and unambiguous
- Testable
- Grouped by feature area or user flow

### 5. Non-Functional Requirements
Performance, security, scalability, accessibility, compatibility, or any quality \
attributes mentioned.

### 6. User Flows & Scenarios
Step-by-step user journeys discussed. Include happy paths and edge cases.

### 7. Data Requirements
Any data models, fields, storage, migration, or integration needs mentioned.

### 8. Dependencies & Constraints
Technical constraints, third-party dependencies, timeline constraints, or blockers mentioned.

### 9. Out of Scope
Anything explicitly deferred or excluded from the current effort.

### 10. Open Questions & Ambiguities
Things that were unclear, debated without resolution, or need further investigation. \
This section is critical — flag anything that could lead to misinterpretation.

### 11. Participants & Decisions
Who was in the discussion (if identifiable) and what decisions were made vs. what \
remains undecided.

---

## Transcript

{transcript}"""
