# Agent Harness — `app/harness/`

The harness is the infrastructure for **running and testing agents against defined tasks**. It is not an evaluation module — that lives in `app/eval/`. The harness answers: "Did the agent complete the goal?"

**Module:** 5 | **API prefix:** `/harness`

---

## What an Agent Harness Is

An evaluation module scores a static (question, context, answer) triple.
An agent harness **executes** an agent, captures everything it does, and scores the outcome.

```
Task (goal + success criteria)
    │
    ▼
AgentExecutor → runs agent, captures full trajectory
    │             (every tool call: name, inputs, outputs)
    ▼
TaskScorer → scores three dimensions:
    │   outcome    (0-1) did the response meet success criteria?  [LLM judge]
    │   tool_use   (0-1) did it call the expected tools?          [deterministic]
    │   efficiency (0-1) was the step count reasonable?           [deterministic]
    ▼
HarnessReport → pass_rate, avg_score, GREEN / YELLOW / RED status
```

---

## File Map

| File | What it does |
|------|-------------|
| `tasks.py` | `Task`, `TaskResult`, `TaskSuite`, `default_suite()` |
| `executor.py` | `AgentExecutor` — runs the agent, records full trajectory per step |
| `scorer.py` | `TaskScorer` — scores outcome + tool use + efficiency |
| `runner.py` | `HarnessRunner` — runs a full suite, returns `HarnessReport` |

---

## Trajectory — Why It Matters

The trajectory is what makes the harness useful for debugging. Instead of:
> "The answer was wrong."

You get:
> "Step 2: `calculator({'expression': '15 * 84'})` → `1260` — should have been `0.15 * 840`."

Every `TaskResult` includes:
```json
"trajectory": [
  {"step": 1, "tool_name": "calculator", "tool_input": {"expression": "15 * 84"}, "tool_output": "1260"},
  ...
]
```

---

## Tasks (`tasks.py`)

A `Task` defines what the agent should do and what success looks like:

```python
Task.create(
    description="Compound interest calculation",
    input="If I invest $5,000 at 7% for 3 years (compounded), what do I have?",
    expected_tools=["calculator"],
    success_criteria="Answer is close to $6,125.22. Must use the calculator.",
    tags=["math", "finance"],
    difficulty="medium",
)
```

`default_suite()` ships 5 tasks covering single-tool, multi-tool, multi-step, and error-handling scenarios. Extend it with domain-specific tasks for your project.

---

## Scoring Dimensions (`scorer.py`)

| Dimension | Method | Weight | What it checks |
|-----------|--------|--------|---------------|
| Outcome | LLM judge | 60% | Did the response meet the success criteria? |
| Tool use | Deterministic | 25% | Did it call every expected tool? (partial credit) |
| Efficiency | Deterministic | 15% | Steps used vs expected — penalises excessive looping |

**Pass threshold:** default 0.70. Any task below this is `FAIL`.

---

## API Endpoints

```
GET  /harness/tasks         List tasks in the default suite (shows IDs)
POST /harness/run/task      Run one task by ID — full trajectory + score breakdown
POST /harness/run/suite     Run the full suite — returns HarnessReport
POST /harness/run/custom    Define and run a one-off task inline
```

**Quick test:**
```bash
# 1. List tasks to get an ID
GET /harness/tasks

# 2. Run one task
POST /harness/run/task  {"task_id": "<id from step 1>"}

# 3. Run the full suite
POST /harness/run/suite
```

**Report status:**
- `GREEN` — 100% pass rate
- `YELLOW` — 70-99% pass rate
- `RED` — below 70% (regression signal — investigate before deploying)

---

## Student TODOs

1. **Add domain tasks** — write 5+ tasks specific to your capstone project. Domain-specific tasks are a better regression signal than generic math tasks.
2. **Parallelise the runner** — tasks are independent; run them with `asyncio.gather()` or `ThreadPoolExecutor` to speed up large suites.
3. **Persist reports** — save `HarnessReport.to_dict()` to JSON after each run so you can compare across code changes.
4. **CI integration** — add a step to `.github/workflows/ci.yml` that runs `POST /harness/run/suite` and fails if `status == "RED"`.
5. **Score per tag** — break down pass rates by tag ("math", "multi-tool") to identify which capabilities are weakest.

---

## Relationship to `app/eval/`

| `app/eval/` | `app/harness/` |
|------------|----------------|
| Scores static `(question, context, answer)` triples | Runs agents against tasks and scores outcomes |
| RAG quality metrics: faithfulness, relevance | Task completion: did the agent achieve the goal? |
| No agent execution | Full trajectory capture |
| Input: you supply the answer | Input: the harness generates the answer by running the agent |
