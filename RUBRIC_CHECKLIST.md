# Rubric Checklist

Status meanings: **Complete** means implemented and validated; **Partial** means an
identified limitation remains; **Manual** means the student must perform the final
submission action.

## Agent architecture and tool integration — 30 points

| Requirement | Status | Evidence |
|---|---|---|
| LangChain agent using UF NaviGator | Complete | `src/hurricane_flood_agent/llm.py`, `src/hurricane_flood_agent/agent.py` |
| Four distinct required tools | Complete | `agent_tools.py`, `tests/test_agent.py`; three optional USGS tools are separate |
| Automatic LLM tool selection | Complete | `tool_choice="auto"` in `agent.py`; offline model-loop tests; `navigator_agent_integration_test.py` |
| Multiple tool calls and tool-result feedback | Complete | `FloodAssessmentAgent.run`; chronological tool-call/result tests |
| Prerequisite errors and recovery | Complete | `missing_prerequisite` tool responses and visualization recovery test |
| Actual structured execution logs | Complete | `ExecutionLogger`; Streamlit **Agent activity** tab; downloadable JSON |
| Step limit and credential redaction | Complete | `max_model_steps`, `_redact`, redaction tests |

## Accuracy of data retrieval and analysis — 30 points

| Requirement | Status | Evidence |
|---|---|---|
| Fort Myers station 8725520, Sep 27–30, 2022 | Complete | Saved NOAA metadata; `retrieve_hurricane_ian.py` |
| 960 observed and 960 predicted six-minute records | Complete | Validated metadata and Milestone 2 verification |
| Feet relative to MHHW; UTC | Complete | Artifact request metadata, plot labels, UI case-study caption |
| Exact-timestamp residual calculation | Complete | `analysis.py`; mismatched-timestamp tests |
| Peak 7.257 ft at 2022-09-28 22:30 UTC | Complete | Analysis artifact, tests, indicator cards |
| Maximum residual 7.249 ft at 22:18 UTC | Complete | Analysis artifact, tests, indicator cards |
| NOAA/NOS thresholds 1.69/2.66/3.89 ft MHHW | Complete | Threshold artifact conversion and tests |
| Quality flags, missing values, sigma caveat | Complete | Analysis artifact, tool output, UI notes |

## Visualization and flood interpretation — 20 points

| Requirement | Status | Evidence |
|---|---|---|
| Observed-versus-predicted Matplotlib PNG | Complete | `visualization.py`, `submission/hurricane_ian_water_levels.png` |
| Thresholds, peak, labels, UTC, datum, source | Complete | Packaged PNG and visualization tests |
| Major-flood classification | Complete | Deterministic threshold comparison and UI metric |
| Elevation/residual/inundation distinction | Complete | System prompt and Streamlit limitations panel |
| Geographic and transportation limitations | Complete | Prompt, deterministic limitations panel, required Question 2 |
| Additional authoritative Ian evidence | Complete | NHC report and update links in deterministic sources panel |
| Literal URLs in model-generated prose | Partial | Mistral may omit a literal URL; Streamlit always renders verified links separately |

## Video demonstration — 10 points

| Requirement | Status | Evidence |
|---|---|---|
| Compact demo-ready interface | Complete | `streamlit_app.py`; offline AppTest and visual QA |
| Exact two questions available | Complete | **Chat** tab buttons using constants in `ui.py` |
| Logs, inputs, summaries, results, plot visible | Complete | Streamlit execution, metric, and visualization panels |
| Optional USGS spatial extension | Complete | `spatial.py`; three spatial tools; Visualization tab; deterministic tests and live 257-mark verification |
| Final recording under two minutes | Manual | Follow `VIDEO_DEMO_SCRIPT.md` and record the video |

## Critical reflection — 10 points

| Requirement | Status | Evidence |
|---|---|---|
| 200–300-word architecture reflection | Complete | `REFLECTION_DRAFT.md` |
| Strengths, limitations, improvement | Complete | `REFLECTION_DRAFT.md` |
| Student review and personalization | Manual | Review the draft before submission |

## Final verification

- [x] Offline test suite passes without NOAA or NaviGator network calls.
- [x] Streamlit application starts under `streamlit.testing.v1.AppTest`.
- [x] Local Streamlit server was opened and visually checked.
- [x] PNG exists and is independently downloadable.
- [x] `.env`, generated logs, raw artifacts, caches, and API keys are ignored.
- [ ] Rerun the two live NaviGator questions with the student’s authorized key.
- [ ] Record and submit the final video.
- [ ] Review and personalize the reflection.
