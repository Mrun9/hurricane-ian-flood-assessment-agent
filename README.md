# Hurricane Ian Flood Assessment Agent

An LLM-powered, tool-using agent for a retrospective assessment of coastal
flooding in Lee County, Florida, during Hurricane Ian. This project is being
implemented one milestone at a time for CGN 6905.

## Current status

Milestone 1 provides the project foundation and UF NaviGator/LangChain model
integration. NOAA retrieval, flood calculations, the full agent, and the
Streamlit interface are intentionally not implemented yet.

## Architecture at this milestone

- `src/hurricane_flood_agent/config.py` loads non-secret defaults and the API
  credential from the environment or a local `.env` file.
- `src/hurricane_flood_agent/llm.py` creates a LangChain `ChatOpenAI` wrapper
  pointed at UF NaviGator's OpenAI-compatible endpoint.
- `navigator_smoke_test.py` makes a live model call and checks autonomous tool
  selection. Its NOAA tool is only a harmless schema placeholder.
- `tests/` contains offline tests that never call an external service.

## Setup with Conda

```bash
conda activate agentic-ai
python -m pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` locally and replace the placeholder with a newly generated UF
NaviGator key. Never commit `.env` or paste the key into source code, chat, a
demo recording, or test output.

## Validate the foundation

Run deterministic offline tests:

```bash
pytest
```

Run the live LangChain connectivity and tool-selection check:

```bash
python navigator_smoke_test.py
```

Expected behavior:

1. A request for station `8725520` selects `retrieve_noaa_data` with the station
   ID as a structured argument.
2. A conceptual tide-gauge question receives a direct answer without a tool.

The live test validates tool *selection*, not tool execution. LangChain exposes
the model's requested calls through `AIMessage.tool_calls`; a later agent will
execute real tools, add their results to the conversation, and let the model
produce an evidence-grounded response.

## Model versus agent framework

The model (`mistral-small-3.1`) interprets a question and decides whether to
request a tool. LangChain is the agent framework: it supplies tool schemas,
normalizes model responses, executes tools in an agent loop, returns tool
results to the model, and exposes execution events. Keeping model construction
separate from scientific code will allow the deterministic NOAA and flood
analysis functions to be tested independently and reused with another
authorized model provider.

## Planned structure

Future milestones will add focused modules for NOAA retrieval, artifact
storage, deterministic flood analysis, flood thresholds, visualization,
LangChain tools, execution logging, and the Streamlit interface. They are not
stubbed prematurely so that each milestone remains independently reviewable.
