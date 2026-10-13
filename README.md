# Hurricane Ian Flood Assessment Agent

An LLM-powered, tool-using agent for a retrospective assessment of coastal
flooding in Lee County, Florida, during Hurricane Ian. This project is being
implemented one milestone at a time for CGN 6905.

## Current status

Milestones 1 and 2 provide the project foundation, UF NaviGator/LangChain model
integration, and independently runnable NOAA retrieval and validation. Flood
calculations, visualization, the full agent, and the Streamlit interface are
intentionally not implemented yet.

## Architecture at this milestone

- `src/hurricane_flood_agent/config.py` loads non-secret defaults and the API
  credential from the environment or a local `.env` file.
- `src/hurricane_flood_agent/llm.py` creates a LangChain `ChatOpenAI` wrapper
  pointed at UF NaviGator's OpenAI-compatible endpoint.
- `navigator_smoke_test.py` makes a live model call and checks autonomous tool
  selection. Its NOAA tool is only a harmless schema placeholder.
- `src/hurricane_flood_agent/noaa.py` calls NOAA with bounded retries and parses
  six-minute observations and predictions onto an explicit UTC time grid.
- `src/hurricane_flood_agent/thresholds.py` retrieves current NOAA/NOS flood
  thresholds and converts them from station datum to feet relative to MHHW.
- `src/hurricane_flood_agent/artifacts.py` saves raw responses, validated CSVs,
  metadata, warnings, source URLs, and SHA-256 checksums.
- `retrieve_hurricane_ian.py` reproduces the required September 27–30, 2022
  retrieval without invoking an LLM.
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

Retrieve the Hurricane Ian case-study data directly from NOAA:

```bash
python retrieve_hurricane_ian.py
```

This creates ignored local artifacts under `artifacts/`. The water-level
artifact contains the raw NOAA JSON, validated observation and prediction CSVs,
and metadata. The threshold artifact contains the source responses, values,
datum-conversion method, retrieval date, and retrospective-use caveat.

The retrieval requests six-minute observations and predictions in feet relative
to MHHW, with NOAA's `time_zone=gmt`; timestamps are stored as UTC. Missing
timestamps are inserted as rows with missing values so later plots preserve
gaps. Invalid observations are never imputed.

## NOAA sources

- [CO-OPS Data Retrieval API](https://api.tidesandcurrents.noaa.gov/api/prod/)
- [Fort Myers station metadata](https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/8725520.json)
- [Fort Myers coastal inundation thresholds](https://tidesandcurrents.noaa.gov/inundationdb/inundation.html?id=8725520)

## Model versus agent framework

The model (`mistral-small-3.1`) interprets a question and decides whether to
request a tool. LangChain is the agent framework: it supplies tool schemas,
normalizes model responses, executes tools in an agent loop, returns tool
results to the model, and exposes execution events. Keeping model construction
separate from scientific code will allow the deterministic NOAA and flood
analysis functions to be tested independently and reused with another
authorized model provider.

## Planned structure

Future milestones will add focused modules for deterministic flood analysis,
visualization, LangChain tools, execution logging, and the Streamlit interface.
They are not stubbed prematurely so each milestone remains independently
reviewable.
