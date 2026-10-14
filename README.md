# Hurricane Ian Flood Assessment Agent

An LLM-powered, tool-using agent for a retrospective assessment of coastal
flooding in Lee County, Florida, during Hurricane Ian. This project is being
implemented one milestone at a time for CGN 6905.

## Current status

Milestones 1–4 provide the project foundation, validated NOAA retrieval,
deterministic flood calculations and visualization, plus a stateful LangChain
tool-calling agent for UF NaviGator. The Streamlit interface is intentionally
not implemented yet.

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
- `src/hurricane_flood_agent/analysis.py` loads and integrity-checks the saved
  artifacts, aligns records by exact UTC timestamp, and calculates peak,
  residual, threshold-exceedance, and sigma summaries without an LLM.
- `src/hurricane_flood_agent/visualization.py` creates the observed-versus-
  predicted water-level figure with the saved NOS thresholds and recorded peak.
- `analyze_hurricane_ian.py` runs Milestone 3 entirely from local artifacts and
  writes a portable JSON result plus PNG visualization.
- `src/hurricane_flood_agent/agent_tools.py` exposes the four required LangChain
  tools, manages active artifact identifiers, validates cached inputs, and
  returns actionable prerequisite errors.
- `src/hurricane_flood_agent/agent.py` binds those tool schemas to NaviGator,
  executes requested tools, returns `ToolMessage` results to the model, and
  persists chronological redacted execution logs. It also rejects a proposed
  event-specific numerical answer when the required deterministic evidence has
  not succeeded; this output guard asks the same model to reconsider while
  retaining autonomous `tool_choice="auto"` selection.
- `run_flood_agent.py` provides a command-line entry point for one agent
  question without introducing a user-interface framework.
- `navigator_agent_integration_test.py` runs the two exact professor questions
  against the live NaviGator API.
- `tests/` contains offline synthetic unit tests that never call an external
  service.

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

This legacy smoke test validates tool *selection*, not execution. The Milestone
4 runner below performs the complete loop: it executes requested tools, adds
their results to the conversation, and lets the model produce an
evidence-grounded response.

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

Analyze the saved artifacts without making any NOAA or model call:

```bash
python analyze_hurricane_ian.py
```

This writes `analysis_results.json` and `water_levels.png` under
`artifacts/analysis/hurricane_ian_fort_myers/`. Threshold exceedance durations
are reported both as sample-count × six minutes and as the span between the
first and last qualifying samples, because exact crossing times between samples
are unknown. NOAA sigma is summarized as a measurement-variability indicator,
not interpreted as total measurement uncertainty.

Run the LangChain agent with automatic tool selection:

```bash
python run_flood_agent.py \
  "How severe was coastal flooding in Lee County, Florida, during Hurricane Ian?" \
  --show-log
```

The agent starts with no active artifact selected. Its tools register or create
artifacts as the model requests them. Analysis and plotting tools return
structured `missing_prerequisite` errors when their inputs are unavailable, so
the model can recover without a hard-coded workflow. Each run saves a redacted
JSON execution log under `artifacts/logs/` for later display by Streamlit. The
log includes the registered schema summaries, normalized provider tool-call
metadata, validated inputs, compact outputs, warnings, recovery events, and the
accepted final response; credentials and private reasoning are excluded.

Run deterministic offline tests separately from live provider tests:

```bash
pytest
python navigator_agent_integration_test.py
```

The first command uses model doubles and synthetic scientific data; it never
contacts UF NaviGator or NOAA. The second command makes real NaviGator model
calls and requires a currently authorized `NAVIGATOR_TOOLKIT_API_KEY`. Exact
Hurricane Ian NOAA artifacts are reused from the local cache unless the model
explicitly requests a fresh retrieval.

### Known Milestone 4 limitation

NaviGator may cite an authoritative source by name while omitting its literal
URL from the generated prose. The verified source URLs remain available in the
agent configuration, tool outputs, saved artifacts, and run log. Milestone 5
will render those links deterministically in the interface instead of relying
on model wording. This is a citation-presentation limitation; it does not alter
tool selection or the deterministic NOAA calculations.

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

Milestone 5 will add the Streamlit interface, complete end-to-end verification,
and prepare the submission materials. It is not stubbed prematurely so this
milestone remains independently reviewable.
