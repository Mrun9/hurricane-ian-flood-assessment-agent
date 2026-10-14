# Hurricane Ian Flood Assessment Agent

An LLM-powered, tool-using agent for a retrospective assessment of coastal
flooding in Lee County, Florida, during Hurricane Ian. This project is being
implemented one milestone at a time for CGN 6905.

## Current status

Milestones 1–5 are implemented: validated NOAA retrieval, deterministic flood
calculations and visualization, a LangChain tool-calling agent for UF
NaviGator, a Streamlit interface, tests, and submission-support documents.

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
- `src/hurricane_flood_agent/agent_tools.py` exposes the four required NOAA
  LangChain tools plus three optional USGS spatial tools, manages active artifact
  identifiers, validates cached inputs, and returns actionable prerequisite errors.
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
- `src/hurricane_flood_agent/ui.py` converts real tool outputs and saved analysis
  artifacts into safe presentation values and provides verified source links.
- `streamlit_app.py` provides chat, actual execution logs, flood indicators,
  visualization and download controls, and scientific limitations.
- `tests/` contains offline synthetic unit tests that never call an external
  service.
- `submission/hurricane_ian_water_levels.png` is the required tracked PNG.
- `submission/hurricane_ian_lee_county_usgs_hwm_map.png` is the optional tracked
  USGS spatial-map fallback.
- `VIDEO_DEMO_SCRIPT.md`, `REFLECTION_DRAFT.md`, `RUBRIC_CHECKLIST.md`, and
  `SUBMISSION_CHECKLIST.md` support final review and submission.

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

## Run the Streamlit application

Prepare the deterministic local artifacts once, then start the app:

```bash
python retrieve_hurricane_ian.py
python analyze_hurricane_ian.py
streamlit run streamlit_app.py
```

The app uses five simple tabs: **Chat**, **Analysis**, **Visualization**, **Agent
activity**, and **Sources & limitations**. Chat includes the two exact professor
questions and an optional spatial question. Analysis shows deterministic NOAA and
USGS indicators; Visualization provides the time-series figure and USGS point map.
When the agent generates a figure during a response, it is also displayed inline
with that answer. Agent activity displays chronological model decisions, validated
tool inputs, concise structured outputs, warnings, and failures.

After each response, the visible transcript and structured run records are saved
under the ignored `artifacts/conversations/` directory. A conversation can also be
downloaded as JSON from the sidebar. Because Streamlit Community Cloud storage may
reset when an app restarts, the JSON download is the durable cloud option. API keys
and private model reasoning are never included in these records.

The validated indicator panel reports:

- peak recorded level: **7.257 ft MHHW at 2022-09-28 22:30 UTC**;
- maximum exact-time residual: **7.249 ft at 22:18 UTC**;
- classification: **major flooding** under current NOAA/NOS thresholds;
- **960 valid** six-minute observations and the NOAA sigma caveat.

For the final two-minute recording, follow `VIDEO_DEMO_SCRIPT.md`.

### Optional spatial extension

The optional extension retrieves surveyed USGS Hurricane Ian high-water marks for
Lee County, calculates a deterministic spatial summary, and produces a static point
map. The validated live response contains 257 marks at 243 sites, all reported in
feet relative to NAVD88. These values remain separate from the NOAA MHHW series.
Numerical comparison requires an appropriate datum transformation, and local
inundation-depth interpretation still requires compatible ground or infrastructure
elevation data.

### Known Milestone 4 limitation

NaviGator may cite an authoritative source by name while omitting its literal
URL from the generated prose. The Streamlit **Evidence & limitations** panel
therefore renders the verified NOAA and NHC links deterministically rather than
relying on model wording. This is a citation-presentation limitation; it does
not alter tool selection or the deterministic NOAA calculations.

## NOAA sources

- [CO-OPS Data Retrieval API](https://api.tidesandcurrents.noaa.gov/api/prod/)
- [Fort Myers station metadata](https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/8725520.json)
- [Fort Myers coastal inundation thresholds](https://tidesandcurrents.noaa.gov/inundationdb/inundation.html?id=8725520)

## USGS spatial sources

- [Hurricane Ian Flood Event Viewer](https://apps.usgs.gov/fev/event/2022-ian)
- [STN filtered high-water-mark service](https://stn.wim.usgs.gov/STNServices/Documentation/HWM/FilteredHWMs)

## Model versus agent framework

The model (`mistral-small-3.1`) interprets a question and decides whether to
request a tool. LangChain is the agent framework: it supplies tool schemas,
normalizes model responses, executes tools in an agent loop, returns tool
results to the model, and exposes execution events. Keeping model construction
separate from scientific code will allow the deterministic NOAA and flood
analysis functions to be tested independently and reused with another
authorized model provider.

## Submission artifacts

- Required PNG: `submission/hurricane_ian_water_levels.png`
- Video plan: `VIDEO_DEMO_SCRIPT.md`
- Reflection draft: `REFLECTION_DRAFT.md`
- Rubric mapping: `RUBRIC_CHECKLIST.md`
- Final manual checks: `SUBMISSION_CHECKLIST.md`
