# Two-Minute Video Demonstration

## Before recording

1. Activate the environment: `conda activate agentic-ai`.
2. Confirm the valid NaviGator key is available through `.env` or the environment.
3. Generate the local deterministic artifacts if needed:
   `python retrieve_hurricane_ian.py` and `python analyze_hurricane_ian.py`.
4. Start the interface: `streamlit run streamlit_app.py`.
5. Confirm the indicator cards and visualization are visible before recording.
6. Clear the chat so the demonstration begins from a clean agent state.

## Suggested recording sequence

**0:00–0:10 — Purpose**

Show the title and state: “This is a retrospective Hurricane Ian flood-assessment
agent for NOAA Fort Myers station 8725520. Python performs the scientific
calculations, while NaviGator selects tools.”

**0:10–1:05 — Required Question 1**

Click **Run Question 1**. While it runs, explain that the trace is generated from
the real LangChain execution. Open **Agent execution** and briefly show:

- `retrieve_noaa_data` with station, dates, datum, and units;
- `get_flood_thresholds` with station 8725520;
- `calculate_flood_indicators` with artifact identifiers;
- the returned peak, matching-timestamp residual, quality summary, and warnings.

Show the final response and indicator cards. Open **Visualization** and point out
the observed/predicted curves, flood thresholds, UTC axis, and recorded peak.

**1:05–1:40 — Required Question 2**

Click **Run Question 2**. Show that the model answers the conceptual limitation
directly without an unnecessary NOAA calculation tool. Emphasize that gauge
elevation and residual are not local street inundation depth.

**1:40–1:55 — Evidence and limitations**

Open **Evidence & limitations**. Show the deterministic NOAA/NHC links and note
that one gauge cannot represent all of Lee County. Mention that current flood
thresholds are applied retrospectively.

**1:55–2:00 — Close**

State: “The run logs and PNG can be downloaded, and no API credential is displayed.”

## Recording checklist

- [ ] Recording is no longer than two minutes.
- [ ] Both professor-required questions are shown exactly.
- [ ] Actual tool names and validated inputs are visible.
- [ ] Concise tool results and numerical indicators are visible.
- [ ] Final responses are visible.
- [ ] The PNG visualization is visible.
- [ ] Sources and scientific limitations are visible.
- [ ] No `.env`, API key, terminal history, or private credential is visible.
