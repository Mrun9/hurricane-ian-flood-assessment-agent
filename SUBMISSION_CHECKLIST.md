# Final Submission Checklist

## Completed in the repository

- [x] LangChain NaviGator agent with automatic tool selection.
- [x] Four deterministic, independently callable tools.
- [x] Validated NOAA observations, predictions, thresholds, and provenance workflow.
- [x] Peak, matching-timestamp residual, exceedance, data-quality, and sigma analysis.
- [x] Academic Matplotlib PNG in `submission/hurricane_ian_water_levels.png`.
- [x] Streamlit chat interface with real execution logs and downloads.
- [x] Flood-indicator cards and scientific limitations.
- [x] Deterministic NOAA/NHC source links that address model URL omission.
- [x] Offline unit, agent-loop, visualization, and Streamlit startup tests.
- [x] Two-minute demonstration script.
- [x] 200–300-word reflection draft.
- [x] Rubric-to-evidence mapping.
- [x] Credential and generated-file exclusions in `.gitignore`.

## Student actions before submission

- [ ] Activate `agentic-ai` and install the final editable project dependencies.
- [ ] Put the current NaviGator key in the ignored `.env` file or environment.
- [ ] Run `pytest -q` and save the passing summary.
- [ ] Run `python navigator_agent_integration_test.py` with the authorized key.
- [ ] Start `streamlit run streamlit_app.py` and rehearse both required questions.
- [ ] Confirm the application does not display an API key or `.env` contents.
- [ ] Record the demonstration using `VIDEO_DEMO_SCRIPT.md` in under two minutes.
- [ ] Review and personalize `REFLECTION_DRAFT.md`.
- [ ] Confirm the PNG, source code, video, and reflection are included in the submission.
- [ ] Push the final reviewed Milestone 5 commit to GitHub.

## Known limitation to disclose

Mistral Small 3.1 can name an authoritative source while omitting the literal URL
from its generated prose. The Streamlit evidence panel always renders the verified
NOAA and NHC links deterministically. This presentation safeguard does not alter
the LLM’s automatic tool selection or the deterministic flood calculations.
