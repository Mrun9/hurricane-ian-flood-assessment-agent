"""Stateful LangChain tool-calling loop with transparent execution logs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from hurricane_flood_agent.agent_tools import AgentSessionState, create_flood_tools
from hurricane_flood_agent.llm import create_chat_model
from hurricane_flood_agent.spatial import USGS_EVENT_URL, USGS_HWM_URL


NHC_REPORT_URL = "https://www.nhc.noaa.gov/data/tcr/AL092022_Ian.pdf"
NHC_UPDATE_URL = (
    "https://www.nhc.noaa.gov/archive/2022/al09/al092022.update.09282257.shtml"
)
NOAA_RESPONSE_HELP_URL = (
    "https://api.tidesandcurrents.noaa.gov/api/prod/responseHelp.html"
)


SYSTEM_PROMPT = f"""You are an evidence-based coastal flood assessment assistant.

You have seven deterministic Python tools and autonomously decide which ones to call
and in what order. There is no mandatory fixed sequence. Respect each tool's stated
prerequisites: if a tool returns a missing_prerequisite error, use the suggested tools
to obtain the missing artifact and then retry the blocked operation. Tool selection is
automatic, so answer purely conceptual questions directly when no event-specific data
or calculation is needed.

The validated Hurricane Ian study scope is NOAA Fort Myers station 8725520, from
2022-09-27 through 2022-09-30 UTC, in English units relative to MHHW. Exact cached
artifacts may be reused by the tools. For any question asking about the actual event's
severity, peak, residual, thresholds, timestamps, station measurements, or a plot:
- Call the available tools before answering. A severity assessment requires verified
  station data, verified thresholds, and deterministic flood calculations.
- Never supply a station ID, measurement, threshold, timestamp, duration, residual,
  or flood category from pretrained memory.
- Do not treat a URL printed in this prompt as evidence for a numerical gauge result.
  Numerical claims must come from successful tool outputs in this conversation.
- Use generate_visualization when the user asks for a figure, chart, graph, or plot.
- For a broad quantitative flood-severity assessment, autonomously select
  generate_visualization after the calculation so the evidence-based answer is
  accompanied by the appropriate time-series graph. Do not generate plots for
  conceptual questions that do not need event data.

Tool roles:
- retrieve_noaa_data establishes the validated observations and predicted tides.
- get_flood_thresholds establishes authoritative station-specific thresholds.
- calculate_flood_indicators produces the peak, matching-timestamp residual,
  exceedances, severity category, quality summary, and sigma summary.
- generate_visualization creates a PNG only after a completed analysis exists.
- retrieve_usgs_high_water_marks establishes official surveyed Hurricane Ian marks
  for Lee County from the USGS Short-Term Network.
- analyze_spatial_flooding summarizes spatial coverage, mark elevations, environments,
  and quality while preserving datum limitations.
- generate_spatial_visualization creates a point map only after spatial analysis.

Scientific rules:
- For numerical or historical Hurricane Ian conclusions, use tool results; never
  calculate or invent measurements yourself. If tools fail, report the limitation
  without substituting remembered numbers.
- Observed water-surface elevation is a gauge elevation relative to a datum.
- Residual is observed minus predicted astronomical tide at the same timestamp.
- Inundation depth is water depth above local ground or infrastructure elevation.
  Never call gauge elevation or residual a street-level inundation depth.
- One Fort Myers gauge cannot represent every location in Fort Myers Beach, Sanibel,
  or Lee County because elevations, surge, waves, drainage, and distance vary.
- Discuss transportation only as plausible exposure of low-lying roads, bridges,
  causeways, and evacuation routes unless a cited source directly documents an impact.
- State UTC, feet, and MHHW wherever relevant. Mention data quality and the
  retrospective nature of current NOS thresholds.
- Sigma is NOAA's variability measure for the one-second samples used to form a
  six-minute value, not automatically total measurement uncertainty.
- USGS high-water-mark elevations use NAVD88 in the validated Lee County artifact.
  Never directly compare them with NOAA MHHW gauge values without a defensible,
  location-specific datum conversion.
- A high-water mark is a point observation. Do not interpolate a continuous flood
  surface or claim a road/building inundation depth without local ground or
  infrastructure elevation and compatible datum information.

Independent authoritative evidence available for citation:
- Official National Hurricane Center Hurricane Ian report: {NHC_REPORT_URL}
  It documents severe southwest Florida storm-surge impacts, including the Fort Myers
  NOS gauge record. Use it as independent context, not as a substitute for tool results.
- Contemporaneous NHC update: {NHC_UPDATE_URL}
  It documented the Fort Myers-area NOS station during the event.
- NOAA data-field definitions: {NOAA_RESPONSE_HELP_URL}
- USGS Hurricane Ian Flood Event Viewer: {USGS_EVENT_URL}
- USGS high-water-mark REST source: {USGS_HWM_URL}

For a broad request assessing Hurricane Ian flood severity, produce approximately
170-230 words plus a compact Sources line. Address severity, peak, maximum residual,
transportation implications, spatial limits, and the elevation/residual/inundation
distinction. Cite real source URLs. If evidence is unavailable, say the assessment is
incomplete. Do not expose hidden reasoning or claim that the model executed tools;
the application records tool execution separately.
"""


@dataclass(frozen=True)
class ExecutionLogEntry:
    """One observable event from the actual model/tool execution loop."""

    sequence: int
    timestamp_utc: str
    event_type: str
    status: str
    details: dict[str, Any]


@dataclass
class ExecutionLogger:
    """Chronological, serializable run log that excludes credentials and reasoning."""

    entries: list[ExecutionLogEntry] = field(default_factory=list)

    def record(
        self,
        event_type: str,
        *,
        status: str = "ok",
        **details: Any,
    ) -> None:
        self.entries.append(
            ExecutionLogEntry(
                sequence=len(self.entries) + 1,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                status=status,
                details=_redact(details),
            )
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self.entries]


@dataclass(frozen=True)
class AgentRunResult:
    """Final response and inspectable evidence of how it was produced."""

    run_id: str
    question: str
    response: str
    status: str
    tool_names: tuple[str, ...]
    execution_log: tuple[dict[str, Any], ...]
    state: dict[str, Any]
    log_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FloodAssessmentAgent:
    """Minimal LangChain agent loop for automatic, iterative tool calling."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        state: AgentSessionState | None = None,
        tools: Sequence[BaseTool] | None = None,
        max_model_steps: int = 10,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        if max_model_steps <= 0:
            raise ValueError("max_model_steps must be positive.")
        self.state = state or AgentSessionState()
        self.tools = list(tools) if tools is not None else create_flood_tools(self.state)
        self.tool_map = {tool.name: tool for tool in self.tools}
        if len(self.tool_map) != len(self.tools):
            raise ValueError("Tool names must be unique.")
        self.model = model or create_chat_model()
        self.model_with_tools = self.model.bind_tools(self.tools, tool_choice="auto")
        self.max_model_steps = max_model_steps
        self.messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    def run(self, question: str) -> AgentRunResult:
        """Run model/tool iterations until a final answer or the step limit."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question cannot be empty.")

        run_id = uuid4().hex
        logger = ExecutionLogger()
        used_tools: list[str] = []
        successful_tools: set[str] = set()
        grounding_rejections = 0
        self.messages.append(HumanMessage(content=normalized_question))
        logger.record("user_question", question=normalized_question)
        logger.record(
            "tool_configuration",
            tool_choice="auto",
            tools=[_tool_schema_summary(tool) for tool in self.tools],
        )

        for model_step in range(1, self.max_model_steps + 1):
            try:
                response = self.model_with_tools.invoke(self.messages)
            except Exception as error:
                final = f"The model request failed: {type(error).__name__}: {error}"
                logger.record(
                    "model_error",
                    status="error",
                    model_step=model_step,
                    error_type=type(error).__name__,
                    message=str(error),
                )
                logger.record("final_response", status="error", response=final)
                return self._finish(
                    run_id, normalized_question, final, "error", used_tools, logger
                )

            if not isinstance(response, AIMessage):
                final = "The model returned an unsupported response type."
                logger.record(
                    "model_error",
                    status="error",
                    model_step=model_step,
                    error_type="UnexpectedResponseType",
                )
                logger.record("final_response", status="error", response=final)
                return self._finish(
                    run_id, normalized_question, final, "error", used_tools, logger
                )

            tool_calls = list(response.tool_calls)
            invalid_calls = list(response.invalid_tool_calls)
            logger.record(
                "model_response",
                model_step=model_step,
                requested_tools=[call.get("name") for call in tool_calls],
                tool_call_ids=[call.get("id") for call in tool_calls],
                invalid_tool_call_count=len(invalid_calls),
                has_final_text=bool(_message_text(response)),
                provider_metadata=_safe_provider_metadata(response),
            )
            self.messages.append(response)

            for invalid_call in invalid_calls:
                logger.record(
                    "invalid_tool_call",
                    status="error",
                    model_step=model_step,
                    name=invalid_call.get("name"),
                    error=invalid_call.get("error"),
                )

            if tool_calls:
                for call in tool_calls:
                    tool_name = str(call.get("name", ""))
                    call_id = str(call.get("id", ""))
                    raw_arguments = call.get("args", {})
                    tool = self.tool_map.get(tool_name)

                    if tool is None:
                        output = {
                            "status": "error",
                            "error_type": "unknown_tool",
                            "message": f"No tool named {tool_name!r} is registered.",
                            "available_tools": sorted(self.tool_map),
                            "warnings": [],
                        }
                        logger.record(
                            "tool_call",
                            status="error",
                            model_step=model_step,
                            tool_name=tool_name,
                            validated_input=raw_arguments,
                        )
                    else:
                        try:
                            schema = tool.get_input_schema()
                            validated_arguments = schema.model_validate(
                                raw_arguments
                            ).model_dump(mode="json", exclude_none=True)
                            logger.record(
                                "tool_call",
                                model_step=model_step,
                                tool_name=tool_name,
                                tool_call_id=call_id,
                                validated_input=validated_arguments,
                            )
                            raw_output = tool.invoke(validated_arguments)
                            output = _normalize_tool_output(raw_output)
                        except ValidationError as error:
                            output = {
                                "status": "error",
                                "error_type": "input_validation_error",
                                "message": str(error),
                                "warnings": [],
                            }
                        except Exception as error:
                            output = {
                                "status": "error",
                                "error_type": type(error).__name__,
                                "message": str(error),
                                "warnings": [],
                            }

                    result_status = (
                        "ok" if output.get("status") == "ok" else "error"
                    )
                    logger.record(
                        "tool_result",
                        status=result_status,
                        model_step=model_step,
                        tool_name=tool_name,
                        tool_call_id=call_id,
                        output=output,
                        warnings=output.get("warnings", []),
                    )
                    used_tools.append(tool_name)
                    if result_status == "ok":
                        successful_tools.add(tool_name)
                    self.messages.append(
                        ToolMessage(
                            content=json.dumps(output, sort_keys=True, allow_nan=False),
                            tool_call_id=call_id,
                            name=tool_name,
                            status=("success" if result_status == "ok" else "error"),
                        )
                    )
                continue

            final_text = _message_text(response)
            if invalid_calls and not final_text:
                self.messages.append(
                    HumanMessage(
                        content=(
                            "Automatic validator feedback: your previous tool call was "
                            "malformed. Reissue it using one registered tool and "
                            "arguments matching its schema, or provide a direct answer "
                            "if no tool is needed."
                        )
                    )
                )
                logger.record(
                    "tool_call_recovery_requested",
                    status="error",
                    model_step=model_step,
                )
                continue
            if not final_text:
                final_text = "The model returned an empty response."
                final_status = "error"
            else:
                final_status = "ok"

            missing_evidence = _missing_grounding_evidence(
                normalized_question,
                final_text,
                successful_tools,
            )
            if final_status == "ok" and missing_evidence:
                grounding_rejections += 1
                logger.record(
                    "grounding_rejection",
                    status="error",
                    model_step=model_step,
                    missing_evidence=missing_evidence,
                    proposed_response=final_text,
                    reason=(
                        "The proposed final response lacked required deterministic "
                        "evidence or required scientific qualifications."
                    ),
                )
                if grounding_rejections >= 3:
                    final = (
                        "The model repeatedly attempted an ungrounded answer. No "
                        "quantitative flood assessment was returned; review the "
                        "execution log and retry with a tool-capable model."
                    )
                    logger.record("final_response", status="error", response=final)
                    return self._finish(
                        run_id,
                        normalized_question,
                        final,
                        "error",
                        used_tools,
                        logger,
                    )
                self.messages.append(
                    HumanMessage(
                        content=(
                            "Automatic validator feedback: the proposed answer was "
                            "rejected because the quantitative result was not grounded "
                            "in successful deterministic tool output. Missing successful "
                            f"tools: {', '.join(missing_evidence)}. "
                            "Reconsider the question and autonomously call whichever "
                            "registered tools are needed. Respect dependency errors "
                            "and retry blocked operations. Do not repeat remembered "
                            "numbers or provide a final answer until the required "
                            "evidence is available."
                        )
                    )
                )
                continue
            logger.record("final_response", status=final_status, response=final_text)
            return self._finish(
                run_id,
                normalized_question,
                final_text,
                final_status,
                used_tools,
                logger,
            )

        final = (
            f"Agent stopped after {self.max_model_steps} model steps without a final "
            "response. Review the execution log for repeated or unresolved tool calls."
        )
        logger.record(
            "step_limit",
            status="error",
            maximum_model_steps=self.max_model_steps,
        )
        logger.record("final_response", status="error", response=final)
        return self._finish(
            run_id, normalized_question, final, "error", used_tools, logger
        )

    def _finish(
        self,
        run_id: str,
        question: str,
        response: str,
        status: str,
        used_tools: list[str],
        logger: ExecutionLogger,
    ) -> AgentRunResult:
        state_snapshot = self.state.snapshot()
        payload = {
            "run_id": run_id,
            "question": question,
            "response": response,
            "status": status,
            "tool_names": used_tools,
            "execution_log": logger.to_list(),
            "state": state_snapshot,
            "authoritative_sources": [
                NHC_REPORT_URL,
                NHC_UPDATE_URL,
                NOAA_RESPONSE_HELP_URL,
                USGS_EVENT_URL,
                USGS_HWM_URL,
            ],
        }
        log_path = self.state.artifact_root / "logs" / f"agent_run_{run_id}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(_redact(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return AgentRunResult(
            run_id=run_id,
            question=question,
            response=response,
            status=status,
            tool_names=tuple(used_tools),
            execution_log=tuple(logger.to_list()),
            state=state_snapshot,
            log_path=str(log_path.resolve()),
        )


def _normalize_tool_output(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"status": "ok", "content": value, "warnings": []}
        if isinstance(parsed, dict):
            return parsed
    return {"status": "ok", "content": str(value), "warnings": []}


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    if isinstance(message.content, list):
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return str(message.content).strip()


def _tool_schema_summary(tool: BaseTool) -> dict[str, Any]:
    """Return the exact registered schema shape without executable implementation."""

    schema = tool.get_input_schema().model_json_schema()
    return {
        "name": tool.name,
        "description": tool.description,
        "required_arguments": schema.get("required", []),
        "argument_names": list(schema.get("properties", {})),
    }


def _safe_provider_metadata(message: AIMessage) -> dict[str, Any]:
    """Capture useful response metadata while excluding content and private reasoning."""

    metadata = message.response_metadata or {}
    safe_keys = (
        "model_name",
        "model",
        "finish_reason",
        "system_fingerprint",
        "service_tier",
    )
    return {
        key: metadata[key]
        for key in safe_keys
        if key in metadata and metadata[key] is not None
    }


def _missing_grounding_evidence(
    question: str,
    _proposed_answer: str,
    successful_tools: set[str],
) -> list[str]:
    """Require deterministic evidence for questions with quantitative intent.

    This is an output acceptance policy, not a tool router: the model still receives
    ``tool_choice='auto'`` and decides which tools to call and in what order.
    """

    question_lower = question.lower()
    requested_analysis = any(
        pattern in question_lower
        for pattern in (
            "how severe",
            "flood severity",
            "severity of",
            "peak level",
            "peak water",
            "maximum water",
            "maximum residual",
            "max residual",
            "observed-minus-predicted",
            "recorded water level",
            "threshold exceed",
            "flood category",
        )
    )
    requested_visualization = any(
        term in question_lower
        for term in ("visualization", "visualisation", "plot", "chart", "graph", "figure")
    )
    requested_spatial = any(
        term in question_lower
        for term in (
            "spatial",
            "high-water mark",
            "high water mark",
            "usgs",
            "across lee county",
            "flood map",
        )
    )
    required: list[str] = []
    if requested_spatial:
        required.append("analyze_spatial_flooding")
    elif requested_analysis:
        required.append("calculate_flood_indicators")
    if requested_visualization:
        required.append(
            "generate_spatial_visualization"
            if requested_spatial
            else "generate_visualization"
        )
    missing_tools = [
        tool_name for tool_name in required if tool_name not in successful_tools
    ]
    return missing_tools


def _redact(value: Any) -> Any:
    """Recursively remove common credential fields before persistence or display."""

    sensitive_fragments = ("api_key", "apikey", "authorization", "password", "secret", "token")
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(fragment in str(key).lower() for fragment in sensitive_fragments)
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        redacted = re.sub(
            r"\bsk-[A-Za-z0-9_-]{8,}\b",
            "[REDACTED_CREDENTIAL]",
            value,
        )
        return re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*",
            "Bearer [REDACTED_CREDENTIAL]",
            redacted,
        )
    return value
