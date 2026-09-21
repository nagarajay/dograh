"""The persisted per-turn latency breakdown event."""

from pipecat.observers.user_bot_latency_observer import (
    FunctionCallMetrics,
    LatencyBreakdown,
    TextAggregationBreakdownMetrics,
    TTFBBreakdownMetrics,
)

from api.services.pipecat.realtime_feedback_events import (
    LATENCY_BREAKDOWN_EVENT_TYPE,
    build_latency_breakdown_event,
)


def test_event_lists_every_stage_of_the_turn():
    breakdown = LatencyBreakdown(
        ttfb=[
            TTFBBreakdownMetrics(
                processor="OpenAILLMService#1",
                model="gpt-5.6-luna",
                start_time=1.0,
                duration_secs=0.81549,
            ),
            TTFBBreakdownMetrics(
                processor="RecordingRouterProcessor#0",
                start_time=1.9,
                duration_secs=0.14321,
            ),
            TTFBBreakdownMetrics(
                processor="DeepgramTTSService#1",
                start_time=2.1,
                duration_secs=0.30456,
            ),
        ],
        text_aggregation=TextAggregationBreakdownMetrics(
            processor="DeepgramTTSService#1", start_time=2.0, duration_secs=0.0612
        ),
        user_turn_start_time=0.0,
        user_turn_secs=0.80012,
        function_calls=[
            FunctionCallMetrics(
                function_name="caller_wants_to_end",
                start_time=3.0,
                duration_secs=0.0312,
            )
        ],
    )

    event = build_latency_breakdown_event(breakdown)

    assert event["type"] == LATENCY_BREAKDOWN_EVENT_TYPE == "rtf-latency-breakdown"
    assert event["payload"] == {
        "user_turn_secs": 0.8001,
        "ttfb": [
            {
                "processor": "OpenAILLMService#1",
                "model": "gpt-5.6-luna",
                "secs": 0.8155,
            },
            {"processor": "RecordingRouterProcessor#0", "model": None, "secs": 0.1432},
            {"processor": "DeepgramTTSService#1", "model": None, "secs": 0.3046},
        ],
        "text_aggregation_secs": 0.0612,
        "function_calls": [{"function_name": "caller_wants_to_end", "secs": 0.0312}],
    }


def test_event_tolerates_a_turn_with_nothing_measured():
    event = build_latency_breakdown_event(LatencyBreakdown())

    assert event["payload"] == {
        "user_turn_secs": None,
        "ttfb": [],
        "text_aggregation_secs": None,
        "function_calls": [],
    }
