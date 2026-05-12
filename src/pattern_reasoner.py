from typing import Dict, Any, Optional, List
import json
from .timeline_builder import TimelineEvent
from .llm_client import LLMClient
from .schema import validate_patterns_json, repair_or_extract_json


async def analyze_patterns(timelines: Dict[str, List[TimelineEvent]], selected_user: str, llm_client: LLMClient) -> Optional[Dict[str, Any]]:
    """
    Analyze patterns in user timeline using LLM.

    Args:
        timelines: Dictionary mapping user_id to list of TimelineEvent objects
        selected_user: User ID to analyze
        llm_client: LLM client for analysis

    Returns:
        Dictionary containing pattern analysis results, or None if analysis fails
    """
    # Get user timeline
    user_timeline = timelines.get(selected_user, [])

    if not user_timeline:
        return None

    # Sort timeline by timestamp
    sorted_timeline = sorted(user_timeline, key=lambda x: x.timestamp)

    # Format timeline for LLM
    timeline_text = format_timeline_for_llm(sorted_timeline)

    # Create analysis prompt
    prompt = create_analysis_prompt(timeline_text)

    # Get LLM response
    response_text = await get_llm_response(llm_client, prompt)

    if not response_text:
        return None

    # Parse and validate response
    try:
        analysis = validate_patterns_json(response_text)
        return analysis.dict()
    except Exception:
        # Try to repair/extract JSON
        try:
            repaired_analysis = repair_or_extract_json(response_text)
            return repaired_analysis.dict()
        except Exception:
            return None


def format_timeline_for_llm(timeline_events: list) -> str:
    """
    Format timeline events into readable text for LLM analysis.

    Args:
        timeline_events: List of TimelineEvent objects

    Returns:
        Formatted timeline text
    """
    if not timeline_events:
        return "No timeline events available."

    formatted_events = []

    for event in timeline_events:
        event_text = f"""
Week {event.week_number} - {event.timestamp.strftime('%Y-%m-%d %H:%M')}
Session: {event.session_id}

User Message: {event.user_message}

{f"User Follow-up: {event.user_followup}" if event.user_followup else ""}

{f"Clary Response: {event.clary_response}" if event.clary_response else ""}

{f"Severity: {event.severity}" if event.severity else ""}

{f"Tags: {', '.join(event.tags)}" if event.tags else ""}

Context: {event.extracted_text_context}
---
"""
        formatted_events.append(event_text.strip())

    return "\n".join(formatted_events)


def create_analysis_prompt(timeline_text: str) -> str:
    """
    Create the analysis prompt for the LLM.

    Args:
        timeline_text: Formatted timeline text

    Returns:
        Complete analysis prompt
    """
    prompt = f"""You are a pattern analysis expert. Analyze the user's conversation timeline and identify meaningful patterns in their reported experiences, symptoms, and interactions.

IMPORTANT RULES:
- Do NOT diagnose medical conditions
- Do NOT invent facts or information not present in the timeline
- Use ONLY the provided conversation timeline data
- Prefer temporal reasoning over simple keyword matching
- Focus on patterns of behavior, timing, and relationships between events
- Be conservative - only identify clear, well-supported patterns

TIMELINE DATA:
{timeline_text}

ANALYSIS PROCESS:
1. Read through the entire timeline carefully
2. Extract and categorize elements:
   - Symptoms or issues reported
   - Lifestyle factors mentioned
   - Interventions or actions taken
   - Patterns of worsening or improvement
   - Timing relationships between events

3. Identify repeated patterns across time:
   - Recurring issues or symptoms
   - Cyclical patterns
   - Response patterns to interventions
   - Temporal relationships between triggers and outcomes

4. Compare before/after timing:
   - What precedes certain events
   - What follows interventions
   - Delays between cause and effect
   - Patterns of recurrence

5. Look for specific relationship types:
   - Delay effects (time between trigger and response)
   - Recurrence patterns (issues that return)
   - Intervention response (what works/doesn't work)
   - Dose-response relationships (intensity correlations)
   - Absence patterns (symptoms missing when triggers are absent)

6. Generate candidate hypotheses about patterns you observe

7. Critically evaluate each hypothesis:
   - Reject hypotheses not well supported by timeline evidence
   - Reject hypotheses that require inventing information
   - Only keep hypotheses with strong temporal and evidentiary support

OUTPUT REQUIREMENTS:
Return ONLY valid JSON matching this exact structure:

{{
  "patterns": [
    {{
      "user_id": "string",
      "pattern_title": "string",
      "sessions_involved": ["string"],
      "evidence_timeline": [
        {{
          "session_id": "string",
          "week": "string or number",
          "timestamp": "string",
          "evidence": "string"
        }}
      ],
      "temporal_reasoning": "string",
      "confidence": "low | medium | high | very_high",
      "confidence_justification": "string",
      "counter_evidence": ["string"],
      "rejected_hypotheses": [
        {{
          "hypothesis": "string",
          "reason_rejected": "string"
        }}
      ],
      "reasoning_trace": ["string"]
    }}
  ]
}}

JSON OUTPUT ONLY - no markdown, no explanations, no additional text."""
    return prompt


async def get_llm_response(llm_client: LLMClient, prompt: str) -> str:
    """
    Get response from LLM client.

    Args:
        llm_client: LLM client instance
        prompt: Analysis prompt

    Returns:
        LLM response text
    """
    messages = [
        {
            "role": "system",
            "content": "You are a pattern analysis expert. Always respond with valid JSON only, following the exact schema provided."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response_text = ""
    async for chunk in llm_client.generate_response(messages, temperature=0.3, stream=False):
        response_text += chunk

    return response_text.strip()