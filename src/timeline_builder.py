from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .schema import User, Conversation, TimelineEvent, Dataset


def parse_conversation_content(content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse conversation content and metadata to extract structured fields.

    Args:
        content: Raw conversation content string
        metadata: Optional metadata dictionary

    Returns:
        Dictionary with parsed fields: user_message, user_followup, clary_response,
        severity, tags, session_id
    """
    parsed = {
        "user_message": "",
        "user_followup": None,
        "clary_response": None,
        "severity": None,
        "tags": [],
        "session_id": ""
    }

    # If metadata contains structured data, use it
    if metadata:
        parsed.update({
            "user_message": metadata.get("user_message", content),
            "user_followup": metadata.get("user_followup"),
            "clary_response": metadata.get("clary_response"),
            "severity": metadata.get("severity"),
            "tags": metadata.get("tags", []),
            "session_id": metadata.get("session_id", "")
        })
    else:
        # Fallback: treat entire content as user_message
        parsed["user_message"] = content
        parsed["session_id"] = f"session_{content[:10].replace(' ', '_')}"

    return parsed


def calculate_week_number(timestamp: datetime, first_conversation_date: datetime) -> int:
    """
    Calculate the week number from the first conversation date.

    Args:
        timestamp: Current conversation timestamp
        first_conversation_date: First conversation timestamp for this user

    Returns:
        Week number (1-based)
    """
    if first_conversation_date is None:
        return 1

    # Calculate days difference and convert to weeks
    days_diff = (timestamp.date() - first_conversation_date.date()).days
    week_number = (days_diff // 7) + 1

    return max(1, week_number)  # Ensure minimum week 1


def build_user_timeline(user: User, conversations: List[Conversation]) -> List[TimelineEvent]:
    """
    Build timeline events for a specific user.

    Args:
        user: User object
        conversations: List of all conversations (will be filtered by user_id)

    Returns:
        List of TimelineEvent objects for this user
    """
    # Filter conversations for this user
    user_conversations = [conv for conv in conversations if conv.user_id == user.id]

    if not user_conversations:
        return []

    # Sort conversations by timestamp
    user_conversations.sort(key=lambda x: x.timestamp)

    # Find first conversation date for this user
    first_conversation_date = user_conversations[0].timestamp

    timeline_events = []

    for conv in user_conversations:
        # Use conversation data directly (now available in the schema)
        # Create extracted text context from combined_text or build it
        extracted_text_context = conv.combined_text or conv.user_message

        # Create timeline event
        event = TimelineEvent(
            user_id=user.id,
            session_id=conv.session_id,
            timestamp=conv.timestamp,
            week_number=calculate_week_number(conv.timestamp, first_conversation_date),
            user_message=conv.user_message,
            user_followup=conv.user_followup,
            clary_response=conv.clary_response,
            severity=conv.severity,
            tags=conv.tags or [],
            extracted_text_context=extracted_text_context
        )

        timeline_events.append(event)

    return timeline_events


def build_all_timelines(dataset: Dataset) -> Dict[str, List[TimelineEvent]]:
    """
    Build timeline events for all users in the dataset.

    Args:
        dataset: Dataset object containing users and conversations

    Returns:
        Dictionary mapping user_id to list of TimelineEvent objects
    """
    timelines = {}

    for user in dataset.users:
        user_timeline = build_user_timeline(user, dataset.conversations)
        timelines[user.id] = user_timeline

    return timelines


def get_timeline_summary(timeline_events: List[TimelineEvent]) -> Dict[str, Any]:
    """
    Generate summary statistics for timeline events.

    Args:
        timeline_events: List of TimelineEvent objects

    Returns:
        Dictionary with timeline summary statistics
    """
    if not timeline_events:
        return {
            "total_events": 0,
            "week_range": None,
            "severity_distribution": {},
            "tags_distribution": {},
            "avg_events_per_week": 0
        }

    # Basic counts
    total_events = len(timeline_events)

    # Week range
    weeks = [event.week_number for event in timeline_events]
    week_range = {
        "min": min(weeks),
        "max": max(weeks)
    }

    # Severity distribution
    severity_dist = {}
    for event in timeline_events:
        severity = event.severity or "unknown"
        severity_dist[severity] = severity_dist.get(severity, 0) + 1

    # Tags distribution
    tags_dist = {}
    for event in timeline_events:
        for tag in event.tags:
            tags_dist[tag] = tags_dist.get(tag, 0) + 1

    # Average events per week
    total_weeks = week_range["max"] - week_range["min"] + 1
    avg_events_per_week = total_events / total_weeks if total_weeks > 0 else 0

    return {
        "total_events": total_events,
        "week_range": week_range,
        "severity_distribution": severity_dist,
        "tags_distribution": tags_dist,
        "avg_events_per_week": round(avg_events_per_week, 2)
    }