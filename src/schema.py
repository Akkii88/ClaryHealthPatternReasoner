from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json
import re

class User(BaseModel):
    id: str
    name: str
    occupation: Optional[str] = None
    onboarding_notes: Optional[str] = None

class Conversation(BaseModel):
    session_id: str
    user_id: str
    timestamp: datetime
    user_message: str
    clary_questions: Optional[List[str]] = None
    user_followup: Optional[str] = None
    clary_response: Optional[str] = None
    severity: Optional[str] = None
    tags: Optional[List[str]] = None

    # Computed fields for internal use
    combined_text: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Create combined_text from available fields
        text_parts = []
        if self.user_message:
            text_parts.append(f"User: {self.user_message}")
        if self.user_followup:
            text_parts.append(f"User Followup: {self.user_followup}")
        if self.clary_response:
            text_parts.append(f"Clary: {self.clary_response}")
        if self.clary_questions:
            text_parts.extend([f"Clary Question: {q}" for q in self.clary_questions])

        self.combined_text = " | ".join(text_parts) if text_parts else ""

class TimelineEvent(BaseModel):
    user_id: str
    session_id: str
    timestamp: datetime
    week_number: int
    user_message: str
    user_followup: Optional[str] = None
    clary_response: Optional[str] = None
    severity: Optional[str] = None
    tags: List[str] = []
    extracted_text_context: str

class Dataset(BaseModel):
    users: List[User]
    conversations: List[Conversation]
    # hidden_patterns_reference will be removed during loading


class EvidenceTimeline(BaseModel):
    """Timeline evidence for a pattern."""
    session_id: str
    week: Union[str, int]
    timestamp: str
    evidence: str


class RejectedHypothesis(BaseModel):
    """A hypothesis that was considered but rejected."""
    hypothesis: str
    reason_rejected: str


class Pattern(BaseModel):
    """A discovered pattern in user behavior."""
    user_id: str
    pattern_title: str
    sessions_involved: List[str]
    evidence_timeline: List[EvidenceTimeline]
    temporal_reasoning: str
    confidence: str = Field(..., pattern=r'^(low|medium|high|very_high)$')
    confidence_justification: str
    counter_evidence: List[str] = Field(default_factory=list)
    rejected_hypotheses: List[RejectedHypothesis] = Field(default_factory=list)
    reasoning_trace: List[str] = Field(default_factory=list)


class PatternsAnalysis(BaseModel):
    """Complete pattern analysis results."""
    patterns: List[Pattern] = Field(default_factory=list)


def validate_patterns_json(raw_text: str) -> PatternsAnalysis:
    """
    Validate and parse patterns JSON text.

    Args:
        raw_text: Raw JSON string containing pattern analysis results

    Returns:
        PatternsAnalysis object if valid

    Raises:
        ValueError: If JSON is invalid or doesn't match expected structure
    """
    try:
        # Parse JSON
        data = json.loads(raw_text)

        # Validate with Pydantic
        analysis = PatternsAnalysis(**data)

        return analysis

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

    except Exception as e:
        raise ValueError(f"Invalid pattern structure: {str(e)}")


def repair_or_extract_json(raw_text: str) -> PatternsAnalysis:
    """
    Attempt to repair or extract valid JSON from potentially malformed text.

    Args:
        raw_text: Raw text that may contain JSON

    Returns:
        PatternsAnalysis object if extraction succeeds

    Raises:
        ValueError: If no valid JSON can be extracted
    """
    # Clean up common issues
    text = raw_text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r'```\w*\n?', '', text)
    text = text.strip()

    # Try direct parsing first
    try:
        return validate_patterns_json(text)
    except ValueError:
        pass

    # Look for JSON-like structure in the text
    # Find the first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        potential_json = text[start_idx:end_idx + 1]
        try:
            return validate_patterns_json(potential_json)
        except ValueError:
            pass

    # Try to find JSON within ```json``` blocks
    json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return validate_patterns_json(match.strip())
        except ValueError:
            continue

    # If all else fails, try to construct minimal valid structure
    # Look for pattern-like content and create a basic structure
    if 'patterns' in text.lower() or 'user_id' in text:
        try:
            # Create minimal valid structure
            minimal_structure = {
                "patterns": []
            }

            # Try to extract some basic information
            # This is a fallback - in practice, LLM should provide proper JSON

            return PatternsAnalysis(**minimal_structure)

        except Exception:
            pass

    raise ValueError(
        "Could not extract valid pattern analysis JSON from the provided text. "
        "Please ensure the LLM response contains properly formatted JSON with the required structure: "
        '{"patterns": [...]} where each pattern has user_id, pattern_title, etc.'
    )