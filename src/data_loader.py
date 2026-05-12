import json
import streamlit as st
from typing import Dict, Any, Tuple, List
from datetime import datetime
from .schema import Dataset, User, Conversation

def load_dataset_from_uploaded_file(uploaded_file) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Load and validate dataset from uploaded Streamlit file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (Dataset object, summary dict)

    Raises:
        ValueError: If JSON is invalid or required fields missing
    """
    try:
        # Read file content
        content = uploaded_file.read().decode('utf-8')
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {str(e)}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid file encoding: {str(e)}")

    # Remove hidden_patterns_reference if it exists
    hidden_patterns_removed = False
    if "hidden_patterns_reference" in data:
        del data["hidden_patterns_reference"]
        hidden_patterns_removed = True

    # Validate required fields
    if "users" not in data:
        raise ValueError("Dataset must contain 'users' field")

    # Parse and validate users, and extract conversations
    users = []
    conversations = []

    for user_data in data["users"]:
        # Validate user structure
        if "user_id" not in user_data:
            raise ValueError("Each user must have 'user_id' field")
        if "name" not in user_data:
            raise ValueError("Each user must have 'name' field")
        if "conversations" not in user_data:
            raise ValueError("Each user must have 'conversations' field")

        # Create user object (map user_id to id)
        user_dict = {
            "id": user_data["user_id"],
            "name": user_data["name"],
            "occupation": user_data.get("occupation"),
            "onboarding_notes": user_data.get("onboarding_notes")
        }

        try:
            user = User(**user_dict)
            users.append(user)
        except Exception as e:
            raise ValueError(f"Invalid user data: {str(e)}")

        # Extract conversations for this user
        for conv_data in user_data["conversations"]:
            try:
                # Validate required fields for conversations
                if "session_id" not in conv_data:
                    raise ValueError("Conversation must have 'session_id' field")
                if "user_message" not in conv_data:
                    raise ValueError("Conversation must have 'user_message' field")
                if "timestamp" not in conv_data:
                    raise ValueError("Conversation must have 'timestamp' field")

                # Prepare conversation data with user_id
                conversation_data = {
                    "session_id": conv_data["session_id"],
                    "user_id": user_data["user_id"],
                    "user_message": conv_data["user_message"],
                    "clary_questions": conv_data.get("clary_questions", []),
                    "user_followup": conv_data.get("user_followup"),
                    "clary_response": conv_data.get("clary_response"),
                    "severity": conv_data.get("severity"),
                    "tags": conv_data.get("tags", [])
                }

                # Convert timestamp string to datetime if needed
                if isinstance(conv_data.get("timestamp"), str):
                    conversation_data["timestamp"] = datetime.fromisoformat(conv_data["timestamp"].replace('Z', '+00:00'))
                else:
                    conversation_data["timestamp"] = conv_data["timestamp"]

                conversation = Conversation(**conversation_data)
                conversations.append(conversation)
            except Exception as e:
                raise ValueError(f"Invalid conversation data for user {user_data['user_id']}: {str(e)}")

    # Create dataset
    dataset = Dataset(users=users, conversations=conversations)

    # Generate summary
    summary = generate_dataset_summary(dataset)

    # Add success messages to summary
    summary["load_success_message"] = "Dataset loaded successfully"
    if hidden_patterns_removed:
        summary["hidden_patterns_message"] = "hidden_patterns_reference removed successfully"

    return dataset, summary

def generate_dataset_summary(dataset: Dataset) -> Dict[str, Any]:
    """
    Generate summary statistics for the dataset.

    Args:
        dataset: Dataset object

    Returns:
        Dictionary with summary statistics
    """
    # Basic counts
    total_users = len(dataset.users)
    total_conversations = len(dataset.conversations)

    # Date range
    if dataset.conversations:
        timestamps = [conv.timestamp for conv in dataset.conversations if conv.timestamp]
        if timestamps:
            date_range = {
                "start": min(timestamps),
                "end": max(timestamps)
            }
        else:
            date_range = None
    else:
        date_range = None

    # User IDs
    user_ids = [user.id for user in dataset.users]

    # Conversations per user
    conversations_per_user = {}
    for conv in dataset.conversations:
        conversations_per_user[conv.user_id] = conversations_per_user.get(conv.user_id, 0) + 1

    # User preview info
    user_preview = []
    for user in dataset.users:
        user_preview.append({
            "id": user.id,
            "name": user.name,
            "occupation": user.occupation,
            "conversations_count": conversations_per_user.get(user.id, 0),
            "onboarding_notes": user.onboarding_notes
        })

    return {
        "total_users": total_users,
        "total_conversations": total_conversations,
        "date_range": date_range,
        "user_ids": user_ids,
        "user_preview": user_preview
    }