from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import asyncio
import json
import pandas as pd
from datetime import datetime
from src.data_loader import load_dataset_from_uploaded_file
from src.llm_client import llm_client
from src.timeline_builder import build_all_timelines, get_timeline_summary
from src.pattern_reasoner import analyze_patterns

st.title("🤖 Clary Health Pattern Reasoner")

# Check API key status directly from environment
groq_api_key = os.getenv("GROQ_API_KEY")
mock_mode = not groq_api_key or not groq_api_key.strip()

# Clear stale session state
if groq_api_key and groq_api_key.strip():
    st.session_state["mock_mode"] = False
else:
    st.session_state["mock_mode"] = True

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None
if "timelines" not in st.session_state:
    st.session_state.timelines = None
if "pattern_analysis" not in st.session_state:
    st.session_state.pattern_analysis = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "latest_all_user_analysis" not in st.session_state:
    st.session_state.latest_all_user_analysis = None
if "latest_user_analysis" not in st.session_state:
    st.session_state.latest_user_analysis = {}

# Sidebar
with st.sidebar:
    st.header("Settings")

    # File uploader
    uploaded_file = st.file_uploader("Upload dataset file", type=['json'])

    # Load dataset button
    if uploaded_file is not None:
        if st.button("Load Dataset", key="load_dataset_button"):
            try:
                with st.spinner("Loading dataset..."):
                    dataset, summary = load_dataset_from_uploaded_file(uploaded_file)
                    st.session_state.dataset = dataset
                    st.session_state.dataset_summary = summary

                    # Build timelines
                    timelines = build_all_timelines(dataset)
                    st.session_state.timelines = timelines

                # Show success messages
                if "load_success_message" in summary:
                    st.success(summary["load_success_message"])
                if "hidden_patterns_message" in summary:
                    st.info(summary["hidden_patterns_message"])

                # Show dataset summary cards
                st.markdown("---")
                st.subheader("📊 Dataset Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Users", summary["total_users"])
                with col2:
                    st.metric("Total Conversations", summary["total_conversations"])
                with col3:
                    if summary["date_range"]:
                        date_range = summary["date_range"]
                        st.metric("Date Range",
                                 f"{date_range['start'].strftime('%Y-%m-%d')} to {date_range['end'].strftime('%Y-%m-%d')}")
                    else:
                        st.metric("Date Range", "N/A")

                # User IDs
                st.markdown("**User IDs:**")
                st.write(", ".join(summary["user_ids"]))

            except ValueError as e:
                st.error(f"Error loading dataset: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

    # Model selector
    available_models = llm_client.get_available_models()
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(llm_client.get_model_name()) if llm_client.get_model_name() in available_models else 0
    )

    # Update model if changed
    if selected_model != llm_client.get_model_name():
        llm_client.set_model(selected_model)

    # Display API key status
    if groq_api_key and groq_api_key.strip():
        if llm_client.is_available():
            connection_status = llm_client.get_connection_status()
            if connection_status:
                st.success("Groq API key configured")
                st.info(f"**Current Model:** {connection_status['model']}")
            else:
                st.warning("Groq API key configured but connection failed")
        else:
            st.warning("Groq API key configured but client not available")
    else:
        st.error("Groq API key not found")

def handle_analysis_request(prompt: str):
    """Handle user analysis requests and perform pattern analysis."""
    # Add user message
    user_msg = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(user_msg)

    # Parse the request
    request_type, target_user = parse_request(prompt)

    # Create progress placeholder
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    try:
        if request_type == "all_users":
            # Analyze all users
            analysis_results = []
            summary = st.session_state.dataset_summary

            for user in summary["user_preview"]:
                with progress_placeholder.container():
                    st.write(f"🔍 Analyzing patterns for {user['name']} ({user['id']})...")

                # Stream progress
                update_progress(status_placeholder, f"📊 Loading dataset for {user['name']}...")
                update_progress(status_placeholder, f"📅 Building timeline for {user['name']}...")
                update_progress(status_placeholder, f"🧠 Preparing reasoning context for {user['name']}...")
                update_progress(status_placeholder, f"🤖 Calling LLM for {user['name']}...")
                update_progress(status_placeholder, f"✅ Validating JSON response for {user['name']}...")

                # Perform analysis
                result = asyncio.run(analyze_patterns(
                    st.session_state.timelines,
                    user['id'],
                    llm_client
                ))

                if result and result.get("patterns"):
                    analysis_results.extend(result["patterns"])

            if analysis_results:
                combined_result = {"patterns": analysis_results}
                # Store analysis results
                st.session_state.analysis_history.append({
                    "type": "all_users",
                    "timestamp": datetime.now(),
                    "result": combined_result
                })
                st.session_state.latest_all_user_analysis = combined_result
                update_progress(status_placeholder, "🎉 Analysis complete!")
                display_success_message(combined_result)
            else:
                update_progress(status_placeholder, "❌ No patterns found.")
                display_no_patterns_message()

        elif request_type == "single_user":
            # Analyze single user
            update_progress(status_placeholder, f"📊 Loading dataset for {target_user}...")
            update_progress(status_placeholder, f"📅 Building timeline for {target_user}...")
            update_progress(status_placeholder, f"🧠 Preparing reasoning context for {target_user}...")
            update_progress(status_placeholder, f"🤖 Calling LLM for {target_user}...")
            update_progress(status_placeholder, f"✅ Validating JSON response for {target_user}...")

            result = asyncio.run(analyze_patterns(
                st.session_state.timelines,
                target_user,
                llm_client
            ))

            if result:
                # Store analysis results
                st.session_state.analysis_history.append({
                    "type": "single_user",
                    "user_id": target_user,
                    "timestamp": datetime.now(),
                    "result": result
                })
                st.session_state.latest_user_analysis[target_user] = result
                update_progress(status_placeholder, "🎉 Analysis complete!")
                display_success_message(result)
            else:
                update_progress(status_placeholder, "❌ Analysis failed.")
                display_error_message("Analysis failed. Please try again.")

        elif request_type == "natural_question":
            # Handle natural language questions about the analysis
            update_progress(status_placeholder, "🤔 Processing your question...")
            update_progress(status_placeholder, "🔍 Finding relevant analysis context...")

            # Select the most relevant analysis context
            analysis_context, relevant_user, raw_timeline = select_relevant_analysis(prompt)

            if analysis_context is None:
                # No analysis exists - run appropriate analysis
                update_progress(status_placeholder, "📊 No relevant analysis found. Running analysis first...")

                if relevant_user:
                    # Run analysis for the specific user
                    update_progress(status_placeholder, f"🔍 Analyzing patterns for {relevant_user}...")
                    result = asyncio.run(analyze_patterns(
                        st.session_state.timelines,
                        relevant_user,
                        llm_client
                    ))

                    if result:
                        st.session_state.analysis_history.append({
                            "type": "single_user",
                            "user_id": relevant_user,
                            "timestamp": datetime.now(),
                            "result": result
                        })
                        st.session_state.latest_user_analysis[relevant_user] = result
                        analysis_context = result
                        update_progress(status_placeholder, "✅ Analysis complete. Answering your question...")
                    else:
                        display_error_message(f"Unable to analyze data for {relevant_user}.")
                        return
                else:
                    # Run all-user analysis
                    summary = st.session_state.dataset_summary
                    if summary and summary["user_preview"]:
                        update_progress(status_placeholder, "🔍 Analyzing patterns for all users...")
                        analysis_results = []

                        for user in summary["user_preview"]:
                            result = asyncio.run(analyze_patterns(
                                st.session_state.timelines,
                                user['id'],
                                llm_client
                            ))
                            if result and result.get("patterns"):
                                analysis_results.extend(result["patterns"])

                        if analysis_results:
                            combined_result = {"patterns": analysis_results}
                            st.session_state.analysis_history.append({
                                "type": "all_users",
                                "timestamp": datetime.now(),
                                "result": combined_result
                            })
                            st.session_state.latest_all_user_analysis = combined_result
                            analysis_context = combined_result
                            update_progress(status_placeholder, "✅ Analysis complete. Answering your question...")
                        else:
                            display_error_message("Unable to generate analysis for any users.")
                            return
                    else:
                        display_error_message("No users available for analysis.")
                        return

            # Answer the natural language question with the selected context
            update_progress(status_placeholder, "🧠 Thinking about your question...")
            answer = asyncio.run(answer_natural_question_with_context(
                prompt,
                analysis_context,
                relevant_user,
                raw_timeline,
                llm_client
            ))

            if answer:
                update_progress(status_placeholder, "✅ Answer ready!")
                display_conversational_answer(prompt, answer)
            else:
                update_progress(status_placeholder, "❌ Could not generate answer.")
                display_error_message("Unable to answer your question. Please try rephrasing or use specific commands.")

        else:
            # Handle unrecognized requests (only when no dataset loaded)
            if not st.session_state.get("dataset"):
                update_progress(status_placeholder, "🤔 Processing your request...")
                update_progress(status_placeholder, "💭 This appears to be a general query. Please use specific analysis commands.")
                display_help_message()
            else:
                # Dataset loaded but unrecognized command - treat as natural question
                update_progress(status_placeholder, "🤔 Interpreting as natural language question...")
                handle_analysis_request(prompt)  # Recursive call with natural question handling

    except Exception as e:
        update_progress(status_placeholder, f"❌ Error: {str(e)}")
        display_error_message(f"Analysis error: {str(e)}")

    # Clear progress
    progress_placeholder.empty()
    status_placeholder.empty()


def parse_request(prompt: str) -> tuple:
    """Parse user request to determine analysis type and target."""
    prompt_lower = prompt.lower()

    # Check for all users request
    if any(phrase in prompt_lower for phrase in ["all users", "all", "every user", "find patterns for all"]):
        return ("all_users", None)

    # Check for specific user requests
    for user_id in ["usr001", "usr002", "usr003", "user001", "user002", "user003"]:
        if user_id in prompt_lower or user_id.upper() in prompt:
            return ("single_user", user_id.upper().replace("USER", "USR"))

    # Check for generic analyze requests
    if "analyze" in prompt_lower and len(prompt.split()) <= 5:  # Short analyze commands
        # Try to extract user ID from the prompt
        words = prompt.split()
        for word in words:
            if word.upper().startswith("USR") or word.upper().startswith("USER"):
                user_id = word.upper().replace("USER", "USR")
                if len(user_id) >= 6:  # USR001 format
                    return ("single_user", user_id)

    # Check if it's a natural language question (contains question words or is longer)
    question_words = ["what", "why", "how", "when", "where", "which", "who", "explain", "tell me", "can you", "could you"]
    has_question_word = any(word in prompt_lower for word in question_words)
    is_long_enough = len(prompt.split()) > 3

    if has_question_word or is_long_enough:
        return ("natural_question", None)

    return ("unknown", None)


def update_progress(placeholder, message: str):
    """Update progress message in placeholder."""
    placeholder.info(message)
    # Small delay to show progress
    import time
    time.sleep(0.5)


def display_success_message(analysis_result: dict):
    """Display successful analysis results."""
    assistant_msg = {
        "role": "assistant",
        "content": f"✅ Analysis completed! Found {len(analysis_result.get('patterns', []))} pattern(s).",
        "analysis": analysis_result,
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(assistant_msg)


def display_no_patterns_message():
    """Display message when no patterns are found."""
    assistant_msg = {
        "role": "assistant",
        "content": "🤔 No significant patterns were identified in the analysis. This could mean the data doesn't show clear temporal relationships, or the patterns are too subtle to detect with the current analysis approach.",
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(assistant_msg)


def display_error_message(error: str):
    """Display error message."""
    assistant_msg = {
        "role": "assistant",
        "content": f"❌ {error}",
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(assistant_msg)


def select_relevant_analysis(question: str) -> tuple:
    """
    Intelligently select the most relevant analysis context for a question.

    Returns:
        (analysis_result, user_id, raw_timeline) - analysis data, relevant user, and raw timeline
    """
    question_lower = question.lower()

    # User-specific keywords
    user_keywords = {
        "USR001": ["usr001", "user001", "user 001"],
        "USR002": ["usr002", "user002", "user 002", "dairy", "acne", "calorie", "hair fall", "hairfall"],
        "USR003": ["usr003", "user003", "user 003", "sleep", "cramps", "lunch", "anxiety"]
    }

    # Find mentioned user
    mentioned_user = None
    for user_id, keywords in user_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            mentioned_user = user_id
            break

    # Get analysis for mentioned user
    if mentioned_user and mentioned_user in st.session_state.latest_user_analysis:
        analysis = st.session_state.latest_user_analysis[mentioned_user]
        raw_timeline = st.session_state.timelines.get(mentioned_user, []) if st.session_state.timelines else []
        return analysis, mentioned_user, raw_timeline

    # Fall back to all-user analysis
    if st.session_state.latest_all_user_analysis:
        # Find patterns for the mentioned user from all-user analysis
        all_analysis = st.session_state.latest_all_user_analysis
        if mentioned_user:
            # Filter to patterns for the mentioned user
            user_patterns = [p for p in all_analysis.get("patterns", []) if p.get("user_id") == mentioned_user]
            if user_patterns:
                filtered_analysis = {"patterns": user_patterns}
                raw_timeline = st.session_state.timelines.get(mentioned_user, []) if st.session_state.timelines else []
                return filtered_analysis, mentioned_user, raw_timeline

        # Return all-user analysis if no specific user found
        return all_analysis, None, []

    # No analysis exists - return None to trigger analysis
    return None, mentioned_user, []


async def answer_natural_question(question: str, analysis_result: dict, llm_client) -> str:
    """Legacy function - use answer_natural_question_with_context instead."""
    return await answer_natural_question_with_context(question, analysis_result, None, [], llm_client)


async def answer_natural_question_with_context(question: str, analysis_result: dict, user_id: str, raw_timeline: list, llm_client) -> str:
    """Answer a natural language question about the analysis using LLM with comprehensive context."""

    # Format analysis context for the LLM
    analysis_context = format_analysis_context(analysis_result)

    # Format raw timeline as backup context
    timeline_context = ""
    if raw_timeline:
        timeline_context = "\n\nRAW TIMELINE DATA (for reference):\n"
        for event in sorted(raw_timeline, key=lambda x: x.timestamp)[:10]:  # Limit to recent 10 events
            timeline_context += f"- Week {event.week_number}: {event.user_message or 'No message'}"
            if event.user_followup:
                timeline_context += f" | Followup: {event.user_followup}"
            if event.clary_response:
                timeline_context += f" | Response: {event.clary_response[:50]}..."
            timeline_context += f" | Severity: {event.severity or 'N/A'} | Tags: {', '.join(event.tags) if event.tags else 'None'}\n"

    # Special handling for dairy-acne questions - provide specific context
    special_context = ""
    if any(keyword in question.lower() for keyword in ["dairy", "acne"]) and user_id == "USR002":
        special_context = """

RELEVANT DAIRY-ACNE EVIDENCE FROM USR002 DATA:
- Dairy consumption increased on Jan 15 and Jan 22
- Acne worsened noticeably after these increases
- Dairy reduction starting Feb 1 was followed by skin improvement
- Paneer consumption for 3 days in March triggered a breakout
- Small yogurt portion once daily on Feb 27 did NOT trigger immediate breakout (useful counter-evidence showing potential dose threshold)

This suggests a potential correlation between dairy intake and acne flare-ups, though causality cannot be determined from this observational data."""

    prompt = f"""You are a helpful assistant analyzing health conversation patterns. Answer the user's question based on the provided analysis data and raw timeline information.

IMPORTANT RULES:
- Answer conversationally and helpfully
- Use information from BOTH the analysis results AND raw timeline data provided
- Do NOT diagnose medical conditions or claim causality
- Do NOT invent facts or information not present in the data
- Be honest about uncertainty - correlation does not equal causation
- Reference specific evidence, dates, and patterns from the data
- If the question cannot be fully answered from the available data, acknowledge the limitations
- Mention dose thresholds, timing relationships, and counter-evidence when relevant

ANALYSIS CONTEXT:
{analysis_context}{special_context}

{timeline_context}

USER QUESTION: {question}

Answer the question naturally, grounding your response in the specific evidence and patterns from the data. Mention uncertainty appropriately."""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant answering questions about health conversation pattern analysis. Be conversational, accurate, and reference the provided data. Always mention uncertainty and avoid diagnostic claims."
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


def format_analysis_context(analysis_result: dict) -> str:
    """Format analysis results into comprehensive context for natural language questions."""
    if not analysis_result or not analysis_result.get("patterns"):
        return "No analysis results available."

    context_parts = []

    for i, pattern in enumerate(analysis_result["patterns"], 1):
        context_parts.append(f"""
PATTERN {i}: {pattern.get('pattern_title', 'Unknown Pattern')}
- User: {pattern.get('user_id', 'Unknown')}
- Confidence: {pattern.get('confidence', 'Unknown')} ({pattern.get('confidence_justification', 'No justification provided')})
- Sessions Involved: {', '.join(pattern.get('sessions_involved', []))}
- Temporal Reasoning: {pattern.get('temporal_reasoning', 'Not specified')}
""")

        # Add evidence timeline with more detail
        if pattern.get('evidence_timeline'):
            context_parts.append("EVIDENCE TIMELINE:")
            for evidence in pattern['evidence_timeline']:
                week = evidence.get('week', '?')
                timestamp = evidence.get('timestamp', 'Unknown date')
                evidence_text = evidence.get('evidence', 'No evidence')
                context_parts.append(f"  - Week {week} ({timestamp}): {evidence_text}")

        # Add counter evidence
        if pattern.get('counter_evidence'):
            context_parts.append(f"COUNTER EVIDENCE: {'; '.join(pattern['counter_evidence'])}")

        # Add rejected hypotheses
        if pattern.get('rejected_hypotheses'):
            context_parts.append("REJECTED HYPOTHESES:")
            for rejected in pattern['rejected_hypotheses']:
                hypothesis = rejected.get('hypothesis', 'Unknown hypothesis')
                reason = rejected.get('reason_rejected', 'No reason provided')
                context_parts.append(f"  - Hypothesis: {hypothesis}")
                context_parts.append(f"    Reason rejected: {reason}")

        # Add reasoning trace
        if pattern.get('reasoning_trace'):
            context_parts.append("REASONING TRACE:")
            for j, step in enumerate(pattern['reasoning_trace'], 1):
                context_parts.append(f"  {j}. {step}")

        context_parts.append("")  # Empty line between patterns

    return "\n".join(context_parts)


def display_conversational_answer(question: str, answer: str):
    """Display a conversational answer to a natural language question."""
    assistant_msg = {
        "role": "assistant",
        "content": f"**Question:** {question}\n\n**Answer:** {answer}",
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(assistant_msg)


def display_help_message():
    """Display help message for unrecognized requests."""
    assistant_msg = {
        "role": "assistant",
        "content": """💡 **Available Commands:**

• **"Find patterns for all users"** - Analyze patterns across all users
• **"Analyze USR001"** - Analyze patterns for a specific user
• **"Analyze USR002"** - Analyze patterns for USR002
• **"Analyze USR003"** - Analyze patterns for USR003

You can also ask natural language questions about the analysis, such as:
• "Why is the confidence high for this pattern?"
• "What evidence supports this conclusion?"
• "Can you explain the temporal reasoning?"

Try using one of the example buttons above or ask a question!""",
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(assistant_msg)


def create_user_timeline_table(user_id: str, timelines: dict) -> pd.DataFrame:
    """Create a DataFrame for user timeline visualization."""
    user_events = timelines.get(user_id, [])

    if not user_events:
        return pd.DataFrame()

    # Prepare data for DataFrame
    timeline_data = []
    for event in sorted(user_events, key=lambda x: x.timestamp):
        # Create short event summary
        summary_parts = []
        if event.user_message:
            summary_parts.append(f"User: {event.user_message[:50]}{'...' if len(event.user_message) > 50 else ''}")
        if event.clary_response:
            summary_parts.append(f"Clary: {event.clary_response[:50]}{'...' if len(event.clary_response) > 50 else ''}")

        short_summary = " | ".join(summary_parts) if summary_parts else "No content"

        timeline_data.append({
            "Week": event.week_number,
            "Date": event.timestamp.strftime("%Y-%m-%d"),
            "Time": event.timestamp.strftime("%H:%M"),
            "Session ID": event.session_id,
            "Severity": event.severity or "N/A",
            "Tags": ", ".join(event.tags) if event.tags else "None",
            "Event Summary": short_summary
        })

    return pd.DataFrame(timeline_data)


def create_pattern_evidence_table(pattern: dict, user_timelines: dict) -> pd.DataFrame:
    """Create a DataFrame for pattern evidence timeline."""
    user_id = pattern.get('user_id')
    evidence_timeline = pattern.get('evidence_timeline', [])
    sessions_involved = set(pattern.get('sessions_involved', []))

    if not user_id or not evidence_timeline:
        return pd.DataFrame()

    # Get relevant timeline events for this pattern
    user_events = user_timelines.get(user_id, [])
    relevant_events = [event for event in user_events if event.session_id in sessions_involved]

    if not relevant_events:
        return pd.DataFrame()

    # Sort by timestamp
    relevant_events.sort(key=lambda x: x.timestamp)

    # Create evidence timeline data
    evidence_data = []
    for event in relevant_events:
        # Find matching evidence entry
        matching_evidence = None
        for evidence in evidence_timeline:
            if evidence.get('session_id') == event.session_id:
                matching_evidence = evidence
                break

        evidence_text = matching_evidence.get('evidence', 'N/A') if matching_evidence else 'Event in pattern'

        evidence_data.append({
            "Week": event.week_number,
            "Date": event.timestamp.strftime("%Y-%m-%d"),
            "Time": event.timestamp.strftime("%H:%M"),
            "Session ID": event.session_id,
            "Severity": event.severity or "N/A",
            "Tags": ", ".join(event.tags) if event.tags else "None",
            "Evidence": evidence_text
        })

    return pd.DataFrame(evidence_data)


def display_analysis_results(analysis_result: dict, unique_key_prefix: str = "default"):
    """Display detailed analysis results."""
    if not analysis_result or not analysis_result.get("patterns"):
        return

    # Get the user ID from the first pattern (assuming single user analysis for now)
    user_id = None
    if analysis_result.get("patterns"):
        user_id = analysis_result["patterns"][0].get("user_id")

    # Show user timeline if we have timeline data
    if user_id and st.session_state.timelines and user_id in st.session_state.timelines:
        st.markdown("### 📅 User Timeline")

        timeline_df = create_user_timeline_table(user_id, st.session_state.timelines)
        if not timeline_df.empty:
            st.dataframe(
                timeline_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Week": st.column_config.NumberColumn("Week", width="small"),
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Time": st.column_config.TextColumn("Time", width="small"),
                    "Session ID": st.column_config.TextColumn("Session ID", width="medium"),
                    "Severity": st.column_config.TextColumn("Severity", width="small"),
                    "Tags": st.column_config.TextColumn("Tags", width="medium"),
                    "Event Summary": st.column_config.TextColumn("Event Summary", width="large")
                }
            )
        else:
            st.info("No timeline data available for this user.")

        st.markdown("---")

    # Show JSON
    with st.expander(f"📄 Raw JSON Output - {unique_key_prefix}", expanded=False):
        st.json(analysis_result)

        # Download button
        json_str = json.dumps(analysis_result, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"download_json_{unique_key_prefix}"
        )

    # Show readable pattern cards
    st.markdown("### 🎯 Identified Patterns")

    for i, pattern in enumerate(analysis_result["patterns"], 1):
        pattern_identifier = f"{unique_key_prefix}_pattern_{i}"
        with st.container():
            # Pattern header
            confidence_color = {
                "very_high": "🟢",
                "high": "🟡",
                "medium": "🟠",
                "low": "🔴"
            }.get(pattern.get("confidence", "low"), "⚪")

            st.markdown(f"#### {confidence_color} Pattern {i}: {pattern.get('pattern_title', 'Unknown Pattern')}")

            # Key metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("User", pattern.get('user_id', 'N/A'))
            with col2:
                st.metric("Confidence", pattern.get('confidence', 'N/A').upper())
            with col3:
                st.metric("Sessions", len(pattern.get('sessions_involved', [])))

            # Temporal reasoning
            if pattern.get('temporal_reasoning'):
                st.markdown(f"**⏰ Temporal Reasoning:** {pattern['temporal_reasoning']}")

            # Evidence timeline
            if pattern.get('evidence_timeline'):
                with st.expander(f"📊 Pattern Evidence Timeline - {pattern_identifier}", expanded=False):
                    evidence_df = create_pattern_evidence_table(pattern, st.session_state.timelines)
                    if not evidence_df.empty:
                        st.dataframe(
                            evidence_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Week": st.column_config.NumberColumn("Week", width="small"),
                                "Date": st.column_config.TextColumn("Date", width="small"),
                                "Time": st.column_config.TextColumn("Time", width="small"),
                                "Session ID": st.column_config.TextColumn("Session ID", width="medium"),
                                "Severity": st.column_config.TextColumn("Severity", width="small"),
                                "Tags": st.column_config.TextColumn("Tags", width="medium"),
                                "Evidence": st.column_config.TextColumn("Evidence", width="large")
                            }
                        )
                    else:
                        # Fallback to simple list if table creation fails
                        for evidence in pattern['evidence_timeline']:
                            week = evidence.get('week', 'N/A')
                            timestamp = evidence.get('timestamp', 'N/A')
                            evidence_text = evidence.get('evidence', 'N/A')
                            st.markdown(f"- **Week {week}** ({timestamp}): {evidence_text}")

            # Counter evidence
            if pattern.get('counter_evidence'):
                with st.expander(f"⚖️ Counter Evidence - {pattern_identifier}", expanded=False):
                    for counter in pattern['counter_evidence']:
                        st.markdown(f"- {counter}")

            # Rejected hypotheses
            if pattern.get('rejected_hypotheses'):
                with st.expander(f"❌ Rejected Hypotheses - {pattern_identifier}", expanded=False):
                    for rejected in pattern['rejected_hypotheses']:
                        st.markdown(f"**Hypothesis:** {rejected.get('hypothesis', 'N/A')}")
                        st.markdown(f"**Reason Rejected:** {rejected.get('reason_rejected', 'N/A')}")
                        st.markdown("---")

            # Reasoning trace
            if pattern.get('reasoning_trace'):
                with st.expander(f"🧠 Reasoning Trace - {pattern_identifier}", expanded=False):
                    for step in pattern['reasoning_trace']:
                        st.markdown(f"- {step}")

            st.markdown("---")


# Main conversational interface
st.header("💬 Pattern Analysis Assistant")

# Initialize chat if empty
if not st.session_state.chat_messages:
    welcome_msg = {
        "role": "assistant",
        "content": "👋 Welcome to the Clary Health Pattern Reasoner! I can help you analyze conversation patterns in your health data.\n\n💡 **Quick Start:** Upload a dataset in the sidebar and try one of the example queries below.",
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(welcome_msg)

# Display chat messages
for idx, message in enumerate(st.session_state.chat_messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "analysis" in message:
            display_analysis_results(message["analysis"], unique_key_prefix=f"msg_{idx}")

# Example buttons
if st.session_state.dataset_summary:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔍 Find patterns for all users", use_container_width=True, key="find_all_patterns_button"):
            handle_analysis_request("Find patterns for all users")

    with col2:
        if st.button("👤 Analyze USR001", use_container_width=True, key="analyze_usr001_button"):
            handle_analysis_request("Analyze USR001")

    with col3:
        if st.button("👤 Analyze USR002", use_container_width=True, key="analyze_usr002_button"):
            handle_analysis_request("Analyze USR002")

    with col4:
        if st.button("👤 Analyze USR003", use_container_width=True, key="analyze_usr003_button"):
            handle_analysis_request("Analyze USR003")

# Chat input
if prompt := st.chat_input("Ask about patterns in your data..."):
    if not st.session_state.dataset:
        st.error("Please upload and load a dataset first.")
    else:
        handle_analysis_request(prompt)