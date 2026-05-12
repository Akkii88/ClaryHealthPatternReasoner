import os
from typing import Optional, AsyncGenerator, List, Dict, Any
from groq import Groq

class LLMClient:
    """Groq-based LLM client for pattern reasoning."""

    # Available models
    AVAILABLE_MODELS = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]

    # Default model
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = self.DEFAULT_MODEL
        self._connection_status = None

        if self.api_key and self.api_key.strip():
            try:
                self.client = Groq(api_key=self.api_key)
                # Test connection
                self._test_connection()
                if self.client:
                    self._connection_status = {
                        "status": "connected",
                        "model": self.model
                    }
            except Exception:
                # Silently fail - don't expose any error details
                self.client = None
                self._connection_status = None
        # If no valid key, client remains None (mock mode)

    def _test_connection(self):
        """Test connection with a simple request."""
        if not self.client:
            return

        try:
            # Test with a simple request
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            # Connection successful
        except Exception:
            # Silently fail - don't expose error details
            self.client = None

    def is_available(self) -> bool:
        """Check if the LLM client is properly configured and available."""
        return self.client is not None

    def is_api_key_configured(self) -> bool:
        """Check if API key is configured (safe to call, doesn't expose key)."""
        return self.api_key is not None and self.api_key.strip() != ""

    def get_model_name(self) -> str:
        """Get the currently selected model name."""
        return self.model

    def set_model(self, model_name: str):
        """Set the model to use for requests."""
        if model_name in self.AVAILABLE_MODELS:
            self.model = model_name
            # Update connection status with new model
            if self._connection_status:
                self._connection_status["model"] = self.model

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.AVAILABLE_MODELS.copy()

    def get_connection_status(self) -> Optional[Dict[str, str]]:
        """
        Get connection status in safe format.
        Returns dict with 'status' and 'model' if connected, None if not.
        """
        return self._connection_status

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Yields:
            Response text chunks
        """
        if not self.is_available():
            # Mock mode - return sample valid JSON response
            mock_response = '''{
  "patterns": [
    {
      "user_id": "mock_user_123",
      "pattern_title": "Sample Escalating Support Pattern",
      "sessions_involved": ["session_001", "session_002"],
      "evidence_timeline": [
        {
          "session_id": "session_001",
          "week": "1",
          "timestamp": "2024-01-01T10:00:00Z",
          "evidence": "Initial medium severity request for setup help"
        },
        {
          "session_id": "session_002",
          "week": "2",
          "timestamp": "2024-01-08T11:00:00Z",
          "evidence": "Follow-up with high severity and urgent tags"
        }
      ],
      "temporal_reasoning": "Pattern shows increasing urgency over two weeks with consistent escalation in severity levels",
      "confidence": "high",
      "confidence_justification": "Clear temporal progression with matching severity increases and urgent language patterns",
      "counter_evidence": ["One unrelated low-severity interaction"],
      "rejected_hypotheses": [
        {
          "hypothesis": "Random unrelated requests",
          "reason_rejected": "Consistent escalation pattern across sessions"
        }
      ],
      "reasoning_trace": [
        "Identified baseline medium severity in week 1",
        "Found escalation to high severity in week 2",
        "Confirmed temporal relationship between sessions",
        "Evaluated alternative explanations"
      ]
    }
  ]
}'''
            if stream:
                # Yield in chunks for streaming simulation
                import asyncio
                for char in mock_response:
                    yield char
                    await asyncio.sleep(0.01)  # Simulate streaming delay
            else:
                yield mock_response
            return

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield response.choices[0].message.content

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            yield error_msg

# Global client instance
llm_client = LLMClient()