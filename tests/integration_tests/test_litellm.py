"""Test ChatLiteLLM chat model."""

from typing import Type

import pytest
from langchain_core.messages import HumanMessage
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_litellm.chat_models import ChatLiteLLM


class TestChatLiteLLMIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatLiteLLM]:
        return ChatLiteLLM

    @property
    def chat_model_params(self) -> dict:
        return {
            "custom_llm_provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": "<your_api_key>",
            "max_retries": 1,
        }

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def supports_json_mode(self) -> bool:
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False


@pytest.mark.integration
class TestUsageMetadataBedrock:
    """Test that usage metadata is returned for non-OpenAI providers."""

    model = "bedrock/amazon.nova-micro-v1:0"
    messages = [HumanMessage(content="Say hi in exactly 3 words")]

    def test_non_streaming_usage_metadata(self) -> None:
        llm = ChatLiteLLM(model=self.model)
        result = llm.invoke(self.messages)
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] > 0
        assert result.usage_metadata["output_tokens"] > 0
        assert result.usage_metadata["total_tokens"] > 0

    def test_streaming_usage_metadata(self) -> None:
        llm = ChatLiteLLM(model=self.model)
        chunks = list(llm.stream(self.messages))
        usage_chunks = [c for c in chunks if c.usage_metadata]
        assert len(usage_chunks) >= 1
        usage = usage_chunks[-1].usage_metadata
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
        assert usage["total_tokens"] > 0
