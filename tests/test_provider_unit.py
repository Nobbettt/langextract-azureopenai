"""Unit tests for Azure OpenAI provider."""

import langextract as lx
import pytest

from langextract_azureopenai import AzureOpenAILanguageModel
from langextract_azureopenai.schema import AzureOpenAISchema


@pytest.mark.unit
class TestAzureOpenAIProvider:
    """Test Azure OpenAI provider functionality."""

    def test_provider_initialization(self, mock_azure_credentials, mock_openai_client):
        """Test provider can be initialized with valid credentials."""
        provider = AzureOpenAILanguageModel(
            model_id="azureopenai-gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
        )

        assert provider.model_id == "azureopenai-gpt-4"
        assert provider.deployment_name == "gpt-4"
        assert provider.api_key == "test-key"
        assert provider.azure_endpoint == "https://test.openai.azure.com/"

    def test_deployment_name_extraction(
        self, mock_azure_credentials, mock_openai_client
    ):
        """Test deployment name is correctly extracted from model ID."""
        test_cases = [
            ("azureopenai-gpt-4", "gpt-4"),
            ("azureopenai-gpt-35-turbo", "gpt-35-turbo"),
            ("azureopenai-custom-model", "custom-model"),
            ("direct-deployment", "direct-deployment"),  # No prefix
        ]

        for model_id, expected_deployment in test_cases:
            provider = AzureOpenAILanguageModel(
                model_id=model_id,
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/",
            )
            assert provider.deployment_name == expected_deployment

    def test_parameter_filtering(self, mock_azure_credentials, mock_openai_client):
        """Test that parameter filtering works correctly."""
        # Valid and invalid parameters
        kwargs = {
            'frequency_penalty': 0.5,  # Valid
            'presence_penalty': 0.3,  # Valid
            'invalid_param': 'test',  # Invalid - should be filtered
            'malicious_code': 'rm -rf /',  # Invalid - should be filtered
        }

        provider = AzureOpenAILanguageModel(
            model_id="azureopenai-gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            **kwargs,
        )

        # Check that valid parameters are stored
        assert 'frequency_penalty' in provider._extra_kwargs
        assert 'presence_penalty' in provider._extra_kwargs
        assert provider._extra_kwargs['frequency_penalty'] == 0.5
        assert provider._extra_kwargs['presence_penalty'] == 0.3

        # Check that invalid parameters are filtered out
        assert 'invalid_param' not in provider._extra_kwargs
        assert 'malicious_code' not in provider._extra_kwargs

    def test_schema_support(
        self, mock_azure_credentials, mock_openai_client, sample_extraction_examples
    ):
        """Test schema creation and application."""
        provider = AzureOpenAILanguageModel(
            model_id="azureopenai-gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
        )

        # Test schema class
        schema_class = provider.get_schema_class()
        assert schema_class == AzureOpenAISchema

        # Test schema creation from examples
        schema = AzureOpenAISchema.from_examples(sample_extraction_examples)
        assert schema.schema_dict is not None
        assert 'type' in schema.schema_dict
        assert 'properties' in schema.schema_dict

        # Test schema application
        provider.apply_schema(schema)
        # Schema should be stored in base class
        assert provider._schema == schema
        # When schema is applied, provider should enable structured output mode
        assert getattr(provider, '_enable_structured_output', False) is True

    def test_inference_basic(self, mock_azure_credentials, mock_openai_client):
        """Test basic inference functionality."""
        provider = AzureOpenAILanguageModel(
            model_id="azureopenai-gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
        )

        prompts = ["Test prompt"]
        results = list(provider.infer(prompts))

        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].output == "Mock response content"
        assert results[0][0].score == 1.0

        # Verify API was called correctly
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs['model'] == 'gpt-4'  # deployment name
        assert call_kwargs['messages'] == [{'role': 'user', 'content': 'Test prompt'}]

    def test_inference_with_parameters(
        self, mock_azure_credentials, mock_openai_client
    ):
        """Test inference with various parameters."""
        provider = AzureOpenAILanguageModel(
            model_id="azureopenai-gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
        )

        prompts = ["Test prompt"]
        results = list(
            provider.infer(
                prompts,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                max_completion_tokens=100,
            )
        )

        assert len(results) == 1

        # Check that parameters were passed to API
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['top_p'] == 0.9
        assert call_kwargs['frequency_penalty'] == 0.1
        assert call_kwargs['presence_penalty'] == 0.1
        assert call_kwargs['max_completion_tokens'] == 100

    def test_unsupported_parameters_raise(
        self, mock_azure_credentials, mock_openai_client
    ):
        """Unsupported parameters should raise InferenceConfigError."""
        # Unsupported at construction
        with pytest.raises(lx.exceptions.InferenceConfigError):
            AzureOpenAILanguageModel(
                model_id="azureopenai-gpt-4",
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/",
                stream=True,
            )

        # Unsupported at inference time
        provider = AzureOpenAILanguageModel(
            model_id="azureopenai-gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
        )
        with pytest.raises(lx.exceptions.InferenceConfigError):
            list(provider.infer(["hello"], stream=True))

    def test_missing_credentials_error(self):
        """Test that missing credentials raise appropriate errors."""
        # Ensure env does not accidentally satisfy credentials for this test
        from unittest.mock import patch

        with patch.dict(
            'os.environ',
            {
                'AZURE_OPENAI_API_KEY': '',
                'AZURE_OPENAI_API_VERSION': '',
            },
            clear=False,
        ):
            with pytest.raises(
                lx.exceptions.InferenceConfigError, match="API key not provided"
            ):
                AzureOpenAILanguageModel(
                    model_id="azureopenai-gpt-4",
                    azure_endpoint="https://test.openai.azure.com/",
                    # Missing api_key
                )

        with patch.dict(
            'os.environ',
            {
                'AZURE_OPENAI_ENDPOINT': '',
                'AZURE_OPENAI_API_VERSION': '',
            },
            clear=False,
        ):
            with pytest.raises(
                lx.exceptions.InferenceConfigError, match="endpoint not provided"
            ):
                AzureOpenAILanguageModel(
                    model_id="azureopenai-gpt-4",
                    api_key="test-key",
                    # Missing azure_endpoint
                )
