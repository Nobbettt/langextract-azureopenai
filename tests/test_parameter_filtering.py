#!/usr/bin/env python3
"""Parameter filtering tests without requiring API credentials.

Validates that:
- Valid parameters are accepted and stored
- Invalid/unsupported parameters are filtered or rejected
- Parameters are correctly passed to the API call structure
"""

import os
from unittest.mock import MagicMock, patch

import langextract as lx
import pytest

from langextract_azureopenai import AzureOpenAILanguageModel
from langextract_azureopenai.provider import _AZURE_OPENAI_CONFIG_KEYS


@pytest.mark.unit
def test_parameter_filtering():
    """Valid params are kept, invalid are dropped; API call receives only allowed values."""
    test_kwargs = {
        # Valid parameters (should be kept)
        'temperature': 0.7,
        'frequency_penalty': 0.5,
        'presence_penalty': 0.3,
        'stop': ["\n"],
        'logprobs': True,
        'top_logprobs': 2,
        'seed': 42,
        'user': "test-user",
        'logit_bias': {"50256": -100},
        # Invalid parameters (should be filtered out)
        'invalid_param': "should_be_filtered",
        'random_setting': 123,
        'malicious_code': "rm -rf /",
        'internal_langextract_param': "internal",
    }

    with patch.dict(
        os.environ,
        {
            'AZURE_OPENAI_API_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_VERSION': '2024-12-01-preview',
        },
    ):
        with patch('openai.AzureOpenAI') as mock_azure_client:
            mock_client = MagicMock()
            mock_azure_client.return_value = mock_client

            provider = AzureOpenAILanguageModel(
                model_id="azureopenai-test", **test_kwargs
            )
            assert provider is not None

            stored_params = provider._extra_kwargs
            expected_valid = {
                'frequency_penalty',
                'presence_penalty',
                'stop',
                'logprobs',
                'top_logprobs',
                'seed',
                'user',
                'logit_bias',
            }
            expected_invalid = {
                'invalid_param',
                'random_setting',
                'malicious_code',
                'internal_langextract_param',
            }

            assert expected_valid.issubset(stored_params.keys())
            assert not expected_invalid.intersection(stored_params.keys())

            # Mock the API response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_client.chat.completions.create.return_value = mock_response

            runtime_kwargs = {
                'temperature': 0.9,  # override
                'top_p': 0.8,
                'invalid_runtime': 'should_be_filtered',
            }
            results = list(provider.infer(["Test prompt"], **runtime_kwargs))
            assert results and results[0][0].output == "Test response"

            api_params = mock_client.chat.completions.create.call_args[1]
            assert api_params['temperature'] == 0.9
            assert api_params['top_p'] == 0.8
            assert api_params['frequency_penalty'] == 0.5
            assert api_params['presence_penalty'] == 0.3
            assert 'invalid_runtime' not in api_params


@pytest.mark.unit
def test_individual_parameters_supported_vs_unsupported():
    """Each whitelisted parameter is either accepted (stored) or rejected (raises)."""
    test_values = {
        'frequency_penalty': 0.5,
        'presence_penalty': 0.3,
        'stop': ["END"],
        'logprobs': True,
        'top_logprobs': 2,
        'seed': 42,
        'user': "test-user",
        'response_format': {"type": "text"},
        'tools': [],
        'tool_choice': "auto",
        'logit_bias': {},
        'stream': False,
        'parallel_tool_calls': True,
    }

    with patch.dict(
        os.environ,
        {
            'AZURE_OPENAI_API_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_VERSION': '2024-12-01-preview',
        },
    ):
        with patch('openai.AzureOpenAI'):
            for param_name in sorted(_AZURE_OPENAI_CONFIG_KEYS):
                val = test_values.get(param_name, 'x')
                if param_name in {
                    'stream',
                    'tools',
                    'tool_choice',
                    'parallel_tool_calls',
                }:
                    with pytest.raises(lx.exceptions.InferenceConfigError):
                        AzureOpenAILanguageModel(
                            model_id="azureopenai-test", **{param_name: val}
                        )
                else:
                    provider = AzureOpenAILanguageModel(
                        model_id="azureopenai-test", **{param_name: val}
                    )
                    # accepted params should be stored in _extra_kwargs or handled specially
                    if param_name != 'response_format':
                        assert param_name in provider._extra_kwargs
