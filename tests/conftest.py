"""Pytest configuration and fixtures for langextract-azureopenai tests.

Notes:
- `tests/test_azure_parameters.py` is a script-style integration validator that is
  executed directly (e.g., via `python tests/test_azure_parameters.py`) when you
  have real Azure credentials. It is not a pytest test module and should be
  excluded from pytest collection to avoid fixture resolution errors.
"""

import os
from unittest.mock import patch

import pytest

# Exclude the script-style integration validator from pytest collection. This
# prevents errors like "fixture 'param_name' not found" during unit test runs.
# It remains executable directly via `python tests/test_azure_parameters.py`.
collect_ignore = ["test_azure_parameters.py"]


@pytest.fixture
def mock_azure_credentials():
    """Mock Azure OpenAI credentials for testing."""
    with patch.dict(
        os.environ,
        {
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_VERSION': '2024-12-01-preview',
        },
    ):
        yield


@pytest.fixture
def mock_openai_client():
    """Mock the AzureOpenAI client to avoid real API calls."""
    with patch('openai.AzureOpenAI') as mock_client_class:
        mock_client = mock_client_class.return_value

        # Mock successful response
        mock_response = type('MockResponse', (), {})()
        mock_response.choices = [type('MockChoice', (), {})()]
        mock_response.choices[0].message = type('MockMessage', (), {})()
        mock_response.choices[0].message.content = "Mock response content"

        mock_client.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def sample_extraction_examples():
    """Sample extraction examples for testing."""
    import langextract as lx

    return [
        lx.data.ExampleData(
            text="John Smith works at Microsoft in Seattle.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="person",
                    extraction_text="John Smith",
                    attributes={"role": "employee"},
                ),
                lx.data.Extraction(
                    extraction_class="organization",
                    extraction_text="Microsoft",
                    attributes={"type": "company"},
                ),
                lx.data.Extraction(
                    extraction_class="location",
                    extraction_text="Seattle",
                    attributes={"type": "city"},
                ),
            ],
        )
    ]
