# LangExtract Azure OpenAI Provider

A provider plugin for [LangExtract](https://github.com/google/langextract) that integrates the Azure OpenAI Chat Completions API for robust, structured information extraction.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- **Native Azure OpenAI**: Uses the official `openai` Python SDK with Azure endpoints.
- **Safe parameter handling**: Whitelist filtering; unsupported params raise clear errors.
- **Concurrent batching**: Parallel inference for multi-prompt workloads.
- **Schema-aware**: Optional structured output mode (JSON mode) from LangExtract examples.
- **Modern packaging**: `pyproject.toml` with Hatch; works well with `uv`.

## Installation

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv add langextract-azureopenai
```

### Using pip

```bash
pip install langextract-azureopenai
```

### From Source

```bash
git clone <repository-url>
cd langextract-azureopenai
uv sync
```

## Quick Start

### 1. Set up Azure OpenAI credentials

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"  # or your current API version
```

### 2. Basic Usage

```python
import os
import langextract as lx

# Explicit configuration is recommended
config = lx.factory.ModelConfig(
    model_id="azureopenai-gpt-4.1",  # your Azure deployment name
    provider="AzureOpenAILanguageModel",
    provider_kwargs={
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    },
)

prompt = "Extract people, organizations, and locations from the text."
examples = [
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

result = lx.extract(
    text_or_documents="Sarah Johnson is the CEO of TechCorp in San Francisco.",
    prompt_description=prompt,
    examples=examples,
    config=config,
)

for e in result.extractions:
    print(e.extraction_class, "->", e.extraction_text, e.attributes)
```

### 3. Advanced Parameters

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    config=config,                 # reuse the explicit configuration
    # Azure OpenAI generation params
    temperature=0.3,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    seed=42,
    user="user-123",
    logprobs=True,
    top_logprobs=2,
)
```

## Supported Model IDs

The provider handles model IDs with the pattern `^azureopenai`:

- `azureopenai-gpt-4` → Uses deployment: `gpt-4`
- `azureopenai-gpt-35-turbo` → Uses deployment: `gpt-35-turbo` 
- `azureopenai-your-custom-deployment` → Uses deployment: `your-custom-deployment`

You can also specify the deployment name explicitly:

```python
result = lx.extract(
    # ... other parameters
    model_id="azureopenai-any-name",
    provider_kwargs={"deployment_name": "your-actual-deployment-name"}
)
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | ✅ Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | ✅ Yes |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version (e.g., `2024-12-01-preview`) | ✅ Yes |

## Parameters

- Supported: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `logprobs`, `top_logprobs`, `seed`, `user`, `logit_bias`, and advanced `response_format`.
- Unsupported (raises `InferenceConfigError`): `stream`, `tools`, `tool_choice`, `parallel_tool_calls`.

Notes:
- When schema constraints are enabled via examples, the provider sets `response_format={"type": "json_object"}` to encourage valid JSON output. Strict JSON Schema mode is not enabled at this time.

## Development

### Prerequisites

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation  
uv --version
```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd langextract-azureopenai

# Install dependencies
uv sync

# Install development dependencies
uv sync --group dev
```

### Testing

```bash
# Run unit tests
uv run pytest tests/ -v

# Run parameter filtering tests (no credentials required)
uv run python tests/test_parameter_filtering.py

# Run full Azure API tests (requires credentials)
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
uv run python tests/test_azure_parameters.py

# Run with coverage
uv run pytest tests/ --cov=langextract_azureopenai --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black .
uv run isort .

# Lint code  
uv run ruff check .

# Type checking
uv run mypy langextract_azureopenai
```

### Building and Publishing

```bash
# Build package
uv build

# Check build
ls dist/

# Publish to PyPI (requires API token)
uv publish --token your-pypi-token
```

### Version Management

```bash
# Bump version
python scripts/bump_version.py patch   # 0.1.0 -> 0.1.1
python scripts/bump_version.py minor   # 0.1.0 -> 0.2.0  
python scripts/bump_version.py major   # 0.1.0 -> 1.0.0
```

### Developer Scripts

For a quick overview of the helper scripts (testing, releasing, and versioning), see:

- `scripts/README.md` — summaries and usage for `bump_version.py`, `run_tests.py`, `check_build.py`, and `release.py`.

## Examples

See the `examples/` directory for complete usage examples:

- `examples/example_usage.py` - Basic entity extraction
- `tests/test_azure_parameters.py` - Parameter compatibility testing
- `tests/test_parameter_filtering.py` - Security validation

## Testing

The package includes a comprehensive test suite:

- **Unit Tests**: Core functionality without API calls
- **Integration Tests**: Real Azure OpenAI API testing  
- **Security Tests**: Parameter filtering validation
- **Performance Tests**: Batch processing and concurrency

Run specific test categories:

```bash
# Unit tests only
uv run pytest tests/ -m "unit"

# Integration tests (requires credentials)
uv run pytest tests/ -m "integration"
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `uv run pytest`
5. Format code: `uv run black . && uv run isort .`
6. Submit a pull request

## Troubleshooting

### Common Issues

**ImportError after installation:**
```bash
# Ensure package is properly installed
uv sync
uv run python -c "import langextract_azureopenai; print('✅ Import successful')"
```

**Authentication errors:**
```bash
# Verify credentials are set
echo $AZURE_OPENAI_API_KEY
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_API_VERSION

# Test credentials
uv run python tests/test_provider_basic.py
```

**Parameter errors:**
```bash
# Check parameter compatibility
uv run python tests/test_azure_parameters.py
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Links

- [LangExtract Documentation](https://github.com/google/langextract)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/)
- [UV Documentation](https://docs.astral.sh/uv/)

---

**Happy Extracting with Azure OpenAI! 🚀**
