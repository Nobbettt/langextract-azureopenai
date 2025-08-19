# LangExtract Anthropic Provider

A provider plugin for [LangExtract](https://github.com/google/langextract) that integrates Anthropic's Claude API for robust, structured information extraction.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- **Native Anthropic API**: Uses the official `anthropic` Python SDK for Claude models.
- **Safe parameter handling**: Whitelist filtering; unsupported params raise clear errors.
- **Concurrent batching**: Parallel inference for multi-prompt workloads.
- **Schema-aware**: Optional structured output mode (JSON) from LangExtract examples.
- **Modern packaging**: `pyproject.toml` with Hatch; works well with `uv`.

## Installation

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv add langextract-anthropic
```

### Using pip

```bash
pip install langextract-anthropic
```

### From Source

```bash
git clone <repository-url>
cd langextract-anthropic
uv sync
```

## Quick Start

### 1. Set up Anthropic API credentials

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### 2. Use with LangExtract

```python
import langextract as lx

# Define extraction examples
examples = [
    lx.ExampleData(
        text="John Smith works at Microsoft in Seattle.",
        extractions=[
            lx.ExtractionData(
                extraction_class="Person",
                attributes={"name": "John Smith"}
            ),
            lx.ExtractionData(
                extraction_class="Organization", 
                attributes={"name": "Microsoft"}
            ),
            lx.ExtractionData(
                extraction_class="Location",
                attributes={"name": "Seattle"}
            ),
        ],
    ),
]

# Extract information using Anthropic Claude
result = lx.extract(
    text_or_documents="Sarah Johnson is a data scientist at Google in Mountain View.",
    prompt_description="Extract people, organizations, and locations.",
    examples=examples,
    model_id="anthropic-claude-3-5-sonnet-latest",
    temperature=0.1,
    max_tokens=512,
)

print(result.extractions)
```

## Supported Models

This provider supports all Anthropic Claude models:

- `claude-3-5-sonnet-latest` (recommended)
- `claude-3-5-sonnet-20241022`  
- `claude-3-5-haiku-latest`
- `claude-3-opus-latest`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Model ID Format

Use the `anthropic-` prefix or specify the model name directly:

- `anthropic-claude-3-5-sonnet-latest` → Uses model: `claude-3-5-sonnet-latest`
- `anthropic-claude-3-opus-latest` → Uses model: `claude-3-opus-latest`
- `claude-3-5-sonnet-latest` → Uses model directly

## Configuration Parameters

### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_id` | `str` | Model identifier | `"claude-3-5-sonnet-latest"` |
| `api_key` | `str` | Anthropic API key | `ANTHROPIC_API_KEY` env var |
| `temperature` | `float` | Controls randomness (0-1) | `None` |
| `max_workers` | `int` | Parallel request workers | `10` |

### Anthropic API Parameters

| Parameter | Type | Description | Range |
|-----------|------|-------------|--------|
| `max_tokens` | `int` | Maximum tokens to generate | 1-8192 |
| `temperature` | `float` | Sampling temperature | 0.0-1.0 |
| `top_p` | `float` | Nucleus sampling | 0.0-1.0 |
| `top_k` | `int` | Top-k sampling | 0-200 |
| `stop_sequences` | `list[str]` | Stop sequences | Max 4 items |
| `metadata` | `dict` | Request tracking metadata | - |

### Usage Examples

```python
# Basic extraction
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="anthropic-claude-3-5-sonnet-latest",
)

# With custom parameters
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="anthropic-claude-3-5-sonnet-latest",
    temperature=0.3,
    max_tokens=1000,
    top_p=0.9,
    stop_sequences=["END", "STOP"],
    metadata={"user_id": "user123"},
)
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes |

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd langextract-anthropic

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --dev
```

### Running Tests

```bash
# Run unit tests (no API calls)
uv run pytest tests/ -m "unit"

# Run integration tests (requires ANTHROPIC_API_KEY)
uv run pytest tests/ -m "integration" 

# Run all tests with coverage
uv run pytest tests/ --cov=langextract_anthropic --cov-report=html
```

### Development Commands

```bash
# Format code
uv run black langextract_anthropic tests
uv run isort langextract_anthropic tests

# Lint code
uv run ruff check langextract_anthropic tests
uv run mypy langextract_anthropic

# Build package
uv build

# Bump version
python scripts/bump_version.py patch  # 0.1.0 -> 0.1.1
python scripts/bump_version.py minor  # 0.1.0 -> 0.2.0
python scripts/bump_version.py major  # 0.1.0 -> 1.0.0
```

## Testing

This provider includes comprehensive testing:

- **Unit tests**: Mock-based testing of provider logic
- **Parameter tests**: Validation of API parameter filtering
- **Integration tests**: Real API testing (requires credentials)

```bash
# Set up test environment
export ANTHROPIC_API_KEY="your-api-key"

# Run specific test categories
uv run pytest tests/test_provider_unit.py -v
uv run pytest tests/test_parameter_filtering.py -v
uv run pytest tests/test_anthropic_integration.py -v  # requires API key
```

## Error Handling

The provider provides clear error messages for common issues:

```python
try:
    result = lx.extract(...)
except lx.exceptions.InferenceConfigError as e:
    # Configuration errors (missing API key, invalid params)
    print(f"Configuration error: {e}")
except lx.exceptions.InferenceRuntimeError as e:
    # Runtime errors (API failures, network issues)
    print(f"Runtime error: {e}")
    print(f"Original error: {e.original}")
```

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   InferenceConfigError: Anthropic API key not provided
   ```
   Solution: Set `ANTHROPIC_API_KEY` environment variable or pass `api_key` parameter.

2. **Invalid Model Name**
   ```
   AnthropicAPIError: model not found
   ```
   Solution: Use a valid Claude model name (see supported models above).

3. **Rate Limiting**
   ```
   AnthropicAPIError: 429 Too Many Requests
   ```
   Solution: Reduce `max_workers` or add retry logic in your application.

4. **Token Limit Exceeded**
   ```
   AnthropicAPIError: maximum context length exceeded
   ```
   Solution: Reduce input text length or increase `max_tokens` parameter.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.