# Azure OpenAI Provider Testing Suite

This directory contains comprehensive tests for the Azure OpenAI provider, including parameter validation and API compatibility testing.

## Test Files

### 1. `test_plugin.py` 
Basic provider functionality test (auto-generated):
- Provider registration and pattern matching
- Schema support validation  
- Factory integration testing

```bash
python3 test_plugin.py
```

### 2. `test_parameter_filtering.py`
Parameter security and filtering logic test (no credentials required):
- Validates that invalid parameters are filtered out
- Ensures valid parameters are accepted and stored
- Tests parameter passing to API calls
- Verifies runtime parameter overrides

```bash
python3 test_parameter_filtering.py
```

### 3. `test_azure_parameters.py` 
Comprehensive Azure OpenAI API parameter compatibility test (requires credentials):
- Tests each parameter with actual Azure OpenAI API calls
- Validates parameter combinations
- Measures success rates for each parameter
- Generates detailed test report

```bash
# Set environment variables first
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

python3 test_azure_parameters.py
```

## Tested Parameters

The following Azure OpenAI parameters are validated:

| Parameter | Type | Description | Test Coverage |
|-----------|------|-------------|---------------|
| `frequency_penalty` | number (-2.0 to 2.0) | Reduces repetition based on frequency | ✅ Multiple values |
| `presence_penalty` | number (-2.0 to 2.0) | Reduces repetition based on presence | ✅ Multiple values |
| `stop` | string/array | Stop sequences | ✅ Various formats |
| `logprobs` | boolean | Return log probabilities | ✅ True/False |
| `top_logprobs` | integer (0-5) | Number of top tokens | ✅ Multiple values |
| `seed` | integer | Deterministic generation | ✅ Various seeds |
| `user` | string | User identifier | ✅ Various formats |
| `response_format` | object | Output format control | ✅ Text/JSON modes |
| `tools` | array | Function calling tools | ⏭️ Skipped (complex) |
| `tool_choice` | string | Tool selection | ⏭️ Skipped (complex) |
| `logit_bias` | object | Token probability bias | ✅ Various bias maps |
| `stream` | boolean | Streaming responses | ✅ False only |
| `parallel_tool_calls` | boolean | Parallel function calls | ✅ True/False |

## Test Results

### Parameter Filtering Test Results
```
✅ Security: Invalid parameters filtered out (4/4)
✅ Functionality: Valid parameters accepted (13/13) 
✅ API Integration: Parameters correctly passed to API
✅ Override Logic: Runtime parameters override stored values
```

### API Compatibility Test Results
Run `python3 test_azure_parameters.py` to generate:
- Individual parameter success rates
- Parameter combination testing
- Detailed JSON report (`azure_parameter_test_results.json`)
- Recommendations for parameter support

## Usage in Production

Based on test results, you can safely use any of the whitelisted parameters:

```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    model_id="azureopenai-gpt-4",
    # Generation control
    temperature=0.7,
    top_p=0.9,
    # Repetition control  
    frequency_penalty=0.1,
    presence_penalty=0.1,
    # Reproducibility
    seed=42,
    # Safety
    stop=["\n", "END"],
    user="user-123",
    # Advanced features
    logprobs=True,
    top_logprobs=2,
    logit_bias={"50256": -100}  # Reduce end-token probability
)
```

## Security Benefits

The parameter filtering system provides:
- **Input Validation**: Only whitelisted parameters reach the API
- **Injection Prevention**: Malicious parameters are filtered out
- **Error Avoidance**: Invalid parameters don't cause API failures  
- **Future Compatibility**: Easy to add new parameters as Azure OpenAI evolves

## Adding New Parameters

To add support for new Azure OpenAI parameters:

1. Add the parameter name to `_AZURE_OPENAI_CONFIG_KEYS` in `provider.py`
2. Add test cases to `test_azure_parameters.py`
3. Run the test suite to validate compatibility
4. Update this documentation

## Continuous Testing

These tests should be run:
- **Before releases** to ensure parameter compatibility
- **After Azure OpenAI API updates** to validate continued support
- **When adding new parameters** to verify functionality

The test suite is designed to catch regressions and ensure reliable operation with Azure OpenAI services.