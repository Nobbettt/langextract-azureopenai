#!/usr/bin/env python3
"""Comprehensive parameter testing for Azure OpenAI provider.

This script tests each parameter in _AZURE_OPENAI_CONFIG_KEYS to ensure
they work correctly with actual Azure OpenAI API calls.
"""

import json
import os
import sys
from typing import Any

import langextract as lx

from langextract_azureopenai import AzureOpenAILanguageModel

# Test parameters with safe values
TEST_PARAMETERS = {
    'temperature': [0.0, 0.5, 1.0, 2.0],
    'top_p': [0.1, 0.5, 0.9, 1.0],
    'frequency_penalty': [-2.0, -1.0, 0.0, 1.0, 2.0],
    'presence_penalty': [-2.0, -1.0, 0.0, 1.0, 2.0],
    'stop': [["END"], ["\n", "."], "STOP"],
    'logprobs': [True, False],
    'top_logprobs': [0, 1, 3, 5],  # Only valid when logprobs=True
    'seed': [42, 123, 999],
    'user': ["test-user", "demo-123", "user-id-456"],
    'response_format': [{"type": "text"}, {"type": "json_object"}],
    'logit_bias': [
        {},  # Empty bias
        {"50256": -100},  # Ban a specific token (end token)
        {"1820": 5, "262": -5},  # Bias some tokens
    ],
    'stream': [False],  # Note: True requires different handling
    'parallel_tool_calls': [True, False],
}


def check_environment() -> tuple[str, str, str]:
    """Check if required environment variables are set."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not api_key:
        print("❌ AZURE_OPENAI_API_KEY environment variable not set")
        return None, None, None
    if not endpoint:
        print("❌ AZURE_OPENAI_ENDPOINT environment variable not set")
        return None, None, None
    if not version:
        print("❌ AZURE_OPENAI_API_VERSION environment variable not set")
        return None, None, None

    return api_key, endpoint, version


def test_parameter(
    param_name: str, param_value: Any, provider: AzureOpenAILanguageModel
) -> dict[str, Any]:
    """Test a single parameter with the Azure OpenAI provider."""
    print(f"  Testing {param_name}={param_value}...")

    # Simple test prompt
    test_prompt = "Say 'Hello, this is a test response.' and nothing else."

    try:
        # Special handling for logprobs-related parameters
        kwargs = {param_name: param_value}

        # If testing top_logprobs, we need logprobs=True
        if param_name == 'top_logprobs' and param_value > 0:
            kwargs['logprobs'] = True

        # If testing tools or tool_choice, skip for now (requires complex setup)
        if param_name in ['tools', 'tool_choice']:
            return {"status": "skipped", "reason": "Complex tool setup required"}

        # Run inference with the parameter
        results = list(provider.infer([test_prompt], **kwargs))

        if results and results[0] and results[0][0].output:
            response_text = results[0][0].output
            return {
                "status": "success",
                "response_length": len(response_text),
                "response_preview": (
                    response_text[:50] + "..."
                    if len(response_text) > 50
                    else response_text
                ),
            }
        else:
            return {"status": "error", "error": "No response received"}

    except Exception as e:
        error_msg = str(e)
        # Check for specific API errors that might indicate parameter issues
        if "400" in error_msg or "invalid" in error_msg.lower():
            return {"status": "invalid_parameter", "error": error_msg}
        else:
            return {"status": "api_error", "error": error_msg}


def test_parameter_combinations(provider: AzureOpenAILanguageModel) -> dict[str, Any]:
    """Test common parameter combinations."""
    print("\n🔬 Testing parameter combinations...")

    combinations = [
        {
            "name": "Basic generation control",
            "params": {"temperature": 0.7, "top_p": 0.9},
        },
        {
            "name": "Penalty controls",
            "params": {"frequency_penalty": 0.5, "presence_penalty": 0.3},
        },
        {"name": "Reproducible generation", "params": {"seed": 42, "temperature": 0.1}},
        {"name": "Log probabilities", "params": {"logprobs": True, "top_logprobs": 2}},
        {
            "name": "User tracking with stops",
            "params": {"user": "test-combo", "stop": ["\n"]},
        },
        {
            "name": "JSON format with bias",
            "params": {
                "response_format": {"type": "json_object"},
                "logit_bias": {"50256": -50},
            },
        },
    ]

    results = {}
    test_prompt = (
        'Respond with: {"message": "Hello from Azure OpenAI", "status": "success"}'
    )

    for combo in combinations:
        print(f"  Testing: {combo['name']}")
        try:
            result_list = list(provider.infer([test_prompt], **combo['params']))
            if result_list and result_list[0] and result_list[0][0].output:
                results[combo['name']] = {
                    "status": "success",
                    "response_preview": result_list[0][0].output[:100],
                }
            else:
                results[combo['name']] = {"status": "no_response"}
        except Exception as e:
            results[combo['name']] = {"status": "error", "error": str(e)}

    return results


def main():
    """Main test function."""
    print("🧪 Azure OpenAI Parameter Validation Test Suite")
    print("=" * 60)

    # Check environment
    api_key, endpoint, version = check_environment()
    if not api_key or not endpoint or not version:
        print("\nPlease set the required environment variables:")
        print("  export AZURE_OPENAI_API_KEY='your-api-key'")
        print(
            "  export AZURE_OPENAI_ENDPOINT='https://your-endpoint.openai.azure.com/'"
        )
        print("  export AZURE_OPENAI_API_VERSION='2024-12-01-preview'")
        sys.exit(1)

    # Create provider
    try:
        provider = AzureOpenAILanguageModel(
            model_id="azureopenai-test",
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=version,
        )
        print("✅ Provider created successfully")
        print(f"   Deployment: {provider.deployment_name}")
    except Exception as e:
        print(f"❌ Failed to create provider: {e}")
        sys.exit(1)

    # Test results storage
    test_results = {}

    # Test each parameter
    print("\n🔍 Testing individual parameters...")
    print("-" * 40)

    for param_name, test_values in TEST_PARAMETERS.items():
        print(f"\n📋 Testing parameter: {param_name}")
        param_results = {}

        for value in test_values:
            result = test_parameter(param_name, value, provider)
            param_results[str(value)] = result

            # Print result summary
            status = result['status']
            if status == 'success':
                print(f"    ✅ {value}: Success")
            elif status == 'skipped':
                print(f"    ⏭️  {value}: Skipped - {result['reason']}")
            elif status == 'invalid_parameter':
                print(f"    ❌ {value}: Invalid parameter - {result['error']}")
            elif status == 'api_error':
                print(f"    🔥 {value}: API Error - {result['error']}")
            else:
                print(f"    ⚠️  {value}: {status}")

        test_results[param_name] = param_results

    # Test parameter combinations
    combo_results = test_parameter_combinations(provider)
    test_results['combinations'] = combo_results

    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)

    total_tests = 0
    successful_tests = 0

    for param_name, results in test_results.items():
        if param_name == 'combinations':
            continue

        param_success = sum(1 for r in results.values() if r['status'] == 'success')
        param_total = len(results)
        total_tests += param_total
        successful_tests += param_success

        print(f"{param_name:20}: {param_success}/{param_total} passed")

    # Combination results
    combo_success = sum(1 for r in combo_results.values() if r['status'] == 'success')
    combo_total = len(combo_results)
    print(f"{'combinations':20}: {combo_success}/{combo_total} passed")

    total_tests += combo_total
    successful_tests += combo_success

    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    print(
        f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})"
    )

    # Save detailed results
    with open("azure_parameter_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print("\n💾 Detailed results saved to: azure_parameter_test_results.json")

    # Recommendations
    print("\n💡 Recommendations:")
    failed_params = []
    for param_name, results in test_results.items():
        if param_name == 'combinations':
            continue
        if not any(r['status'] == 'success' for r in results.values()):
            failed_params.append(param_name)

    if failed_params:
        print(
            "   ⚠️  Consider removing these parameters from _AZURE_OPENAI_CONFIG_KEYS:"
        )
        for param in failed_params:
            print(f"       - {param}")
    else:
        print("   ✅ All parameters appear to be valid!")

    print("\n🏁 Test completed!")


if __name__ == "__main__":
    main()
