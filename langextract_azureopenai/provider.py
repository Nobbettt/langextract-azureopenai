"""Provider implementation for AzureOpenAI."""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Iterator, Sequence
from typing import Any, Final

import langextract as lx  # type: ignore[import-untyped]

from langextract_azureopenai.schema import AzureOpenAISchema

# Azure OpenAI Chat Completions API supported parameters
# Based on: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference
_AZURE_OPENAI_CONFIG_KEYS: Final[set[str]] = {
    'frequency_penalty',  # Number between -2.0 and 2.0
    'presence_penalty',  # Number between -2.0 and 2.0
    'stop',  # String or array of stop sequences
    'logprobs',  # Whether to return log probabilities
    'top_logprobs',  # Number of most likely tokens (0-5)
    'seed',  # Random seed for deterministic outputs
    'user',  # Unique identifier for end-user
    'response_format',  # Output format (text, json_object, json_schema)
    'tools',  # Array of tools/functions model can call (unsupported)
    'tool_choice',  # Controls which tools to use (unsupported)
    'logit_bias',  # Map of token IDs to bias scores (-100 to 100)
    'stream',  # Whether to stream partial responses (unsupported)
    'parallel_tool_calls',  # Whether to enable parallel function calling (unsupported)
}


@lx.providers.registry.register(r'^azureopenai', priority=10)
class AzureOpenAILanguageModel(lx.inference.BaseLanguageModel):
    """Language model inference using Azure OpenAI's API with structured output.

    This provider handles model IDs matching: ['^azureopenai']
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        temperature: float | None = None,
        max_workers: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI language model.

        Args:
            model_id: The Azure OpenAI model ID to use (e.g., 'azureopenai-gpt-4').
            api_key: API key for Azure OpenAI service.
            azure_endpoint: Azure OpenAI endpoint URL.
            api_version: API version to use.
            deployment_name: Deployment name. If None, extracted from model_id.
            temperature: Sampling temperature.
            max_workers: Maximum number of parallel API calls.
            **kwargs: Additional parameters passed to the Azure OpenAI API.
        """
        # Lazy import: OpenAI package required
        try:
            # pylint: disable=import-outside-toplevel
            from openai import AzureOpenAI
        except ImportError as e:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI provider requires openai package. '
                'Install with: pip install openai>=1.0.0'
            ) from e

        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        self.azure_endpoint = azure_endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT')
        # api_version is mandatory: from arg or env
        self.api_version = api_version or os.environ.get('AZURE_OPENAI_API_VERSION')
        self.temperature = temperature
        self.max_workers = max_workers
        self._response_schema: dict[str, Any] | None = None
        self._enable_structured_output: bool = False

        # Extract deployment name from model_id if not provided
        if deployment_name:
            self.deployment_name = deployment_name
        else:
            # Extract deployment name by removing 'azureopenai-' prefix
            if isinstance(model_id, str) and model_id.startswith('azureopenai-'):
                self.deployment_name = model_id[len('azureopenai-') :]
            else:
                self.deployment_name = model_id or ''

        # Validate required parameters
        if not self.api_key:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY '
                'environment variable or pass api_key parameter.'
            )
        if not self.azure_endpoint:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT '
                'environment variable or pass azure_endpoint parameter.'
            )
        if not self.api_version:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI API version not provided. Set AZURE_OPENAI_API_VERSION '
                'environment variable or pass api_version parameter.'
            )

        # Reject unsupported parameters early if provided at construction
        unsupported = {'stream', 'tools', 'tool_choice', 'parallel_tool_calls'}
        for key in unsupported.intersection(set((kwargs or {}).keys())):
            raise lx.exceptions.InferenceConfigError(
                f"Unsupported parameter provided: {key}. This provider does not support it yet."
            )

        # Initialize the Azure OpenAI client
        self._client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
        )

        # Filter extra kwargs to only include valid Azure OpenAI API parameters
        self._extra_kwargs = {
            k: v for k, v in (kwargs or {}).items() if k in _AZURE_OPENAI_CONFIG_KEYS
        }

    @classmethod
    def get_schema_class(cls) -> type[AzureOpenAISchema]:
        """Tell LangExtract about our schema support."""
        return AzureOpenAISchema

    def apply_schema(self, schema_instance: object | None) -> None:
        """Apply or clear schema configuration."""
        super().apply_schema(schema_instance)
        if schema_instance is None:
            self._response_schema = None
            self._enable_structured_output = False
            return
        if isinstance(schema_instance, AzureOpenAISchema):
            cfg = schema_instance.to_provider_config()
            # Support both keys for compatibility with older schema impls
            self._response_schema = cfg.get('response_schema')
            self._enable_structured_output = bool(
                cfg.get('enable_structured_output') or cfg.get('structured_output')
            )

    def _process_single_prompt(
        self, prompt: str, config: dict[str, Any]
    ) -> lx.inference.ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            # Apply stored kwargs that weren't already set in config
            for key, value in self._extra_kwargs.items():
                if key not in config and value is not None:
                    config[key] = value

            # Build messages. When structured output is enabled, include an
            # explicit system instruction mentioning "json" to satisfy the
            # API requirement for response_format={'type': 'json_object'}.
            if self._enable_structured_output:
                messages: list[dict[str, str]] = [
                    {
                        'role': 'system',
                        'content': (
                            'You are a helpful assistant that outputs JSON. '
                            'Return only a valid JSON object (no code fences).'
                        ),
                    },
                    {'role': 'user', 'content': prompt},
                ]
            else:
                messages = [{'role': 'user', 'content': prompt}]

            api_params: dict[str, Any] = {
                'model': self.deployment_name,
                'messages': messages,
            }

            # Only set temperature if explicitly provided
            temp = config.get('temperature', self.temperature)
            if temp is not None:
                api_params['temperature'] = temp

            # Enable JSON mode when structured output is requested
            if self._enable_structured_output:
                api_params['response_format'] = {'type': 'json_object'}
                # If strict JSON Schema mode is desired and supported, integrate here:
                # if self._response_schema:
                #     api_params['response_format'] = {
                #         'type': 'json_schema',
                #         'json_schema': self._response_schema,
                #     }

            # Apply standard configuration parameters
            if (v := config.get('max_completion_tokens')) is not None:
                api_params['max_completion_tokens'] = v
            if (v := config.get('top_p')) is not None:
                api_params['top_p'] = v

            # Apply Azure OpenAI-specific parameters from whitelist
            # Reject unsupported params if present
            for key in _AZURE_OPENAI_CONFIG_KEYS:
                if (
                    key in {'stream', 'tools', 'tool_choice', 'parallel_tool_calls'}
                    and key in config
                ):
                    raise lx.exceptions.InferenceConfigError(
                        f"Unsupported parameter provided: {key}. This provider does not support it yet."
                    )
                if (v := config.get(key)) is not None:
                    api_params[key] = v

            response = self._client.chat.completions.create(**api_params)

            # Extract the response text using the v1.x response format
            output_text = response.choices[0].message.content

            return lx.inference.ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            raise lx.exceptions.InferenceRuntimeError(
                f'Azure OpenAI API error: {str(e)}', original=e
            ) from e

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[lx.inference.ScoredOutput]]:
        """Runs inference on a list of prompts via Azure OpenAI's API.

        Args:
            batch_prompts: A list of string prompts.
            **kwargs: Additional generation params (temperature, top_p, etc.)

        Yields:
            Lists of ScoredOutputs.
        """
        config: dict[str, Any] = {}

        # Handle standard parameters explicitly
        temp = kwargs.get('temperature', self.temperature)
        if temp is not None:
            config['temperature'] = temp
        if 'max_completion_tokens' in kwargs:
            config['max_completion_tokens'] = kwargs['max_completion_tokens']
        if 'top_p' in kwargs:
            config['top_p'] = kwargs['top_p']

        # Handle all other whitelisted Azure OpenAI parameters
        handled_keys = {'temperature', 'max_completion_tokens', 'top_p'}
        for key, value in kwargs.items():
            if (
                key not in handled_keys
                and key in _AZURE_OPENAI_CONFIG_KEYS
                and value is not None
            ):
                # Reject unsupported params at runtime
                if key in {'stream', 'tools', 'tool_choice', 'parallel_tool_calls'}:
                    raise lx.exceptions.InferenceConfigError(
                        f"Unsupported parameter provided: {key}. This provider does not support it yet."
                    )
                config[key] = value

        # Use parallel processing for batches larger than 1
        if len(batch_prompts) > 1 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(batch_prompts))
            ) as executor:
                future_to_index = {
                    executor.submit(
                        self._process_single_prompt, prompt, config.copy()
                    ): i
                    for i, prompt in enumerate(batch_prompts)
                }

                results: list[lx.inference.ScoredOutput | None] = [None] * len(
                    batch_prompts
                )
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        raise RuntimeError(f'Parallel inference error: {str(e)}') from e

                for result in results:
                    if result is None:
                        raise RuntimeError('Failed to process one or more prompts')
                    yield [result]
        else:
            # Sequential processing for single prompt or worker
            for prompt in batch_prompts:
                result = self._process_single_prompt(prompt, config.copy())
                yield [result]
