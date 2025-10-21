# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Factory for creating appropriate tokenizers based on model name and backend.
Includes caching to avoid repeated downloads and initialization.
"""

from typing import Optional
from dimos.agents.tokenizer.base import AbstractTokenizer
from dimos.agents.tokenizer.openai_tokenizer import OpenAITokenizer
from dimos.agents.tokenizer.huggingface_tokenizer import HuggingFaceTokenizer
from dimos.agents.tokenizer.model_mappings import OLLAMA_TO_HF_TOKENIZER
from dimos.utils.logging_config import setup_logger

# Initialize logger
logger = setup_logger("dimos.agents.tokenizer.factory")

# Global cache for tokenizers to avoid repeated initialization
_TOKENIZER_CACHE = {}


def get_tokenizer_for_model(
    model_name: str,
    backend: str = 'openai',
    force_new: bool = False
) -> AbstractTokenizer:
    """
    Get the appropriate tokenizer for a given model and backend.
    
    Args:
        model_name: Name of the model (e.g., 'gpt-4o', 'phi4:14b', 'qwen2.5-coder:32b')
        backend: Backend type ('openai', 'ollama', 'huggingface', etc.)
        force_new: If True, bypass cache and create a new tokenizer instance
        
    Returns:
        AbstractTokenizer: Appropriate tokenizer instance for the model
        
    Examples:
        >>> # OpenAI model - uses tiktoken
        >>> tokenizer = get_tokenizer_for_model('gpt-4o', backend='openai')
        
        >>> # Ollama model - uses HuggingFace tokenizer if mapped
        >>> tokenizer = get_tokenizer_for_model('phi4:14b', backend='ollama')
        
        >>> # Unmapped Ollama model - falls back to OpenAI with cl100k_base
        >>> tokenizer = get_tokenizer_for_model('unknown-model:7b', backend='ollama')
    """
    # Create cache key
    cache_key = f"{backend}:{model_name}"
    
    # Return cached tokenizer if available and not forcing new
    if not force_new and cache_key in _TOKENIZER_CACHE:
        logger.debug(f"Using cached tokenizer for {cache_key}")
        return _TOKENIZER_CACHE[cache_key]
    
    # Create appropriate tokenizer based on backend
    tokenizer = None
    
    if backend == 'ollama':
        # Try to map Ollama model to HuggingFace tokenizer
        hf_model = OLLAMA_TO_HF_TOKENIZER.get(model_name)
        
        if hf_model:
            logger.info(f"Using HuggingFace tokenizer '{hf_model}' for Ollama model '{model_name}'")
            try:
                tokenizer = HuggingFaceTokenizer(model_name=hf_model)
            except Exception as e:
                logger.warning(
                    f"Failed to load HuggingFace tokenizer for '{hf_model}': {e}. "
                    f"Falling back to OpenAI tokenizer with cl100k_base."
                )
                tokenizer = OpenAITokenizer(model_name=model_name)
        else:
            logger.info(
                f"No HuggingFace mapping found for Ollama model '{model_name}'. "
                f"Using OpenAI tokenizer with cl100k_base fallback."
            )
            tokenizer = OpenAITokenizer(model_name=model_name)
    
    elif backend == 'huggingface':
        logger.info(f"Using HuggingFace tokenizer for model '{model_name}'")
        tokenizer = HuggingFaceTokenizer(model_name=model_name)
    
    else:  # Default to OpenAI (backend == 'openai' or other)
        logger.info(f"Using OpenAI tokenizer for model '{model_name}'")
        tokenizer = OpenAITokenizer(model_name=model_name)
    
    # Cache the tokenizer
    _TOKENIZER_CACHE[cache_key] = tokenizer
    logger.debug(f"Cached tokenizer for {cache_key}")
    
    return tokenizer


def clear_tokenizer_cache():
    """
    Clear the tokenizer cache. Useful for testing or when memory needs to be freed.
    """
    global _TOKENIZER_CACHE
    _TOKENIZER_CACHE.clear()
    logger.info("Tokenizer cache cleared")


def get_cache_info():
    """
    Get information about cached tokenizers.
    
    Returns:
        dict: Dictionary with cache statistics
    """
    return {
        'cached_models': list(_TOKENIZER_CACHE.keys()),
        'cache_size': len(_TOKENIZER_CACHE)
    }
