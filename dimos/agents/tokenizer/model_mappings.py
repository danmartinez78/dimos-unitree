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
Mappings between Ollama model names and their corresponding HuggingFace tokenizers.
This allows accurate token counting for local models.
"""

# Mapping of Ollama model names to HuggingFace model identifiers
# These HuggingFace models have compatible tokenizers for the corresponding Ollama models
OLLAMA_TO_HF_TOKENIZER = {
    # Phi models
    'phi4:14b': 'microsoft/phi-4',
    'phi3:14b': 'microsoft/phi-3-medium',
    'phi3': 'microsoft/phi-3-medium',
    'phi4': 'microsoft/phi-4',
    
    # Qwen models
    'qwen2.5-coder:32b': 'Qwen/Qwen2.5-Coder-32B',
    'qwen2.5-coder:7b': 'Qwen/Qwen2.5-Coder-7B',
    'qwen2.5-coder': 'Qwen/Qwen2.5-Coder-7B',
    'qwen2.5:72b': 'Qwen/Qwen2.5-72B',
    'qwen2.5:32b': 'Qwen/Qwen2.5-32B',
    'qwen2.5:14b': 'Qwen/Qwen2.5-14B',
    'qwen2.5:7b': 'Qwen/Qwen2.5-7B',
    'qwen2.5': 'Qwen/Qwen2.5-7B',
    
    # Llama models
    'llama3.1:70b': 'meta-llama/Llama-3.1-70B',
    'llama3.1:13b': 'meta-llama/Llama-3.1-13B',
    'llama3.1:8b': 'meta-llama/Llama-3.1-8B',
    'llama3.1': 'meta-llama/Llama-3.1-8B',
    'llama3.2:3b': 'meta-llama/Llama-3.2-3B',
    'llama3.2:1b': 'meta-llama/Llama-3.2-1B',
    'llama3.2': 'meta-llama/Llama-3.2-3B',
    
    # Mistral models
    'mistral:7b': 'mistralai/Mistral-7B-v0.1',
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'mixtral:8x7b': 'mistralai/Mixtral-8x7B-v0.1',
    'mixtral': 'mistralai/Mixtral-8x7B-v0.1',
    
    # Gemma models
    'gemma2:27b': 'google/gemma-2-27b',
    'gemma2:9b': 'google/gemma-2-9b',
    'gemma2:2b': 'google/gemma-2-2b',
    'gemma2': 'google/gemma-2-9b',
    'gemma:7b': 'google/gemma-7b',
    'gemma:2b': 'google/gemma-2b',
    'gemma': 'google/gemma-7b',
    
    # Deepseek models
    'deepseek-coder:33b': 'deepseek-ai/deepseek-coder-33b-base',
    'deepseek-coder:6.7b': 'deepseek-ai/deepseek-coder-6.7b-base',
    'deepseek-coder': 'deepseek-ai/deepseek-coder-6.7b-base',
}
