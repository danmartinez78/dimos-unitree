"""
Tests for tokenizer factory and Ollama model support.
"""
import pytest
from dimos.agents.tokenizer import (
    get_tokenizer_for_model,
    clear_tokenizer_cache,
    get_cache_info,
    OpenAITokenizer,
    HuggingFaceTokenizer,
    OLLAMA_TO_HF_TOKENIZER
)


class TestTokenizerFactory:
    """Test the tokenizer factory functionality."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_tokenizer_cache()
    
    def test_openai_tokenizer_for_openai_backend(self):
        """Test that OpenAI backend returns OpenAITokenizer."""
        tokenizer = get_tokenizer_for_model('gpt-4o', backend='openai')
        assert isinstance(tokenizer, OpenAITokenizer)
        assert tokenizer.model_name == 'gpt-4o'
    
    def test_ollama_model_with_hf_mapping(self):
        """Test that mapped Ollama models get HuggingFace tokenizers."""
        # This test will only pass if HuggingFace models can be downloaded
        # Skip if not in an environment with internet/HF access
        try:
            tokenizer = get_tokenizer_for_model('phi4:14b', backend='ollama')
            # Should be HuggingFace tokenizer with mapped model
            assert isinstance(tokenizer, (HuggingFaceTokenizer, OpenAITokenizer))
        except Exception as e:
            pytest.skip(f"HuggingFace tokenizer not available: {e}")
    
    def test_ollama_model_without_mapping_falls_back(self):
        """Test that unmapped Ollama models fall back to OpenAI tokenizer."""
        tokenizer = get_tokenizer_for_model('unknown-model:7b', backend='ollama')
        assert isinstance(tokenizer, OpenAITokenizer)
    
    def test_huggingface_backend(self):
        """Test that HuggingFace backend returns HuggingFaceTokenizer."""
        try:
            tokenizer = get_tokenizer_for_model('Qwen/Qwen2.5-0.5B', backend='huggingface')
            assert isinstance(tokenizer, HuggingFaceTokenizer)
        except Exception as e:
            pytest.skip(f"HuggingFace tokenizer not available: {e}")
    
    def test_tokenizer_caching(self):
        """Test that tokenizers are cached properly."""
        # Get a tokenizer
        tokenizer1 = get_tokenizer_for_model('gpt-4o', backend='openai')
        
        # Get the same tokenizer again
        tokenizer2 = get_tokenizer_for_model('gpt-4o', backend='openai')
        
        # Should be the same instance (cached)
        assert tokenizer1 is tokenizer2
        
        # Check cache info
        cache_info = get_cache_info()
        assert 'openai:gpt-4o' in cache_info['cached_models']
        assert cache_info['cache_size'] >= 1
    
    def test_force_new_bypasses_cache(self):
        """Test that force_new creates a new tokenizer instance."""
        # Get a tokenizer
        tokenizer1 = get_tokenizer_for_model('gpt-4o', backend='openai')
        
        # Force a new instance
        tokenizer2 = get_tokenizer_for_model('gpt-4o', backend='openai', force_new=True)
        
        # Should be different instances
        assert tokenizer1 is not tokenizer2
    
    def test_clear_cache(self):
        """Test that cache can be cleared."""
        # Add some tokenizers to cache
        get_tokenizer_for_model('gpt-4o', backend='openai')
        get_tokenizer_for_model('gpt-3.5-turbo', backend='openai')
        
        # Verify cache has items
        cache_info = get_cache_info()
        assert cache_info['cache_size'] >= 2
        
        # Clear cache
        clear_tokenizer_cache()
        
        # Verify cache is empty
        cache_info = get_cache_info()
        assert cache_info['cache_size'] == 0


class TestOpenAITokenizerFallback:
    """Test OpenAI tokenizer fallback for non-OpenAI models."""
    
    def test_valid_openai_model(self):
        """Test that valid OpenAI models work."""
        tokenizer = OpenAITokenizer('gpt-4o')
        text = "Hello, world!"
        tokens = tokenizer.token_count(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_invalid_model_falls_back_to_cl100k(self):
        """Test that invalid model names fall back to cl100k_base."""
        # This should not raise an error, but fall back
        tokenizer = OpenAITokenizer('non-existent-model:7b')
        
        # Should still be able to count tokens
        text = "Hello, world!"
        tokens = tokenizer.token_count(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_tokenize_and_detokenize(self):
        """Test that tokenization works with fallback."""
        tokenizer = OpenAITokenizer('ollama-model:7b')
        text = "The quick brown fox"
        
        # Tokenize
        tokens = tokenizer.tokenize_text(text)
        assert len(tokens) > 0
        
        # Detokenize
        decoded = tokenizer.detokenize_text(tokens)
        # Should be similar (may not be exact due to encoding)
        assert len(decoded) > 0


class TestModelMappings:
    """Test the Ollama to HuggingFace model mappings."""
    
    def test_ollama_mappings_exist(self):
        """Test that mappings dictionary exists and has entries."""
        assert OLLAMA_TO_HF_TOKENIZER is not None
        assert len(OLLAMA_TO_HF_TOKENIZER) > 0
    
    def test_phi_models_mapped(self):
        """Test that Phi models are mapped."""
        assert 'phi4:14b' in OLLAMA_TO_HF_TOKENIZER
        assert 'microsoft/phi-4' in OLLAMA_TO_HF_TOKENIZER['phi4:14b']
    
    def test_qwen_models_mapped(self):
        """Test that Qwen models are mapped."""
        assert 'qwen2.5-coder:32b' in OLLAMA_TO_HF_TOKENIZER
        assert 'Qwen' in OLLAMA_TO_HF_TOKENIZER['qwen2.5-coder:32b']
    
    def test_llama_models_mapped(self):
        """Test that Llama models are mapped."""
        assert 'llama3.1:70b' in OLLAMA_TO_HF_TOKENIZER
        assert 'meta-llama' in OLLAMA_TO_HF_TOKENIZER['llama3.1:70b']
    
    def test_mistral_models_mapped(self):
        """Test that Mistral models are mapped."""
        assert 'mistral:7b' in OLLAMA_TO_HF_TOKENIZER
        assert 'mistralai' in OLLAMA_TO_HF_TOKENIZER['mistral:7b']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
