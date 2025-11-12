"""
Generate test data for Qwen3 Tokenizer comparison tests.

This script uses the official HuggingFace transformers library to generate
expected tokenization outputs that will be used to validate the C# implementation.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer


def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate comprehensive test cases covering various scenarios."""

    test_cases = [
        {"name": "basic_english", "input": "Hello, this is a test of the Qwen3 tokenizer!", "add_special_tokens": True},
        {"name": "basic_english_no_special", "input": "Hello, this is a test of the Qwen3 tokenizer!", "add_special_tokens": False},
        {"name": "simple_sentence", "input": "The quick brown fox jumps over the lazy dog.", "add_special_tokens": True},
        {"name": "simple_sentence_no_special", "input": "The quick brown fox jumps over the lazy dog.", "add_special_tokens": False},
        {"name": "empty_string", "input": "", "add_special_tokens": True},
        {"name": "empty_string_no_special", "input": "", "add_special_tokens": False},
        {"name": "single_space", "input": " ", "add_special_tokens": True},
        {"name": "single_space_no_special", "input": " ", "add_special_tokens": False},
        {"name": "multiple_spaces", "input": "   ", "add_special_tokens": True},
        {"name": "numbers", "input": "12345 67890", "add_special_tokens": True},
        {"name": "numbers_no_special", "input": "12345 67890", "add_special_tokens": False},
        {"name": "mixed_numbers_text", "input": "I have 42 apples and 100 oranges.", "add_special_tokens": True},
        {"name": "punctuation", "input": "Hello! How are you? I'm fine, thanks.", "add_special_tokens": True},
        {"name": "punctuation_no_special", "input": "Hello! How are you? I'm fine, thanks.", "add_special_tokens": False},
        {"name": "special_chars", "input": "@#$%^&*()_+-=[]{}|;:',.<>?/~`", "add_special_tokens": True},
        {"name": "emojis", "input": "Hello 👋 World 🌍! Happy 😊", "add_special_tokens": True},
        {"name": "emojis_no_special", "input": "Hello 👋 World 🌍! Happy 😊", "add_special_tokens": False},
        {"name": "mixed_unicode", "input": "Café résumé naïve", "add_special_tokens": True},
        {"name": "chinese_simple", "input": "你好世界", "add_special_tokens": True},
        {"name": "chinese_simple_no_special", "input": "你好世界", "add_special_tokens": False},
        {"name": "chinese_sentence", "input": "这是一个测试句子。", "add_special_tokens": True},
        {"name": "mixed_english_chinese", "input": "Hello 你好 World 世界", "add_special_tokens": True},
        {"name": "mixed_english_chinese_no_special", "input": "Hello 你好 World 世界", "add_special_tokens": False},
        {"name": "japanese_hiragana", "input": "こんにちは", "add_special_tokens": True},
        {"name": "japanese_mixed", "input": "こんにちは世界 Hello World", "add_special_tokens": True},
        {"name": "special_token_im_start", "input": "<|im_start|>user\nHello!<|im_end|>", "add_special_tokens": False},
        {"name": "special_token_im_start_with_special", "input": "<|im_start|>user\nHello!<|im_end|>", "add_special_tokens": True},
        {"name": "special_token_endoftext", "input": "Text before<|endoftext|>Text after", "add_special_tokens": False},
        {"name": "multiple_special_tokens", "input": "<|im_start|>system\nYou are helpful<|im_end|><|im_start|>user\nHi<|im_end|>", "add_special_tokens": False},
        {"name": "multiple_special_tokens_with_special", "input": "<|im_start|>system\nYou are helpful<|im_end|><|im_start|>user\nHi<|im_end|>", "add_special_tokens": True},
        {"name": "long_text", "input": "The Qwen3 tokenizer is designed for Qwen3-Embedding models. " * 10, "add_special_tokens": True},
        {"name": "long_text_no_special", "input": "The Qwen3 tokenizer is designed for Qwen3-Embedding models. " * 10, "add_special_tokens": False},
        {"name": "newlines", "input": "Line 1\nLine 2\nLine 3", "add_special_tokens": True},
        {"name": "tabs", "input": "Column1\tColumn2\tColumn3", "add_special_tokens": True},
        {"name": "mixed_whitespace", "input": "Text\n\twith\r\nmixed\t\nwhitespace", "add_special_tokens": True},
        {"name": "code_python", "input": "def hello():\n    print('Hello, World!')", "add_special_tokens": True},
        {"name": "code_python_no_special", "input": "def hello():\n    print('Hello, World!')", "add_special_tokens": False},
        {"name": "code_json", "input": '{"name": "test", "value": 123}', "add_special_tokens": True},
        {"name": "uppercase", "input": "HELLO WORLD", "add_special_tokens": True},
        {"name": "lowercase", "input": "hello world", "add_special_tokens": True},
        {"name": "mixed_case", "input": "HeLLo WoRLd", "add_special_tokens": True},
        {"name": "single_char_a", "input": "a", "add_special_tokens": True},
        {"name": "single_char_a_no_special", "input": "a", "add_special_tokens": False},
        {"name": "single_char_space", "input": " ", "add_special_tokens": True},
        {"name": "repeated_chars", "input": "aaaaaa bbbbbb cccccc", "add_special_tokens": True},
        {"name": "url", "input": "Visit https://example.com for more info", "add_special_tokens": True},
        {"name": "email", "input": "Contact us at test@example.com", "add_special_tokens": True},
        {"name": "all_special_tokens", "input": "<|endoftext|><|im_start|><|im_end|><|object_ref_start|><|object_ref_end|><|box_start|><|box_end|><|quad_start|><|quad_end|><|vision_start|><|vision_end|><|vision_pad|><|image_pad|><|video_pad|><tool_call></tool_call><|fim_prefix|><|fim_middle|><|fim_suffix|><|fim_pad|><|repo_name|><|file_sep|><tool_response></tool_response><think></think>", "add_special_tokens": False},
        {"name": "all_special_tokens_with_special", "input": "<|endoftext|><|im_start|><|im_end|><|object_ref_start|><|object_ref_end|><|box_start|><|box_end|><|quad_start|><|quad_end|><|vision_start|><|vision_end|><|vision_pad|><|image_pad|><|video_pad|><tool_call></tool_call><|fim_prefix|><|fim_middle|><|fim_suffix|><|fim_pad|><|repo_name|><|file_sep|><tool_response></tool_response><think></think>", "add_special_tokens": True},
    ]

    return test_cases


def tokenize_with_python(tokenizer, text: str, add_special_tokens: bool) -> Dict[str, Any]:
    """Tokenize text using the Python transformers library and return detailed results."""
    encoding = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)

    ids = encoding['input_ids']

    tokens = [tokenizer.decode([token_id]) for token_id in ids]

    offsets = encoding['offset_mapping']

    offsets_formatted = [(offset[0], offset[1] - offset[0]) for offset in offsets]

    decoded_text = tokenizer.decode(ids, skip_special_tokens=False)
    decoded_text_skip_special = tokenizer.decode(ids, skip_special_tokens=True)

    return {
        "ids": ids,
        "tokens": tokens,
        "offsets": offsets_formatted,
        "decoded": decoded_text,
        "decoded_skip_special_tokens": decoded_text_skip_special,
        "token_count": len(ids),
    }


def generate_for_model(model_name: str, output_file: str):
    """Generate test data for a specific model."""
    print(f"Loading tokenizer from HuggingFace: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    print()

    test_cases = generate_test_cases()
    print(f"Generating {len(test_cases)} test cases...")

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] Processing: {test_case['name']}")

        result = tokenize_with_python(
            tokenizer,
            test_case['input'],
            test_case['add_special_tokens'],
        )

        results.append({
            "name": test_case['name'],
            "input": test_case['input'],
            "add_special_tokens": test_case['add_special_tokens'],
            "expected": result,
        })

    output_data = {
        "model_name": model_name,
        "vocabulary_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "test_cases": results,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to inform user about overwriting
    file_exists = output_path.exists()
    if file_exists:
        print(f"  [INFO] Overwriting existing file: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    action = "overwritten" if file_exists else "created"
    print(f"[OK] Test data {action}: {output_file}")
    print(f"  Total test cases: {len(results)}")
    print()


def main():
    """Main function to generate test data for one or more models."""
    models = [
        ("Qwen/Qwen3-0.6B", "../tests/Yuniko.Software.Qwen3Tokenizer.Tests/TestData/test_data_qwen3_0.6b.json"),
        ("Qwen/Qwen3-Embedding-0.6B", "../tests/Yuniko.Software.Qwen3Tokenizer.Tests/TestData/test_data_qwen3_embedding_0.6b.json"),
        ("Qwen/Qwen3-Next-80B-A3B-Instruct", "../tests/Yuniko.Software.Qwen3Tokenizer.Tests/TestData/test_data_qwen3_next_80b_a3b_instruct.json"),
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", "../tests/Yuniko.Software.Qwen3Tokenizer.Tests/TestData/test_data_qwen3_vl_30b_a3b_instruct.json"),
        ("Qwen/Qwen3-Reranker-0.6B", "../tests/Yuniko.Software.Qwen3Tokenizer.Tests/TestData/test_data_qwen3_reranker_0.6b.json"),
    ]

    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        output_file = "../tests/Yuniko.Software.Qwen3Tokenizer.Tests/test_data.json"
        generate_for_model(model_arg, output_file)
    else:
        for model_name, output_file in models:
            generate_for_model(model_name, output_file)

    print("=" * 60)
    print("All test data generated successfully!")
    print("You can now run the C# tests to compare outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
