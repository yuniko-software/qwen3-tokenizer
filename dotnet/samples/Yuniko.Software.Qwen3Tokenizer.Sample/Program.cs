using Yuniko.Software.Qwen3Tokenizer;

Console.WriteLine("=== Qwen3 Tokenizer Sample ===\n");

// ============================================================================
// 1. Load Tokenizer
// ============================================================================
Console.WriteLine("1. Loading Tokenizer");
Console.WriteLine("-------------------");

var tokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-0.6B");
Console.WriteLine($"✓ Loaded! Vocabulary size: {tokenizer.VocabularySize:N0}\n");

// ============================================================================
// 2. Basic Encoding & Decoding
// ============================================================================
Console.WriteLine("2. Basic Encoding & Decoding");
Console.WriteLine("-----------------------------");

string text = "Hello, this is a test of the Qwen3 tokenizer!";
Console.WriteLine($"Text: \"{text}\"");

// Encode
int[] tokenIds = tokenizer.Encode(text);
Console.WriteLine($"Token IDs: [{string.Join(", ", tokenIds)}]");
Console.WriteLine($"Token count: {tokenIds.Length}");

// Decode
string decoded = tokenizer.Decode(tokenIds);
Console.WriteLine($"Decoded: \"{decoded}\"\n");

// ============================================================================
// 3. ONNX Integration
// ============================================================================
Console.WriteLine("3. ONNX Integration");
Console.WriteLine("-------------------");

var onnxInputs = tokenizer.PrepareForOnnx("Hello, how are you?", maxLength: 512);
Console.WriteLine($"Input IDs length: {onnxInputs.InputIds.Length}");
Console.WriteLine($"Attention mask length: {onnxInputs.AttentionMask.Length}");
Console.WriteLine($"Position IDs length: {onnxInputs.PositionIds.Length}");
Console.WriteLine($"Sequence length: {onnxInputs.SequenceLength}");
Console.WriteLine("Note: Some models (e.g., embedding models) may not require position_ids\n");

// ============================================================================
// 4. Working with Special Tokens
// ============================================================================
Console.WriteLine("4. Working with Special Tokens");
Console.WriteLine("-------------------------------");

string chatText = "<|im_start|>user Hello!<|im_end|>";
int[] chatIds = tokenizer.Encode(chatText);

string withSpecial = tokenizer.Decode(chatIds, skipSpecialTokens: false);
string withoutSpecial = tokenizer.Decode(chatIds, skipSpecialTokens: true);

Console.WriteLine($"Original: \"{chatText}\"");
Console.WriteLine($"With special tokens: \"{withSpecial}\"");
Console.WriteLine($"Without special tokens: \"{withoutSpecial}\"\n");
