using Yuniko.Software.Qwen3Tokenizer;

Console.WriteLine("=== Qwen3 Tokenizer Demo ===\n");

// Load tokenizer from pretrained model (downloads vocab and merges files if needed)
Console.WriteLine("Loading Qwen3 tokenizer...");
var tokenizer = await Qwen3Tokenizer.FromPretrainedAsync();
Console.WriteLine($"Tokenizer loaded! Vocabulary size: {tokenizer.VocabularySize:N0}\n");

// Example 1: Basic encoding
Console.WriteLine("--- Example 1: Basic Encoding ---");
string text1 = "Hello, this is a test of the Qwen3 tokenizer!";
Console.WriteLine($"Input text: \"{text1}\"");

int[] ids = tokenizer.Encode(text1);
Console.WriteLine($"Token IDs: [{string.Join(", ", ids)}]");
Console.WriteLine($"Number of tokens: {ids.Length}\n");

// Example 2: Decoding
Console.WriteLine("--- Example 2: Decoding ---");
string decoded = tokenizer.Decode(ids);
Console.WriteLine($"Decoded text: \"{decoded}\"\n");

// Example 3: Token counting (more efficient than full encoding)
Console.WriteLine("--- Example 3: Token Counting ---");
string longText = "The Qwen3 tokenizer is designed for the Qwen3-Embedding models. It uses byte-level BPE encoding similar to GPT-2.";
int tokenCount = tokenizer.CountTokens(longText);
Console.WriteLine($"Text: \"{longText}\"");
Console.WriteLine($"Token count: {tokenCount}\n");

// Example 4: Detailed encoding with offsets
Console.WriteLine("--- Example 4: Detailed Encoding ---");
string text2 = "Tokenization is fun!";
var detailed = tokenizer.EncodeDetailed(text2);
Console.WriteLine($"Input: \"{text2}\"");
Console.WriteLine("Token details:");
for (int i = 0; i < detailed.Tokens.Length; i++)
{
    Console.WriteLine($"  Token #{i}: ID={detailed.Ids[i],-6} Text=\"{detailed.Tokens[i]}\" Offset=({detailed.Offsets[i].Index}, {detailed.Offsets[i].Length})");
}
Console.WriteLine();

// Example 5: Batch encoding
Console.WriteLine("--- Example 5: Batch Encoding ---");
string[] texts =
[
    "First sentence.",
    "Second sentence.",
    "Third sentence."
];
int[][] batchIds = tokenizer.EncodeBatch(texts);
Console.WriteLine("Batch encoding:");
for (int i = 0; i < texts.Length; i++)
{
    Console.WriteLine($"  \"{texts[i]}\" -> [{string.Join(", ", batchIds[i])}]");
}
Console.WriteLine();

// Example 6: Special tokens
Console.WriteLine("--- Example 6: Special Tokens ---");
Console.WriteLine("Available special tokens:");
foreach (var (token, id) in tokenizer.SpecialTokens)
{
    Console.WriteLine($"  {token,-25} -> {id}");
}
Console.WriteLine();

// Example 7: Working with special tokens
Console.WriteLine("--- Example 7: Using Special Tokens ---");
string textWithSpecial = "<|im_start|>user\nHello!<|im_end|>";
Console.WriteLine($"Input: \"{textWithSpecial}\"");
var idsWithSpecial = tokenizer.Encode(textWithSpecial, addEos: false);
Console.WriteLine($"Encoded: [{string.Join(", ", idsWithSpecial)}]");
string decodedWithSpecial = tokenizer.Decode(idsWithSpecial, skipSpecialTokens: false);
string decodedWithoutSpecial = tokenizer.Decode(idsWithSpecial, skipSpecialTokens: true);
Console.WriteLine($"Decoded (with special tokens): \"{decodedWithSpecial}\"");
Console.WriteLine($"Decoded (without special tokens): \"{decodedWithoutSpecial}\"\n");

// Example 8: Preparing for ONNX inference
Console.WriteLine("--- Example 8: ONNX Preparation ---");
string onnxText = "Example text for ONNX inference";
var (inputIds, attentionMask) = tokenizer.PrepareForOnnx(onnxText, maxLength: 20);
Console.WriteLine($"Input text: \"{onnxText}\"");
Console.WriteLine($"Input IDs (padded to 20): [{string.Join(", ", inputIds)}]");
Console.WriteLine($"Attention mask: [{string.Join(", ", attentionMask)}]\n");

Console.WriteLine("=== Demo Complete ===");