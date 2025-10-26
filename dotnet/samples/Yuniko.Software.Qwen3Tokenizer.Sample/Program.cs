using Yuniko.Software.Qwen3Tokenizer;

Console.WriteLine("=== Qwen3 Tokenizer Sample ===\n");

// ============================================================================
// SECTION 1: Loading the Tokenizer
// ============================================================================
Console.WriteLine("╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║ SECTION 1: Loading the Tokenizer                             ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

// Method 1: Load from HuggingFace (downloads and caches files automatically)
Console.WriteLine("→ Method 1: FromHuggingFace (sync)");
var tokenizer = Qwen3Tokenizer.FromHuggingFace(
    modelName: "Qwen/Qwen3-Embedding-0.6B"
);
Console.WriteLine($"  ✓ Tokenizer loaded! Vocabulary size: {tokenizer.VocabularySize:N0}\n");

// Method 2: Load from HuggingFace with progress reporting (async)
Console.WriteLine("→ Method 2: FromHuggingFaceAsync with progress");
var progress = new Progress<DownloadProgress>(p =>
{
    if (p.PercentComplete.HasValue)
    {
        Console.WriteLine($"  [{p.FileName}] {p.PercentComplete:F1}% - {p.Status}");
    }
    else
    {
        Console.WriteLine($"  [{p.FileName}] {p.Status}");
    }
});

var tokenizerAsync = await Qwen3Tokenizer.FromHuggingFaceAsync(
    modelName: "Qwen/Qwen3-Embedding-0.6B",
    progress: progress
);
Console.WriteLine($"  ✓ Async tokenizer loaded!\n");

// Method 3: Load from local files (if you already have the files)
Console.WriteLine("→ Method 3: FromFiles (direct path loading)");
Console.WriteLine("  Example: Qwen3Tokenizer.FromFiles(\"path/to/vocab.json\", \"path/to/merges.txt\")");
Console.WriteLine("  (Skipped - requires local files)\n");

// Method 4: Load with custom configuration
Console.WriteLine("→ Method 4: FromHuggingFace with custom config");
var config = new HuggingFaceConfig(
    BaseUrl: "https://huggingface.co",
    Branch: "main",
    VocabFileName: "vocab.json",
    MergesFileName: "merges.txt"
);
var provider = new HuggingFaceFileProvider("Qwen/Qwen3-Embedding-0.6B", config: config);
var tokenizerCustom = Qwen3Tokenizer.FromProvider(provider);
Console.WriteLine($"  ✓ Custom tokenizer loaded!\n");

Console.WriteLine("Press any key to continue...\n");
Console.ReadKey();

// ============================================================================
// SECTION 2: Basic Tokenization
// ============================================================================
Console.WriteLine("\n╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║ SECTION 2: Basic Tokenization                                ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

string sampleText = "Hello, this is a test of the Qwen3 tokenizer!";

// Encode text to token IDs
Console.WriteLine("→ Encode (text → token IDs)");
Console.WriteLine($"  Input: \"{sampleText}\"");
int[] tokenIds = tokenizer.Encode(sampleText);
Console.WriteLine($"  Output: [{string.Join(", ", tokenIds)}]");
Console.WriteLine($"  Token count: {tokenIds.Length}\n");

// Decode token IDs back to text
Console.WriteLine("→ Decode (token IDs → text)");
Console.WriteLine($"  Input: [{string.Join(", ", tokenIds)}]");
string decodedText = tokenizer.Decode(tokenIds);
Console.WriteLine($"  Output: \"{decodedText}\"\n");

// Count tokens without full encoding (more efficient)
Console.WriteLine("→ CountTokens (efficient token counting)");
string longText = "The Qwen3 tokenizer is designed for Qwen3-Embedding models.";
int count = tokenizer.CountTokens(longText);
Console.WriteLine($"  Input: \"{longText}\"");
Console.WriteLine($"  Token count: {count}\n");

Console.WriteLine("Press any key to continue...\n");
Console.ReadKey();

// ============================================================================
// SECTION 3: Advanced Tokenization
// ============================================================================
Console.WriteLine("\n╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║ SECTION 3: Advanced Tokenization                             ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

// Detailed encoding with token information
Console.WriteLine("→ EncodeDetailed (get tokens, IDs, and offsets)");
string detailText = "Tokenization is fun!";
var detailed = tokenizer.EncodeDetailed(detailText);
Console.WriteLine($"  Input: \"{detailText}\"\n");
Console.WriteLine("  Token Details:");
Console.WriteLine("  ┌─────┬────────┬──────────────────────┬──────────────┐");
Console.WriteLine("  │ Idx │ ID     │ Token                │ Offset       │");
Console.WriteLine("  ├─────┼────────┼──────────────────────┼──────────────┤");
for (int i = 0; i < detailed.Tokens.Length; i++)
{
    Console.WriteLine($"  │ {i,3} │ {detailed.Ids[i],6} │ {detailed.Tokens[i],-20} │ ({detailed.Offsets[i].Index,3}, {detailed.Offsets[i].Length,2})     │");
}
Console.WriteLine("  └─────┴────────┴──────────────────────┴──────────────┘\n");

// Batch encoding
Console.WriteLine("→ EncodeBatch (encode multiple texts at once)");
string[] batchTexts = ["First sentence.", "Second sentence.", "Third sentence."];
int[][] batchIds = tokenizer.EncodeBatch(batchTexts);
Console.WriteLine("  Inputs:");
for (int i = 0; i < batchTexts.Length; i++)
{
    Console.WriteLine($"    {i + 1}. \"{batchTexts[i]}\"");
}
Console.WriteLine("\n  Outputs:");
for (int i = 0; i < batchIds.Length; i++)
{
    Console.WriteLine($"    {i + 1}. [{string.Join(", ", batchIds[i])}]");
}
Console.WriteLine();

// Batch decoding
Console.WriteLine("→ DecodeBatch (decode multiple token sequences at once)");
string[] decodedBatch = tokenizer.DecodeBatch(batchIds);
Console.WriteLine("  Decoded:");
for (int i = 0; i < decodedBatch.Length; i++)
{
    Console.WriteLine($"    {i + 1}. \"{decodedBatch[i]}\"");
}
Console.WriteLine();

Console.WriteLine("Press any key to continue...\n");
Console.ReadKey();

// ============================================================================
// SECTION 4: Working with Special Tokens
// ============================================================================
Console.WriteLine("\n╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║ SECTION 4: Working with Special Tokens                       ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

// List all special tokens
Console.WriteLine("→ Available Special Tokens");
Console.WriteLine("  ┌───────────────────────────┬────────┐");
Console.WriteLine("  │ Token                     │ ID     │");
Console.WriteLine("  ├───────────────────────────┼────────┤");
foreach (var (token, id) in tokenizer.SpecialTokens)
{
    Console.WriteLine($"  │ {token,-25} │ {id,6} │");
}
Console.WriteLine("  └───────────────────────────┴────────┘\n");

// Encode with special tokens
Console.WriteLine("→ Encoding with Special Tokens");
string textWithSpecial = "<|im_start|>user\nHello!<|im_end|>";
Console.WriteLine($"  Input: \"{textWithSpecial}\"");
var idsWithSpecial = tokenizer.Encode(textWithSpecial, addEos: false);
Console.WriteLine($"  Encoded: [{string.Join(", ", idsWithSpecial)}]\n");

// Decode with and without special tokens
Console.WriteLine("→ Decoding with/without Special Tokens");
string withSpecialTokens = tokenizer.Decode(idsWithSpecial, skipSpecialTokens: false);
string withoutSpecialTokens = tokenizer.Decode(idsWithSpecial, skipSpecialTokens: true);
Console.WriteLine($"  With special tokens:    \"{withSpecialTokens}\"");
Console.WriteLine($"  Without special tokens: \"{withoutSpecialTokens}\"\n");

// Get specific special token IDs
Console.WriteLine("→ Get Specific Special Token ID");
int? eosId = tokenizer.GetSpecialTokenId("<|endoftext|>");
Console.WriteLine($"  GetSpecialTokenId(\"<|endoftext|>\"): {eosId}\n");

Console.WriteLine("Press any key to continue...\n");
Console.ReadKey();

// ============================================================================
// SECTION 5: ONNX Runtime Integration
// ============================================================================
Console.WriteLine("\n╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║ SECTION 5: ONNX Runtime Integration                          ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

// Prepare single text for ONNX
Console.WriteLine("→ PrepareForOnnx (single text)");
string onnxText = "Example text for ONNX inference";
var (inputIds, attentionMask) = tokenizer.PrepareForOnnx(onnxText, maxLength: 20);
Console.WriteLine($"  Input: \"{onnxText}\"");
Console.WriteLine($"  Max Length: 20\n");
Console.WriteLine($"  Input IDs:      [{string.Join(", ", inputIds)}]");
Console.WriteLine($"  Attention Mask: [{string.Join(", ", attentionMask)}]\n");

// Prepare batch for ONNX
Console.WriteLine("→ PrepareForOnnxBatch (batch of texts)");
string[] onnxBatch = ["First text.", "Second text with more tokens."];
var (batchInputIds, batchAttentionMask) = tokenizer.PrepareForOnnxBatch(onnxBatch, maxLength: 15);
Console.WriteLine($"  Batch size: {onnxBatch.Length}");
Console.WriteLine($"  Max Length: 15\n");
for (int i = 0; i < onnxBatch.Length; i++)
{
    Console.WriteLine($"  Text {i + 1}: \"{onnxBatch[i]}\"");
    Console.Write($"    Input IDs:      [");
    for (int j = 0; j < 15; j++)
    {
        Console.Write($"{batchInputIds[i, j]}{(j < 14 ? ", " : "")}");
    }

    Console.WriteLine("]");
    Console.Write($"    Attention Mask: [");
    for (int j = 0; j < 15; j++)
    {
        Console.Write($"{batchAttentionMask[i, j]}{(j < 14 ? ", " : "")}");
    }

    Console.WriteLine("]\n");
}

Console.WriteLine("Press any key to continue...\n");
Console.ReadKey();

// ============================================================================
// SECTION 6: Custom Options and Configuration
// ============================================================================
Console.WriteLine("\n╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║ SECTION 6: Custom Options and Configuration                  ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

// Using default options
Console.WriteLine("→ Default Tokenizer Options");
var defaultOptions = Qwen3TokenizerOptions.Default;
Console.WriteLine($"  EOS Token ID: {defaultOptions.EosTokenId}");
Console.WriteLine($"  Byte Level: {defaultOptions.ByteLevel}");
Console.WriteLine($"  Special Tokens Count: {defaultOptions.SpecialTokens.Count}\n");

// Creating custom options
Console.WriteLine("→ Custom Tokenizer Options (using record 'with' expression)");
var customOptionsExample = Qwen3TokenizerOptions.Default with
{
    ByteLevel = false
};
Console.WriteLine($"  Custom ByteLevel: {customOptionsExample.ByteLevel}");
Console.WriteLine("  Usage: Qwen3Tokenizer.FromHuggingFace(..., options: customOptions)\n");

// Custom HuggingFace configuration
Console.WriteLine("→ Custom HuggingFace Configuration");
var customConfig = new HuggingFaceConfig(
    BaseUrl: "https://huggingface.co",
    Branch: "v1.0.0",  // Use specific version/tag
    VocabFileName: "vocab.json",
    MergesFileName: "merges.txt"
);
Console.WriteLine($"  Base URL: {customConfig.BaseUrl}");
Console.WriteLine($"  Branch: {customConfig.Branch}");
Console.WriteLine($"  Vocab File: {customConfig.VocabFileName}");
Console.WriteLine($"  Merges File: {customConfig.MergesFileName}\n");

// Using custom HttpClient
Console.WriteLine("→ Custom HttpClient (for authentication, proxies, etc.)");
Console.WriteLine("  Example:");
Console.WriteLine("    using var httpClient = new HttpClient();");
Console.WriteLine("    httpClient.DefaultRequestHeaders.Add(\"Authorization\", \"Bearer YOUR_TOKEN\");");
Console.WriteLine("    var tokenizer = Qwen3Tokenizer.FromHuggingFace(");
Console.WriteLine("        modelName: \"...\",");
Console.WriteLine("        httpClient: httpClient");
Console.WriteLine("    );\n");

// Special token constants
Console.WriteLine("→ Special Token Constants");
Console.WriteLine($"  EndOfText Token ID: {Qwen3EmbeddingModelSpecialTokens.EndOfTextTokenId}");
Console.WriteLine($"  ImStart Token ID: {Qwen3EmbeddingModelSpecialTokens.ImStartTokenId}");
Console.WriteLine($"  ImEnd Token ID: {Qwen3EmbeddingModelSpecialTokens.ImEndTokenId}");
Console.WriteLine($"  EndOfText String: \"{Qwen3EmbeddingModelSpecialTokens.EndOfText}\"");
Console.WriteLine($"  ImStart String: \"{Qwen3EmbeddingModelSpecialTokens.ImStart}\"\n");

// ============================================================================
// SECTION 7: Summary of API Methods
// ============================================================================
Console.WriteLine("\n╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║ SECTION 7: Summary of API Methods                            ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

Console.WriteLine("📦 LOADING:");
Console.WriteLine("  • FromFiles(vocabPath, mergesPath, options?)");
Console.WriteLine("  • FromHuggingFace(modelName, cacheDir?, options?, httpClient?, progress?)");
Console.WriteLine("  • FromHuggingFaceAsync(modelName, cancellationToken?, ...)");
Console.WriteLine("  • FromProvider(fileProvider, options?, progress?)");
Console.WriteLine("  • FromProviderAsync(fileProvider, cancellationToken?, ...)\n");

Console.WriteLine("🔤 ENCODING:");
Console.WriteLine("  • Encode(text, addEos?) → int[]");
Console.WriteLine("  • EncodeDetailed(text, addEos?) → EncodingResult");
Console.WriteLine("  • EncodeBatch(texts[], addEos?) → int[][]");
Console.WriteLine("  • CountTokens(text, addEos?) → int\n");

Console.WriteLine("🔠 DECODING:");
Console.WriteLine("  • Decode(ids[], skipSpecialTokens?) → string");
Console.WriteLine("  • DecodeBatch(ids[][], skipSpecialTokens?) → string[]\n");

Console.WriteLine("⚙️  PROPERTIES:");
Console.WriteLine("  • VocabularySize → int");
Console.WriteLine("  • Vocabulary → IReadOnlyDictionary<string, int>");
Console.WriteLine("  • SpecialTokens → IReadOnlyDictionary<string, int>");
Console.WriteLine("  • GetSpecialTokenId(tokenName) → int?\n");

Console.WriteLine("🤖 ONNX:");
Console.WriteLine("  • PrepareForOnnx(text, maxLength?) → (long[], long[])");
Console.WriteLine("  • PrepareForOnnxBatch(texts[], maxLength?) → (long[,], long[,])\n");

Console.WriteLine("\n=== Sample Complete ===");
Console.WriteLine("All examples demonstrate the Qwen3 Tokenizer capabilities.");
Console.WriteLine("For more information, visit: https://github.com/yuniko-software/qwen3-tokenizer...\n");
