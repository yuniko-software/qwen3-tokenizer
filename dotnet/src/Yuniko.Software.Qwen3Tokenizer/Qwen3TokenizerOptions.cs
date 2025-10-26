using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Configuration options for Qwen3 tokenizer.
/// </summary>
/// <param name="SpecialTokens">The special tokens dictionary.</param>
/// <param name="PreTokenizerRegex">The pre-tokenizer regex pattern.</param>
/// <param name="EosTokenId">The end-of-sequence token ID. This is also used as the padding token ID.</param>
/// <param name="Normalizer">The normalizer. Defaults to NFC normalization.</param>
/// <param name="ByteLevel">Whether to use byte-level BPE encoding/decoding. Defaults to true.</param>
public sealed record Qwen3TokenizerOptions(
    IReadOnlyDictionary<string, int> SpecialTokens,
    Regex PreTokenizerRegex,
    int EosTokenId,
    Normalizer? Normalizer = null,
    bool ByteLevel = true)
{
    /// <summary>
    /// Default options for Qwen3 Embedding models.
    /// Matches the HuggingFace tokenizer configuration.
    /// </summary>
    public static readonly Qwen3TokenizerOptions Default = new(
        SpecialTokens: Qwen3EmbeddingModelSpecialTokens.SpecialTokens,
        PreTokenizerRegex: Qwen3EmbeddingModelSpecialTokens.GetPreTokenizerRegex(),
        EosTokenId: Qwen3EmbeddingModelSpecialTokens.EndOfTextTokenId,
        Normalizer: new NfcNormalizer(),
        ByteLevel: true);
}