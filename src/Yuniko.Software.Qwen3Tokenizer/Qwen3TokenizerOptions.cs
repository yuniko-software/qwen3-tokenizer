using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Configuration options for Qwen3 tokenizer.
/// </summary>
/// <param name="AddedTokens">All added tokens. These are treated as atomic during pre-tokenization.</param>
/// <param name="SpecialTokenIds">Token IDs marked as "special": true. These are skipped when decoding with skipSpecialTokens=true.</param>
/// <param name="PreTokenizerRegex">The pre-tokenizer regex pattern.</param>
/// <param name="EosTokenId">The end-of-sequence token ID.</param>
/// <param name="PadTokenId">The padding token ID.</param>
/// <param name="Normalizer">The normalizer. Defaults to NFC normalization.</param>
/// <param name="ByteLevel">Whether to use byte-level BPE encoding/decoding. Defaults to true.</param>
public sealed record Qwen3TokenizerOptions(
    IReadOnlyDictionary<string, int> AddedTokens,
    IReadOnlySet<int> SpecialTokenIds,
    Regex PreTokenizerRegex,
    int EosTokenId,
    int PadTokenId,
    Normalizer? Normalizer = null,
    bool ByteLevel = true)
{
    /// <summary>
    /// Default options for Qwen3 models.
    /// Matches the HuggingFace tokenizer configuration.
    /// </summary>
    public static readonly Qwen3TokenizerOptions Default = new(
        AddedTokens: Qwen3Tokens.AddedTokens,
        SpecialTokenIds: Qwen3Tokens.SpecialTokenIds,
        PreTokenizerRegex: Qwen3Tokens.GetPreTokenizerRegex(),
        EosTokenId: Qwen3Tokens.ImEndTokenId,
        PadTokenId: Qwen3Tokens.EndOfTextTokenId,
        Normalizer: new NfcNormalizer(),
        ByteLevel: true);
}
