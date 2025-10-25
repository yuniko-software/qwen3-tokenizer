using System;
using System.Text;
using Microsoft.ML.Tokenizers;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Normalizer that applies Unicode NFC (Canonical Decomposition followed by Canonical Composition) normalization.
/// This is used by Qwen tokenizers to normalize text before tokenization.
/// </summary>
internal sealed class NfcNormalizer : Normalizer
{
    /// <summary>
    /// Normalizes the input string using Unicode NFC normalization.
    /// </summary>
    /// <param name="original">The original string to normalize.</param>
    /// <returns>The normalized string in NFC form.</returns>
    public override string Normalize(string original)
    {
        if (string.IsNullOrEmpty(original))
        {
            return original;
        }

        return original.Normalize(NormalizationForm.FormC);
    }

    /// <summary>
    /// Normalizes the input span using Unicode NFC normalization.
    /// </summary>
    /// <param name="original">The original span to normalize.</param>
    /// <returns>The normalized string in NFC form.</returns>
    public override string Normalize(ReadOnlySpan<char> original)
    {
        if (original.IsEmpty)
        {
            return string.Empty;
        }

        return original.ToString().Normalize(NormalizationForm.FormC);
    }
}