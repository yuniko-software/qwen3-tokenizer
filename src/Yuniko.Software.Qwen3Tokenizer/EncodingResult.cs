namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Result of encoding operation with detailed token information.
/// </summary>
/// <param name="Ids">Token IDs.</param>
/// <param name="Tokens">Token strings.</param>
/// <param name="Offsets">Token offsets in the original text (UTF-16 char indices).</param>
public record EncodingResult(
    int[] Ids,
    string[] Tokens,
    (int Index, int Length)[] Offsets);
