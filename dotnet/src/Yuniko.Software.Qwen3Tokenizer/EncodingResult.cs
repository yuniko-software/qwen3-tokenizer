namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Result of encoding operation with detailed token information.
/// </summary>
public record EncodingResult(
    int[] Ids,
    string[] Tokens,
    (int Index, int Length)[] Offsets);