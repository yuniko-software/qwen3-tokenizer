namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Detailed encoding result with tokens and offsets.
/// </summary>
public record EncodingResult(int[] Ids, string[] Tokens, (int, int)[] Offsets);
