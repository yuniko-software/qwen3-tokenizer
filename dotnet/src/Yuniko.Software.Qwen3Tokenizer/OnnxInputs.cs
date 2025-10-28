namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Represents ONNX inputs for Qwen3 models.
/// </summary>
/// <param name="InputIds">Token IDs with padding/truncation applied (shape: [SequenceLength])</param>
/// <param name="AttentionMask">Attention mask indicating real vs padded tokens (1 = real, 0 = padding)</param>
/// <param name="PositionIds">Position indices for each token (shape: [SequenceLength]). Some models may not require this field.</param>
/// <param name="SequenceLength">Sequence length (max length with padding)</param>
public sealed record OnnxInputs(
    long[] InputIds,
    long[] AttentionMask,
    long[] PositionIds,
    int SequenceLength);
