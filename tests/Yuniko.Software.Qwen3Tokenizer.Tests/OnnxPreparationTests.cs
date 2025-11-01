namespace Yuniko.Software.Qwen3Tokenizer.Tests;

public class OnnxPreparationTests
{
    private readonly Qwen3Tokenizer _tokenizer;

    public OnnxPreparationTests()
    {
        _tokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-0.6B");
    }

    [Theory]
    [InlineData("Hello world", 10)]
    [InlineData("Short", 20)]
    [InlineData("This is a longer sentence that will be truncated", 5)]
    public void PrepareForOnnx_ReturnsCorrectLength(string text, int maxLength)
    {
        var result = _tokenizer.PrepareForOnnx(text, maxLength);

        Assert.Equal(maxLength, result.InputIds.Length);
        Assert.Equal(maxLength, result.AttentionMask.Length);
        Assert.Equal(maxLength, result.PositionIds.Length);
        Assert.Equal(maxLength, result.SequenceLength);
    }

    [Fact]
    public void PrepareForOnnx_PadsShortSequences()
    {
        const string text = "Hi";
        const int maxLength = 10;

        var result = _tokenizer.PrepareForOnnx(text, maxLength);

        var tokenCount = _tokenizer.CountTokens(text, addEos: true);

        for (int i = 0; i < tokenCount; i++)
        {
            Assert.Equal(1L, result.AttentionMask[i]);
        }

        for (int i = tokenCount; i < maxLength; i++)
        {
            Assert.Equal(0L, result.AttentionMask[i]);
        }
    }

    [Fact]
    public void PrepareForOnnx_TruncatesLongSequences()
    {
        const string text = "This is a very long sentence that will definitely exceed the maximum length limit";
        const int maxLength = 5;

        var result = _tokenizer.PrepareForOnnx(text, maxLength);

        Assert.Equal(maxLength, result.InputIds.Length);
        Assert.All(result.AttentionMask, mask => Assert.Equal(1L, mask));

        // All position IDs should be sequential when all tokens are real (non-padded), regardless of whether truncation occurred
        for (int i = 0; i < maxLength; i++)
        {
            Assert.Equal(i, result.PositionIds[i]);
        }
    }

    [Fact]
    public void PrepareForOnnx_CreatesCorrectPositionIds()
    {
        const string text = "Hello world";
        const int maxLength = 10;

        var result = _tokenizer.PrepareForOnnx(text, maxLength);
        var tokenCount = _tokenizer.CountTokens(text, addEos: true);

        // Position IDs should be sequential for real tokens
        for (int i = 0; i < tokenCount; i++)
        {
            Assert.Equal(i, result.PositionIds[i]);
        }

        // Position IDs should be 0 for padding tokens
        for (int i = tokenCount; i < maxLength; i++)
        {
            Assert.Equal(0L, result.PositionIds[i]);
        }
    }
}
