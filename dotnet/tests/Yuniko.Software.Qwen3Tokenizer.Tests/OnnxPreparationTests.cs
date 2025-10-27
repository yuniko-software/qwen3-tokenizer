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
        var (inputIds, attentionMask) = _tokenizer.PrepareForOnnx(text, maxLength);

        Assert.Equal(maxLength, inputIds.Length);
        Assert.Equal(maxLength, attentionMask.Length);
    }

    [Fact]
    public void PrepareForOnnx_PadsShortSequences()
    {
        var text = "Hi";
        var maxLength = 10;

        var (inputIds, attentionMask) = _tokenizer.PrepareForOnnx(text, maxLength);

        var tokenCount = _tokenizer.CountTokens(text, addEos: true);

        for (int i = 0; i < tokenCount; i++)
        {
            Assert.Equal(1L, attentionMask[i]);
        }

        for (int i = tokenCount; i < maxLength; i++)
        {
            Assert.Equal(0L, attentionMask[i]);
        }
    }

    [Fact]
    public void PrepareForOnnx_TruncatesLongSequences()
    {
        var text = "This is a very long sentence that will definitely exceed the maximum length limit";
        var maxLength = 5;

        var (inputIds, attentionMask) = _tokenizer.PrepareForOnnx(text, maxLength);

        Assert.Equal(maxLength, inputIds.Length);
        Assert.All(attentionMask, mask => Assert.Equal(1L, mask));
    }

    [Fact]
    public void PrepareForOnnxBatch_ReturnsCorrectDimensions()
    {
        var texts = new[] { "First text", "Second text", "Third text" };
        var maxLength = 15;

        var (inputIds, attentionMask) = _tokenizer.PrepareForOnnxBatch(texts, maxLength);

        Assert.Equal(texts.Length, inputIds.GetLength(0));
        Assert.Equal(maxLength, inputIds.GetLength(1));
        Assert.Equal(texts.Length, attentionMask.GetLength(0));
        Assert.Equal(maxLength, attentionMask.GetLength(1));
    }

    [Fact]
    public void PrepareForOnnxBatch_ProducesSameResultsAsIndividual()
    {
        var texts = new[] { "Hello", "World" };
        var maxLength = 10;

        var (batchInputIds, batchAttentionMask) = _tokenizer.PrepareForOnnxBatch(texts, maxLength);

        for (int i = 0; i < texts.Length; i++)
        {
            var (individualInputIds, individualAttentionMask) = _tokenizer.PrepareForOnnx(texts[i], maxLength);

            for (int j = 0; j < maxLength; j++)
            {
                Assert.Equal(individualInputIds[j], batchInputIds[i, j]);
                Assert.Equal(individualAttentionMask[j], batchAttentionMask[i, j]);
            }
        }
    }
}
