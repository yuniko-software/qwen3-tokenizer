namespace Yuniko.Software.Qwen3Tokenizer.Tests;

public class ModelNameValidationTests
{
    [Theory]
    [InlineData("Qwen/Qwen3-0.6B")]
    [InlineData("Qwen/Qwen3-Embedding-0.6B")]
    [InlineData("Qwen/Qwen3-VL-30B-A3B-Instruct")]
    [InlineData("Qwen/Qwen3-Reranker-0.6B")]
    [InlineData("qwen/qwen3-test")]
    [InlineData("custom-org/my-qwen3-model")]
    public void FromHuggingFace_ValidQwen3ModelName_DoesNotThrow(string modelName)
    {
        var exception = Record.Exception(() => Qwen3Tokenizer.FromHuggingFace(modelName));

        Assert.True(exception is null or not ArgumentException,
            $"Should not throw ArgumentException for valid Qwen3 model name: {modelName}");
    }

    [Theory]
    [InlineData("Qwen/Qwen2-0.6B")]
    [InlineData("meta-llama/Llama-3-8B")]
    [InlineData("microsoft/phi-2")]
    [InlineData("Qwen/Qwen-7B")]
    [InlineData("some-random-model")]
    public void FromHuggingFace_NonQwen3ModelName_ThrowsArgumentException(string modelName)
    {
        var exception = Assert.Throws<ArgumentException>(() => Qwen3Tokenizer.FromHuggingFace(modelName));

        Assert.Equal("modelName", exception.ParamName);
        Assert.Contains("qwen3", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FromHuggingFace_NullModelName_ThrowsArgumentException()
    {
        var exception = Assert.Throws<ArgumentException>(() => Qwen3Tokenizer.FromHuggingFace(null!));

        Assert.Equal("modelName", exception.ParamName);
        Assert.Contains("cannot be null", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FromHuggingFace_EmptyModelName_ThrowsArgumentException()
    {
        var exception = Assert.Throws<ArgumentException>(() => Qwen3Tokenizer.FromHuggingFace(""));

        Assert.Equal("modelName", exception.ParamName);
        Assert.Contains("cannot be null", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FromHuggingFace_WhitespaceModelName_ThrowsArgumentException()
    {
        var exception = Assert.Throws<ArgumentException>(() => Qwen3Tokenizer.FromHuggingFace("   "));

        Assert.Equal("modelName", exception.ParamName);
        Assert.Contains("cannot be null", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("Qwen/Qwen3-0.6B")]
    [InlineData("Qwen/QWEN3-Embedding")]
    [InlineData("org/QwEn3-model")]
    public async Task FromHuggingFaceAsync_ValidQwen3ModelName_DoesNotThrow(string modelName)
    {
        var exception = await Record.ExceptionAsync(async () => await Qwen3Tokenizer.FromHuggingFaceAsync(modelName));

        Assert.True(exception is null or not ArgumentException,
            $"Should not throw ArgumentException for valid Qwen3 model name: {modelName}");
    }

    [Theory]
    [InlineData("Qwen/Qwen2-7B")]
    [InlineData("mistralai/Mistral-7B")]
    public async Task FromHuggingFaceAsync_NonQwen3ModelName_ThrowsArgumentException(string modelName)
    {
        var exception = await Assert.ThrowsAsync<ArgumentException>(async () => await Qwen3Tokenizer.FromHuggingFaceAsync(modelName));

        Assert.Equal("modelName", exception.ParamName);
        Assert.Contains("qwen3", exception.Message, StringComparison.OrdinalIgnoreCase);
    }
}
