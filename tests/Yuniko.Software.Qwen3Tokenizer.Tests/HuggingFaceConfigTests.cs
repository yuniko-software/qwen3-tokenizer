namespace Yuniko.Software.Qwen3Tokenizer.Tests;

public class HuggingFaceConfigTests
{
    [Fact]
    public void Default_HasExpectedValues()
    {
        var config = HuggingFaceConfig.Default;

        Assert.Equal("https://huggingface.co", config.BaseUrl);
        Assert.Equal("main", config.Branch);
        Assert.Equal("vocab.json", config.VocabFileName);
        Assert.Equal("merges.txt", config.MergesFileName);
    }

    [Theory]
    [InlineData("Qwen/Qwen3-0.6B", "vocab.json", "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/vocab.json")]
    [InlineData("Qwen/Qwen3-VL-30B-A3B-Instruct", "merges.txt", "https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/resolve/main/merges.txt")]
    [InlineData("user/model-name", "config.json", "https://huggingface.co/user/model-name/resolve/main/config.json")]
    public void GetFileUrl_ReturnsCorrectUrl(string modelName, string fileName, string expected)
    {
        var config = HuggingFaceConfig.Default;

        var actual = config.GetFileUrl(modelName, fileName);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void GetFileUrl_TrimsTrailingSlashFromBaseUrl()
    {
        var config = new HuggingFaceConfig(BaseUrl: "https://example.com/");

        var url = config.GetFileUrl("user/model", "file.txt");

        Assert.Equal("https://example.com/user/model/resolve/main/file.txt", url);
    }

    [Fact]
    public void GetVocabUrl_ReturnsCorrectUrl()
    {
        var config = HuggingFaceConfig.Default;

        var url = config.GetVocabUrl("Qwen/Qwen3-0.6B");

        Assert.Equal("https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/vocab.json", url);
    }

    [Fact]
    public void GetMergesUrl_ReturnsCorrectUrl()
    {
        var config = HuggingFaceConfig.Default;

        var url = config.GetMergesUrl("Qwen/Qwen3-0.6B");

        Assert.Equal("https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/merges.txt", url);
    }

    [Fact]
    public void CustomConfig_WithDifferentBranch_UsesCustomBranch()
    {
        var config = new HuggingFaceConfig(Branch: "dev");

        var url = config.GetFileUrl("user/model", "vocab.json");

        Assert.Equal("https://huggingface.co/user/model/resolve/dev/vocab.json", url);
    }

    [Fact]
    public void CustomConfig_WithCustomBaseUrl_UsesCustomUrl()
    {
        var config = new HuggingFaceConfig(BaseUrl: "https://custom-hub.example.com");

        var url = config.GetFileUrl("user/model", "vocab.json");

        Assert.Equal("https://custom-hub.example.com/user/model/resolve/main/vocab.json", url);
    }

    [Fact]
    public void CustomConfig_WithCustomFileNames_UsesCustomNames()
    {
        var config = new HuggingFaceConfig(
            VocabFileName: "tokenizer.json",
            MergesFileName: "bpe.txt");

        var vocabUrl = config.GetVocabUrl("user/model");
        var mergesUrl = config.GetMergesUrl("user/model");

        Assert.Equal("https://huggingface.co/user/model/resolve/main/tokenizer.json", vocabUrl);
        Assert.Equal("https://huggingface.co/user/model/resolve/main/bpe.txt", mergesUrl);
    }

    [Fact]
    public void RecordEquality_WorksCorrectly()
    {
        var config1 = new HuggingFaceConfig();
        var config2 = new HuggingFaceConfig();
        var config3 = new HuggingFaceConfig(Branch: "dev");

        Assert.Equal(config1, config2);
        Assert.NotEqual(config1, config3);
    }
}
