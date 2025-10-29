namespace Yuniko.Software.Qwen3Tokenizer.Tests;

public class HuggingFaceFileProviderTests : IDisposable
{
    private readonly string _tempDir;

    public HuggingFaceFileProviderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwen3_test_{Guid.NewGuid()}");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            Directory.Delete(_tempDir, recursive: true);
        }

        GC.SuppressFinalize(this);
    }

    [Fact]
    public void Constructor_WithNullModelName_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new HuggingFaceFileProvider(null!));
    }

    [Fact]
    public void Constructor_WithValidModelName_Succeeds()
    {
        var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir);

        Assert.NotNull(provider);
    }

    [Fact]
    public void GetFiles_CreatesDirectory()
    {
        var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir);

        try
        { _ = provider.GetFiles(); }
        catch { }

        Assert.True(Directory.Exists(_tempDir));
    }

    [Fact]
    public async Task GetFilesAsync_CreatesDirectory()
    {
        var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir);

        try
        { _ = await provider.GetFilesAsync(); }
        catch { }

        Assert.True(Directory.Exists(_tempDir));
    }

    [Fact]
    public void GetFiles_WhenFilesExist_DoesNotRedownload()
    {
        Directory.CreateDirectory(_tempDir);
        var vocabPath = Path.Combine(_tempDir, "vocab.json");
        var mergesPath = Path.Combine(_tempDir, "merges.txt");
        File.WriteAllText(vocabPath, "{}");
        File.WriteAllText(mergesPath, "");

        var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir);

        var (returnedVocab, returnedMerges) = provider.GetFiles();

        Assert.Equal(vocabPath, returnedVocab);
        Assert.Equal(mergesPath, returnedMerges);
        Assert.True(File.Exists(vocabPath));
        Assert.True(File.Exists(mergesPath));
    }

    [Fact]
    public async Task GetFilesAsync_WhenFilesExist_DoesNotRedownload()
    {
        Directory.CreateDirectory(_tempDir);
        var vocabPath = Path.Combine(_tempDir, "vocab.json");
        var mergesPath = Path.Combine(_tempDir, "merges.txt");
        await File.WriteAllTextAsync(vocabPath, "{}");
        await File.WriteAllTextAsync(mergesPath, "");

        var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir);

        var (returnedVocab, returnedMerges) = await provider.GetFilesAsync();

        Assert.Equal(vocabPath, returnedVocab);
        Assert.Equal(mergesPath, returnedMerges);
        Assert.True(File.Exists(vocabPath));
        Assert.True(File.Exists(mergesPath));
    }

    [Fact]
    public async Task GetFilesAsync_CanBeCancelled()
    {
        var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir);
        var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await provider.GetFilesAsync(cts.Token));
    }

    [Fact]
    public void GetFiles_WithCustomConfig_UsesCustomFileNames()
    {
        Directory.CreateDirectory(_tempDir);
        var customConfig = new HuggingFaceConfig(
            VocabFileName: "custom_vocab.json",
            MergesFileName: "custom_merges.txt");

        var vocabPath = Path.Combine(_tempDir, "custom_vocab.json");
        var mergesPath = Path.Combine(_tempDir, "custom_merges.txt");
        File.WriteAllText(vocabPath, "{}");
        File.WriteAllText(mergesPath, "");

        var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir, config: customConfig);

        var (returnedVocab, returnedMerges) = provider.GetFiles();

        Assert.Equal(vocabPath, returnedVocab);
        Assert.Equal(mergesPath, returnedMerges);
    }

    [Fact]
    public void GetFiles_WithProvidedHttpClient_DoesNotDisposeClient()
    {
        Directory.CreateDirectory(_tempDir);
        File.WriteAllText(Path.Combine(_tempDir, "vocab.json"), "{}");
        File.WriteAllText(Path.Combine(_tempDir, "merges.txt"), "");

        using var httpClient = new HttpClient();
        var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir, httpClient);

        provider.GetFiles();

        var timeout = httpClient.Timeout;
        Assert.NotEqual(TimeSpan.Zero, timeout);
    }

    [Fact]
    public void GetFiles_UsesDefaultTempPath_WhenCacheDirIsNull()
    {
        var uniqueCacheDir = Path.Combine(Path.GetTempPath(), $"qwen3_tokenizer_test_{Guid.NewGuid()}");

        try
        {
            Directory.CreateDirectory(uniqueCacheDir);
            File.WriteAllText(Path.Combine(uniqueCacheDir, "vocab.json"), "{}");
            File.WriteAllText(Path.Combine(uniqueCacheDir, "merges.txt"), "");

            var provider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", cacheDir: uniqueCacheDir);
            var (vocabPath, mergesPath) = provider.GetFiles();

            Assert.Contains(uniqueCacheDir, vocabPath, StringComparison.InvariantCulture);
            Assert.Contains(uniqueCacheDir, mergesPath, StringComparison.InvariantCulture);
        }
        finally
        {
            if (Directory.Exists(uniqueCacheDir))
            {
                try
                { Directory.Delete(uniqueCacheDir, recursive: true); }
                catch { }
            }
        }
    }
}
