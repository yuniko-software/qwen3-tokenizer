namespace Yuniko.Software.Qwen3Tokenizer.Tests;

public class CustomProviderTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _vocabPath;
    private readonly string _mergesPath;

    public CustomProviderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwen3_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDir);

        _vocabPath = Path.Combine(_tempDir, "vocab.json");
        _mergesPath = Path.Combine(_tempDir, "merges.txt");

        var realProvider = new HuggingFaceFileProvider("Qwen/Qwen3-0.6B", _tempDir);
        (var realVocab, var realMerges) = realProvider.GetFiles();

        _vocabPath = realVocab;
        _mergesPath = realMerges;
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
    public void FromProvider_WithCustomProvider_CreatesTokenizer()
    {
        var provider = new TestFileProvider(_vocabPath, _mergesPath);

        var tokenizer = Qwen3Tokenizer.FromProvider(provider, isForEmbeddingModel: false);

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public async Task FromProviderAsync_WithCustomProvider_CreatesTokenizer()
    {
        var provider = new TestFileProvider(_vocabPath, _mergesPath);

        var tokenizer = await Qwen3Tokenizer.FromProviderAsync(provider, isForEmbeddingModel: false);

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void FromProvider_WithCustomOptions_UsesProvidedOptions()
    {
        var provider = new TestFileProvider(_vocabPath, _mergesPath);
        var options = Qwen3TokenizerOptions.Default;

        var tokenizer = Qwen3Tokenizer.FromProvider(provider, isForEmbeddingModel: false, options);

        Assert.NotNull(tokenizer);
        Assert.Equal(Qwen3Tokens.EndOfTextTokenId, options.PadTokenId);
    }

    [Fact]
    public async Task FromProviderAsync_WithCustomOptions_UsesProvidedOptions()
    {
        var provider = new TestFileProvider(_vocabPath, _mergesPath);
        var options = Qwen3TokenizerOptions.Default;

        var tokenizer = await Qwen3Tokenizer.FromProviderAsync(provider, isForEmbeddingModel: false, options);

        Assert.NotNull(tokenizer);
        Assert.Equal(Qwen3Tokens.EndOfTextTokenId, options.PadTokenId);
    }

    [Fact]
    public void FromProvider_TokenizerFunctionsCorrectly()
    {
        var provider = new TestFileProvider(_vocabPath, _mergesPath);
        var tokenizer = Qwen3Tokenizer.FromProvider(provider, isForEmbeddingModel: false);

        const string text = "Hello world";
        var tokens = tokenizer.Encode(text);

        Assert.NotEmpty(tokens);

        var decoded = tokenizer.Decode(tokens);
        Assert.Equal(text, decoded);
    }

    [Fact]
    public async Task FromProviderAsync_TokenizerFunctionsCorrectly()
    {
        var provider = new TestFileProvider(_vocabPath, _mergesPath);
        var tokenizer = await Qwen3Tokenizer.FromProviderAsync(provider, isForEmbeddingModel: false);

        const string text = "Hello world";
        var tokens = tokenizer.Encode(text);

        Assert.NotEmpty(tokens);

        var decoded = tokenizer.Decode(tokens);
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void FromProvider_CallsGetFilesOnProvider()
    {
        var provider = new TestFileProvider(_vocabPath, _mergesPath);

        Qwen3Tokenizer.FromProvider(provider, isForEmbeddingModel: false);

        Assert.True(provider.GetFilesCalled);
    }

    [Fact]
    public async Task FromProviderAsync_CallsGetFilesAsyncOnProvider()
    {
        var provider = new TestFileProvider(_vocabPath, _mergesPath);

        await Qwen3Tokenizer.FromProviderAsync(provider, isForEmbeddingModel: false);

        Assert.True(provider.GetFilesAsyncCalled);
    }

    [Fact]
    public async Task FromProviderAsync_SupportsCancellation()
    {
        var provider = new CancellableTestFileProvider();
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await Qwen3Tokenizer.FromProviderAsync(provider, isForEmbeddingModel: false, cancellationToken: cts.Token));
    }

    private class TestFileProvider : ITokenizerFileProvider
    {
        private readonly string _vocabPath;
        private readonly string _mergesPath;

        public bool GetFilesCalled { get; private set; }
        public bool GetFilesAsyncCalled { get; private set; }

        public TestFileProvider(string vocabPath, string mergesPath)
        {
            _vocabPath = vocabPath;
            _mergesPath = mergesPath;
        }

        public (string VocabPath, string MergesPath) GetFiles()
        {
            GetFilesCalled = true;
            return (_vocabPath, _mergesPath);
        }

        public Task<(string VocabPath, string MergesPath)> GetFilesAsync(CancellationToken cancellationToken = default)
        {
            GetFilesAsyncCalled = true;
            return Task.FromResult((_vocabPath, _mergesPath));
        }
    }

    private class CancellableTestFileProvider : ITokenizerFileProvider
    {
        public (string VocabPath, string MergesPath) GetFiles()
        {
            return ("", "");
        }

        public async Task<(string VocabPath, string MergesPath)> GetFilesAsync(CancellationToken cancellationToken = default)
        {
            await Task.Delay(10000, cancellationToken);
            return ("", "");
        }
    }
}
