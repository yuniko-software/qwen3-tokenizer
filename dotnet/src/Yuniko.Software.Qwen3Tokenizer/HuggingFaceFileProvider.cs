namespace Yuniko.Software.Qwen3Tokenizer;

public sealed class HuggingFaceFileProvider : ITokenizerFileProvider
{
    private readonly string _modelName;
    private readonly string _cacheDir;
    private readonly HttpClient? _httpClient;
    private readonly bool _ownsHttpClient;
    private readonly HuggingFaceConfig _config;
    private readonly SemaphoreSlim _downloadLock = new(1, 1);

    public HuggingFaceFileProvider(
        string modelName,
        string? cacheDir = null,
        HttpClient? httpClient = null,
        HuggingFaceConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(modelName);

        _modelName = modelName;
        _cacheDir = cacheDir ?? Path.Combine(Path.GetTempPath(), "qwen3_tokenizer");
        _httpClient = httpClient;
        _ownsHttpClient = httpClient == null;
        _config = config ?? HuggingFaceConfig.Default;
    }

    public (string VocabPath, string MergesPath) GetFiles()
    {
        return GetFilesAsync().ConfigureAwait(false).GetAwaiter().GetResult();
    }

    public async Task<(string VocabPath, string MergesPath)> GetFilesAsync(
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(_cacheDir);

        var vocabPath = Path.Combine(_cacheDir, _config.VocabFileName);
        var mergesPath = Path.Combine(_cacheDir, _config.MergesFileName);

        await _downloadLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var client = GetOrCreateHttpClient();
            try
            {
                if (!File.Exists(vocabPath))
                {
                    var vocabUrl = _config.GetVocabUrl(_modelName);
                    await DownloadFileAsync(client, vocabUrl, vocabPath, cancellationToken).ConfigureAwait(false);
                }

                if (!File.Exists(mergesPath))
                {
                    var mergesUrl = _config.GetMergesUrl(_modelName);
                    await DownloadFileAsync(client, mergesUrl, mergesPath, cancellationToken).ConfigureAwait(false);
                }

                return (vocabPath, mergesPath);
            }
            finally
            {
                if (_ownsHttpClient)
                {
                    client.Dispose();
                }
            }
        }
        finally
        {
            _downloadLock.Release();
        }
    }

    private HttpClient GetOrCreateHttpClient()
    {
        if (_httpClient != null)
        {
            return _httpClient;
        }

        return new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10),
        };
    }

    private static async Task DownloadFileAsync(
        HttpClient client,
        string url,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        using var response = await client
            .GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        response.EnsureSuccessStatusCode();

        var tempPath = destinationPath + ".tmp";

        try
        {
            await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            await using (var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, useAsync: true))
            {
                var buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = await contentStream.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken).ConfigureAwait(false)) > 0)
                {
                    await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken).ConfigureAwait(false);
                }

                await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
            }

            File.Move(tempPath, destinationPath, overwrite: true);
        }
        catch
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
            throw;
        }
    }
}
