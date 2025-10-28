namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Provides tokenizer files by downloading from HuggingFace model hub.
/// </summary>
public sealed class HuggingFaceFileProvider : ITokenizerFileProvider
{
    private readonly string _modelName;
    private readonly string _cacheDir;
    private readonly HttpClient? _httpClient;
    private readonly bool _ownsHttpClient;
    private readonly HuggingFaceConfig _config;

    /// <summary>
    /// Creates a new HuggingFace file provider.
    /// </summary>
    /// <param name="modelName">Model name (e.g., "Qwen/Qwen3-0.6B", "Qwen/Qwen3-VL-30B-A3B-Instruct").</param>
    /// <param name="cacheDir">Directory to cache downloaded files. If null, uses temporary directory.</param>
    /// <param name="httpClient">Optional HttpClient to use for downloads. If null, creates a new one.</param>
    /// <param name="config">Configuration for HuggingFace downloads. If null, uses HuggingFaceConfig.Default.</param>
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

    /// <inheritdoc/>
    public (string VocabPath, string MergesPath) GetFiles()
    {
        Directory.CreateDirectory(_cacheDir);

        var vocabPath = Path.Combine(_cacheDir, _config.VocabFileName);
        var mergesPath = Path.Combine(_cacheDir, _config.MergesFileName);

        var client = GetOrCreateHttpClient();
        try
        {
            if (!File.Exists(vocabPath))
            {
                var vocabUrl = _config.GetVocabUrl(_modelName);
                DownloadFile(client, vocabUrl, vocabPath);
            }

            if (!File.Exists(mergesPath))
            {
                var mergesUrl = _config.GetMergesUrl(_modelName);
                DownloadFile(client, mergesUrl, mergesPath);
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

    /// <inheritdoc/>
    public async Task<(string VocabPath, string MergesPath)> GetFilesAsync(
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(_cacheDir);

        var vocabPath = Path.Combine(_cacheDir, _config.VocabFileName);
        var mergesPath = Path.Combine(_cacheDir, _config.MergesFileName);

        var client = GetOrCreateHttpClient();
        try
        {
            if (!File.Exists(vocabPath))
            {
                var vocabUrl = _config.GetVocabUrl(_modelName);
                await DownloadFileAsync(client, vocabUrl, vocabPath, cancellationToken);
            }

            if (!File.Exists(mergesPath))
            {
                var mergesUrl = _config.GetMergesUrl(_modelName);
                await DownloadFileAsync(client, mergesUrl, mergesPath, cancellationToken);
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

    private HttpClient GetOrCreateHttpClient()
    {
        if (_httpClient != null)
        {
            return _httpClient;
        }

        return new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
    }

    private static void DownloadFile(
        HttpClient client,
        string url,
        string destinationPath)
    {
        var response = client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();

        using var contentStream = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult();
        using var fileStream = File.Create(destinationPath);

        var buffer = new byte[8192];
        int bytesRead;

        while ((bytesRead = contentStream.Read(buffer, 0, buffer.Length)) > 0)
        {
            fileStream.Write(buffer, 0, bytesRead);
        }
    }

    private static async Task DownloadFileAsync(
        HttpClient client,
        string url,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var tempPath = destinationPath + ".tmp";

        try
        {
            using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using (var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, useAsync: true))
            {
                var buffer = new byte[8192];
                int bytesRead;

                while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                {
                    await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken).ConfigureAwait(false);
                }

                await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
            }

            // Only move temp file to destination if download completed successfully
            File.Move(tempPath, destinationPath, overwrite: true);
        }
        catch
        {
            // Clean up temporary file on any error (including cancellation)
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
            throw;
        }
    }
}
