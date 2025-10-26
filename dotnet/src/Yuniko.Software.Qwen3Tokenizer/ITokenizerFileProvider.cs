namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Provides tokenizer files (vocabulary and merges) from various sources.
/// </summary>
public interface ITokenizerFileProvider
{
    /// <summary>
    /// Gets the vocabulary and merges file paths synchronously.
    /// </summary>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <returns>Tuple containing paths to vocabulary and merges files.</returns>
    (string VocabPath, string MergesPath) GetFiles(IProgress<DownloadProgress>? progress = null);

    /// <summary>
    /// Gets the vocabulary and merges file paths asynchronously.
    /// </summary>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Tuple containing paths to vocabulary and merges files.</returns>
    Task<(string VocabPath, string MergesPath)> GetFilesAsync(
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);
}
