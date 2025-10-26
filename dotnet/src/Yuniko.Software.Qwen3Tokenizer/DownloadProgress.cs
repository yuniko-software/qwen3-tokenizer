namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Represents progress information for file download operations.
/// </summary>
/// <param name="FileName">Name of the file being downloaded.</param>
/// <param name="BytesDownloaded">Number of bytes downloaded so far.</param>
/// <param name="TotalBytes">Total number of bytes to download, or null if unknown.</param>
/// <param name="Status">Current status message.</param>
public sealed record DownloadProgress(
    string FileName,
    long BytesDownloaded,
    long? TotalBytes,
    string Status)
{
    /// <summary>
    /// Gets the download progress as a percentage (0-100), or null if total size is unknown.
    /// </summary>
    public double? PercentComplete => TotalBytes.HasValue && TotalBytes.Value > 0
        ? (double)BytesDownloaded / TotalBytes.Value * 100
        : null;
}
