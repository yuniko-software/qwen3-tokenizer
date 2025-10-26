namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Configuration for downloading tokenizer files from HuggingFace or compatible model hubs.
/// </summary>
/// <param name="BaseUrl">Base URL for the model hub. Defaults to "https://huggingface.co".</param>
/// <param name="Branch">Branch or revision to download from. Defaults to "main".</param>
/// <param name="VocabFileName">Vocabulary file name. Defaults to "vocab.json".</param>
/// <param name="MergesFileName">Merges file name. Defaults to "merges.txt".</param>
public sealed record HuggingFaceConfig(
    string BaseUrl = "https://huggingface.co",
    string Branch = "main",
    string VocabFileName = "vocab.json",
    string MergesFileName = "merges.txt")
{
    /// <summary>
    /// Default configuration for standard HuggingFace model hub.
    /// </summary>
    public static readonly HuggingFaceConfig Default = new();

    /// <summary>
    /// Gets the full URL for a file in the model repository.
    /// </summary>
    /// <param name="modelName">Model name (e.g., "Qwen/Qwen3-Embedding-0.6B").</param>
    /// <param name="fileName">File name to download.</param>
    /// <returns>Full URL to the file.</returns>
    public string GetFileUrl(string modelName, string fileName)
    {
        var baseUrl = BaseUrl.TrimEnd('/');
        return $"{baseUrl}/{modelName}/resolve/{Branch}/{fileName}";
    }

    /// <summary>
    /// Gets the URL for the vocabulary file.
    /// </summary>
    public string GetVocabUrl(string modelName) => GetFileUrl(modelName, VocabFileName);

    /// <summary>
    /// Gets the URL for the merges file.
    /// </summary>
    public string GetMergesUrl(string modelName) => GetFileUrl(modelName, MergesFileName);
}
