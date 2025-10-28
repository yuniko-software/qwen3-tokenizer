using Microsoft.ML.Tokenizers;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Tokenizer for Qwen3 models.
/// Based on byte-level BPE (Byte-Pair Encoding) similar to GPT-2.
/// Supports all Qwen3 model variants including Embedding, General, Reranker, and Vision-Language models.
/// </summary>
public partial class Qwen3Tokenizer
{
    private readonly BpeTokenizer _tokenizer;
    private readonly Dictionary<string, int> _addedTokens;
    private readonly IReadOnlySet<int> _specialTokenIds;
    private readonly int _eosTokenId;
    private readonly int _padTokenId;

    /// <summary>
    /// Gets the vocabulary size including added tokens.
    /// </summary>
    public int VocabularySize => _tokenizer.Vocabulary.Count + _addedTokens.Count;

    /// <summary>
    /// Gets the vocabulary dictionary including added tokens.
    /// </summary>
    public IReadOnlyDictionary<string, int> Vocabulary
    {
        get
        {
            var combined = new Dictionary<string, int>(_tokenizer.Vocabulary);
            foreach (var (token, id) in _addedTokens)
            {
                combined[token] = id;
            }
            return combined;
        }
    }

    /// <summary>
    /// Gets all added tokens.
    /// These are treated as atomic during pre-tokenization.
    /// </summary>
    public IReadOnlyDictionary<string, int> AddedTokens => _addedTokens;

    /// <summary>
    /// Gets the token IDs marked as "special": true in HuggingFace.
    /// These are skipped during decoding when skipSpecialTokens=true.
    /// </summary>
    public IReadOnlySet<int> SpecialTokenIds => _specialTokenIds;

    /// <summary>
    /// Creates a Qwen3 tokenizer from vocabulary and merges files.
    /// </summary>
    /// <param name="vocabPath">Path to vocab.json file</param>
    /// <param name="mergesPath">Path to merges.txt file</param>
    /// <param name="options">Tokenizer configuration options.</param>
    private Qwen3Tokenizer(
        string vocabPath,
        string mergesPath,
        Qwen3TokenizerOptions options)
    {
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"Vocabulary file not found: {vocabPath}");
        }

        if (!File.Exists(mergesPath))
        {
            throw new FileNotFoundException($"Merges file not found: {mergesPath}");
        }

        _addedTokens = new Dictionary<string, int>(options.AddedTokens);
        _specialTokenIds = options.SpecialTokenIds;
        _eosTokenId = options.EosTokenId;
        _padTokenId = options.PadTokenId;

        var bpeOptions = new BpeOptions(vocabPath, mergesPath)
        {
            ByteLevel = options.ByteLevel,
            Normalizer = options.Normalizer,
            PreTokenizer = new RegexPreTokenizer(options.PreTokenizerRegex, specialTokens: _addedTokens),
            SpecialTokens = _addedTokens
        };

        _tokenizer = BpeTokenizer.Create(bpeOptions);
    }

    /// <summary>
    /// Creates a Qwen3 tokenizer directly from local vocabulary and merges files.
    /// </summary>
    /// <param name="vocabPath">Path to vocab.json file</param>
    /// <param name="mergesPath">Path to merges.txt file</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    public static Qwen3Tokenizer FromFiles(
        string vocabPath,
        string mergesPath,
        Qwen3TokenizerOptions? options = null)
    {
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default);
    }

    /// <summary>
    /// Downloads tokenizer files from HuggingFace and creates a Qwen3 tokenizer.
    /// </summary>
    /// <param name="modelName">Model name (e.g., "Qwen/Qwen3-0.6B", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-VL-30B-A3B-Instruct")</param>
    /// <param name="cacheDir">Directory to cache downloaded files. If null, uses temporary directory.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <param name="httpClient">Optional HttpClient to use for downloads. If null, creates a new one.</param>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    public static Qwen3Tokenizer FromHuggingFace(
        string modelName = "Qwen/Qwen3-0.6B",
        string? cacheDir = null,
        Qwen3TokenizerOptions? options = null,
        HttpClient? httpClient = null,
        IProgress<DownloadProgress>? progress = null)
    {
        var provider = new HuggingFaceFileProvider(modelName, cacheDir, httpClient);
        var (vocabPath, mergesPath) = provider.GetFiles(progress);
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default);
    }

    /// <summary>
    /// Creates a Qwen3 tokenizer using a custom file provider.
    /// </summary>
    /// <param name="fileProvider">The file provider to use for obtaining tokenizer files.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    public static Qwen3Tokenizer FromProvider(
        ITokenizerFileProvider fileProvider,
        Qwen3TokenizerOptions? options = null,
        IProgress<DownloadProgress>? progress = null)
    {
        var (vocabPath, mergesPath) = fileProvider.GetFiles(progress);
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default);
    }

    /// <summary>
    /// Asynchronously downloads tokenizer files from HuggingFace and creates a Qwen3 tokenizer.
    /// </summary>
    /// <param name="modelName">Model name (e.g., "Qwen/Qwen3-0.6B", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-VL-30B-A3B-Instruct")</param>
    /// <param name="cacheDir">Directory to cache downloaded files. If null, uses temporary directory.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <param name="httpClient">Optional HttpClient to use for downloads. If null, creates a new one.</param>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    public static async Task<Qwen3Tokenizer> FromHuggingFaceAsync(
        string modelName = "Qwen/Qwen3-0.6B",
        string? cacheDir = null,
        Qwen3TokenizerOptions? options = null,
        HttpClient? httpClient = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var provider = new HuggingFaceFileProvider(modelName, cacheDir, httpClient);
        var (vocabPath, mergesPath) = await provider.GetFilesAsync(progress, cancellationToken).ConfigureAwait(false);
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default);
    }

    /// <summary>
    /// Asynchronously creates a Qwen3 tokenizer using a custom file provider.
    /// </summary>
    /// <param name="fileProvider">The file provider to use for obtaining tokenizer files.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    public static async Task<Qwen3Tokenizer> FromProviderAsync(
        ITokenizerFileProvider fileProvider,
        Qwen3TokenizerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var (vocabPath, mergesPath) = await fileProvider.GetFilesAsync(progress, cancellationToken).ConfigureAwait(false);
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default);
    }

    /// <summary>
    /// Encodes text into token IDs.
    /// </summary>
    /// <param name="text">Input text to tokenize</param>
    /// <param name="addEos">Whether to add EOS token at the end. Default is false to match HuggingFace behavior.</param>
    /// <returns>Array of token IDs</returns>
    public int[] Encode(string text, bool addEos = false)
    {
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);

        if (addEos)
        {
            var result = new int[ids.Count + 1];
            for (int i = 0; i < ids.Count; i++)
            {
                result[i] = ids[i];
            }
            result[ids.Count] = _eosTokenId;
            return result;
        }

        if (ids is int[] array)
        {
            return array;
        }

        var idsArray = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            idsArray[i] = ids[i];
        }

        return idsArray;
    }

    /// <summary>
    /// Encodes text and returns detailed encoding information.
    /// </summary>
    /// <param name="text">Input text to tokenize</param>
    /// <param name="addEos">Whether to add EOS token at the end. Default is false to match HuggingFace behavior.</param>
    /// <returns>Detailed encoding result with token IDs, strings, and offsets (in UTF-16 char indices)</returns>
    /// <remarks>
    /// Offsets are returned as UTF-16 char indices (matching C# string indexing).
    /// Note that emojis and other characters outside the Basic Multilingual Plane use 2 UTF-16 code units (surrogate pairs).
    /// </remarks>
    public EncodingResult EncodeDetailed(string text, bool addEos = false)
    {
        IReadOnlyList<EncodedToken> encodedTokens = _tokenizer.EncodeToTokens(text, out string? normalizedText);

        var ids = new int[encodedTokens.Count + (addEos ? 1 : 0)];
        var tokens = new string[encodedTokens.Count + (addEos ? 1 : 0)];
        var offsets = new (int Index, int Length)[encodedTokens.Count + (addEos ? 1 : 0)];

        for (int i = 0; i < encodedTokens.Count; i++)
        {
            ids[i] = encodedTokens[i].Id;
            tokens[i] = _tokenizer.Decode([encodedTokens[i].Id]) ?? string.Empty;
            offsets[i] = (encodedTokens[i].Offset.Start.Value, encodedTokens[i].Offset.End.Value - encodedTokens[i].Offset.Start.Value);
        }

        if (addEos)
        {
            ids[encodedTokens.Count] = _eosTokenId;
            tokens[encodedTokens.Count] = Qwen3Tokens.ImEnd;
            offsets[encodedTokens.Count] = (text.Length, 0);
        }

        return new EncodingResult(ids, tokens, offsets);
    }

    /// <summary>
    /// Encodes a batch of texts.
    /// </summary>
    /// <param name="texts">Collection of texts to encode</param>
    /// <param name="addEos">Whether to add EOS token at the end. Default is false to match HuggingFace behavior.</param>
    /// <returns>Array of token ID arrays</returns>
    public int[][] EncodeBatch(IEnumerable<string> texts, bool addEos = false)
    {
        return [.. texts.Select(text => Encode(text, addEos))];
    }

    /// <summary>
    /// Counts the number of tokens in the text without full encoding.
    /// </summary>
    /// <param name="text">Input text</param>
    /// <param name="addEos">Whether to add EOS token at the end. Default is false to match HuggingFace behavior.</param>
    /// <returns>Token count</returns>
    public int CountTokens(string text, bool addEos = false)
    {
        int count = _tokenizer.CountTokens(text);
        return addEos ? count + 1 : count;
    }

    /// <summary>
    /// Decodes token IDs back to text.
    /// </summary>
    /// <param name="ids">Token IDs to decode</param>
    /// <param name="skipSpecialTokens">Whether to skip special tokens in output (only skips tokens marked as "special": true)</param>
    /// <returns>Decoded text</returns>
    public string Decode(int[] ids, bool skipSpecialTokens = true)
    {
        if (skipSpecialTokens)
        {
            ids = [.. ids.Where(id => !_specialTokenIds.Contains(id))];
        }

        return _tokenizer.Decode(ids) ?? string.Empty;
    }

    /// <summary>
    /// Decodes a batch of token ID sequences.
    /// </summary>
    public string[] DecodeBatch(IEnumerable<int[]> idsBatch, bool skipSpecialTokens = true)
    {
        return [.. idsBatch.Select(ids => Decode(ids, skipSpecialTokens))];
    }

    /// <summary>
    /// Gets an added token ID by name.
    /// </summary>
    public int? GetAddedTokenId(string tokenName)
    {
        return _addedTokens.TryGetValue(tokenName, out var id) ? id : null;
    }

    /// <summary>
    /// Prepares input for ONNX Runtime inference.
    /// </summary>
    /// <param name="text">Input text</param>
    /// <param name="maxLength">Maximum sequence length (will pad or truncate)</param>
    /// <returns>Tuple of (input_ids, attention_mask)</returns>
    public (long[] InputIds, long[] AttentionMask) PrepareForOnnx(string text, int maxLength = 512)
    {
        var ids = Encode(text, addEos: true);

        var inputIds = new long[maxLength];
        var attentionMask = new long[maxLength];

        int actualLength = Math.Min(ids.Length, maxLength);

        for (int i = 0; i < actualLength; i++)
        {
            inputIds[i] = ids[i];
            attentionMask[i] = 1;
        }

        for (int i = actualLength; i < maxLength; i++)
        {
            inputIds[i] = _padTokenId;
            attentionMask[i] = 0;
        }

        return (inputIds, attentionMask);
    }

    /// <summary>
    /// Prepares a batch of texts for ONNX Runtime inference.
    /// </summary>
    public (long[,] InputIds, long[,] AttentionMask) PrepareForOnnxBatch(string[] texts, int maxLength = 512)
    {
        int batchSize = texts.Length;
        var inputIds = new long[batchSize, maxLength];
        var attentionMask = new long[batchSize, maxLength];

        for (int i = 0; i < batchSize; i++)
        {
            var (ids, mask) = PrepareForOnnx(texts[i], maxLength);

            for (int j = 0; j < maxLength; j++)
            {
                inputIds[i, j] = ids[j];
                attentionMask[i, j] = mask[j];
            }
        }

        return (inputIds, attentionMask);
    }
}