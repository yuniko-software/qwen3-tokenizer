using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Tokenizer for Qwen3-Embedding models.
/// Based on byte-level BPE (Byte-Pair Encoding) similar to GPT-2.
/// </summary>
public partial class Qwen3Tokenizer
{
    private readonly BpeTokenizer _tokenizer;
    private readonly Dictionary<string, int> _specialTokens;

    public int EndOfTextTokenId { get; } = 151643;
    public int ImStartTokenId { get; } = 151644;
    public int ImEndTokenId { get; } = 151645;
    public int ObjectRefStartTokenId { get; } = 151646;
    public int ObjectRefEndTokenId { get; } = 151647;
    public int BoxStartTokenId { get; } = 151648;
    public int BoxEndTokenId { get; } = 151649;
    public int QuadStartTokenId { get; } = 151650;
    public int QuadEndTokenId { get; } = 151651;
    public int VisionStartTokenId { get; } = 151652;
    public int VisionEndTokenId { get; } = 151653;
    public int VisionPadTokenId { get; } = 151654;
    public int ImagePadTokenId { get; } = 151655;
    public int VideoPadTokenId { get; } = 151656;

    public int EosTokenId => EndOfTextTokenId;
    public int PadTokenId => EndOfTextTokenId;
    public int? BosTokenId => null;

    [GeneratedRegex(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", RegexOptions.Compiled)]
    private static partial Regex PreTokenizerRegex();

    /// <summary>
    /// Creates a Qwen3 tokenizer from vocabulary and merges files.
    /// </summary>
    /// <param name="vocabPath">Path to vocab.json file</param>
    /// <param name="mergesPath">Path to merges.txt file</param>
    public Qwen3Tokenizer(string vocabPath, string mergesPath)
    {
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"Vocabulary file not found: {vocabPath}");
        }

        if (!File.Exists(mergesPath))
        {
            throw new FileNotFoundException($"Merges file not found: {mergesPath}");
        }

        // Initialize special tokens dictionary
        _specialTokens = new Dictionary<string, int>
        {
            { "<|endoftext|>", EndOfTextTokenId },
            { "<|im_start|>", ImStartTokenId },
            { "<|im_end|>", ImEndTokenId },
            { "<|object_ref_start|>", ObjectRefStartTokenId },
            { "<|object_ref_end|>", ObjectRefEndTokenId },
            { "<|box_start|>", BoxStartTokenId },
            { "<|box_end|>", BoxEndTokenId },
            { "<|quad_start|>", QuadStartTokenId },
            { "<|quad_end|>", QuadEndTokenId },
            { "<|vision_start|>", VisionStartTokenId },
            { "<|vision_end|>", VisionEndTokenId },
            { "<|vision_pad|>", VisionPadTokenId },
            { "<|image_pad|>", ImagePadTokenId },
            { "<|video_pad|>", VideoPadTokenId }
        };

        // Create BPE tokenizer with proper configuration matching HuggingFace tokenizer.json
        // The Qwen tokenizer uses:
        // - NFC normalization
        // - Regex pre-tokenizer with specific pattern
        // - ByteLevel encoding/decoding
        var regex = PreTokenizerRegex();

        var options = new BpeOptions(vocabPath, mergesPath)
        {
            ByteLevel = true, // Enable byte-level BPE encoding/decoding (critical for Qwen!)
            Normalizer = new NfcNormalizer(), // Unicode NFC normalization
            PreTokenizer = new RegexPreTokenizer(regex, specialTokens: null), // Regex pattern from Qwen tokenizer.json
            SpecialTokens = _specialTokens
        };

        _tokenizer = BpeTokenizer.Create(options);
    }

    /// <summary>
    /// Downloads tokenizer files from HuggingFace and creates a Qwen3 tokenizer.
    /// </summary>
    /// <param name="modelName">Model name (e.g., "Qwen/Qwen3-Embedding-0.6B")</param>
    /// <param name="cacheDir">Directory to cache downloaded files</param>
    public static Qwen3Tokenizer FromPretrained(string modelName = "Qwen/Qwen3-Embedding-0.6B", string? cacheDir = null)
    {
        cacheDir ??= Path.Combine(Path.GetTempPath(), "qwen3_tokenizer");
        Directory.CreateDirectory(cacheDir);

        var vocabPath = Path.Combine(cacheDir, "vocab.json");
        var mergesPath = Path.Combine(cacheDir, "merges.txt");

        if (!File.Exists(vocabPath))
        {
            var vocabUrl = $"https://huggingface.co/{modelName}/resolve/main/vocab.json";
            DownloadFile(vocabUrl, vocabPath);
        }

        if (!File.Exists(mergesPath))
        {
            var mergesUrl = $"https://huggingface.co/{modelName}/resolve/main/merges.txt";
            DownloadFile(mergesUrl, mergesPath);
        }

        return new Qwen3Tokenizer(vocabPath, mergesPath);
    }

    /// <summary>
    /// Asynchronously downloads tokenizer files from HuggingFace and creates a Qwen3 tokenizer.
    /// </summary>
    /// <param name="modelName">Model name (e.g., "Qwen/Qwen3-Embedding-0.6B")</param>
    /// <param name="cacheDir">Directory to cache downloaded files</param>
    /// <param name="cancellationToken">Cancellation token</param>
    public static async Task<Qwen3Tokenizer> FromPretrainedAsync(
        string modelName = "Qwen/Qwen3-Embedding-0.6B",
        string? cacheDir = null,
        CancellationToken cancellationToken = default)
    {
        cacheDir ??= Path.Combine(Path.GetTempPath(), "qwen3_tokenizer");
        Directory.CreateDirectory(cacheDir);

        var vocabPath = Path.Combine(cacheDir, "vocab.json");
        var mergesPath = Path.Combine(cacheDir, "merges.txt");

        if (!File.Exists(vocabPath))
        {
            var vocabUrl = $"https://huggingface.co/{modelName}/resolve/main/vocab.json";
            await DownloadFileAsync(vocabUrl, vocabPath, cancellationToken);
        }

        if (!File.Exists(mergesPath))
        {
            var mergesUrl = $"https://huggingface.co/{modelName}/resolve/main/merges.txt";
            await DownloadFileAsync(mergesUrl, mergesPath, cancellationToken);
        }

        return new Qwen3Tokenizer(vocabPath, mergesPath);
    }

    /// <summary>
    /// Encodes text into token IDs.
    /// </summary>
    /// <param name="text">Input text to tokenize</param>
    /// <param name="addEos">Whether to add EOS token at the end</param>
    /// <returns>Array of token IDs</returns>
    public int[] Encode(string text, bool addEos = true)
    {
        // Use EncodeToIds which returns IReadOnlyList<int>
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);

        if (addEos)
        {
            // Convert to array and append EOS token
            var result = new int[ids.Count + 1];
            for (int i = 0; i < ids.Count; i++)
            {
                result[i] = ids[i];
            }
            result[ids.Count] = EosTokenId;
            return result;
        }

        // Convert IReadOnlyList to array
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
    public EncodingResult EncodeDetailed(string text, bool addEos = true)
    {
        // Use EncodeToTokens which returns IReadOnlyList<EncodedToken>
        IReadOnlyList<EncodedToken> encodedTokens = _tokenizer.EncodeToTokens(text, out string? normalizedText);

        var ids = new int[encodedTokens.Count + (addEos ? 1 : 0)];
        var tokens = new string[encodedTokens.Count + (addEos ? 1 : 0)];
        var offsets = new (int Index, int Length)[encodedTokens.Count + (addEos ? 1 : 0)];

        for (int i = 0; i < encodedTokens.Count; i++)
        {
            ids[i] = encodedTokens[i].Id;
            tokens[i] = encodedTokens[i].Value;
            offsets[i] = (encodedTokens[i].Offset.Start.Value, encodedTokens[i].Offset.End.Value - encodedTokens[i].Offset.Start.Value);
        }

        if (addEos)
        {
            ids[encodedTokens.Count] = EosTokenId;
            tokens[encodedTokens.Count] = "<|endoftext|>";
            offsets[encodedTokens.Count] = (text.Length, 0);
        }

        return new EncodingResult(ids, tokens, offsets);
    }

    /// <summary>
    /// Encodes a batch of texts.
    /// </summary>
    public int[][] EncodeBatch(IEnumerable<string> texts, bool addEos = true)
    {
        return [.. texts.Select(text => Encode(text, addEos))];
    }

    /// <summary>
    /// Counts the number of tokens in the text without full encoding.
    /// </summary>
    /// <param name="text">Input text</param>
    /// <param name="addEos">Whether to add EOS token at the end</param>
    /// <returns>Token count</returns>
    public int CountTokens(string text, bool addEos = true)
    {
        int count = _tokenizer.CountTokens(text);
        return addEos ? count + 1 : count;
    }

    /// <summary>
    /// Decodes token IDs back to text.
    /// </summary>
    /// <param name="ids">Token IDs to decode</param>
    /// <param name="skipSpecialTokens">Whether to skip special tokens in output</param>
    /// <returns>Decoded text</returns>
    public string Decode(int[] ids, bool skipSpecialTokens = true)
    {
        if (skipSpecialTokens)
        {
            var specialTokenIds = new HashSet<int>(_specialTokens.Values);
            ids = [.. ids.Where(id => !specialTokenIds.Contains(id))];
        }

        // Use the new Decode method
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
    /// Gets the vocabulary size.
    /// </summary>
    public int VocabularySize => _tokenizer.Vocabulary.Count;

    /// <summary>
    /// Gets the vocabulary dictionary.
    /// </summary>
    public IReadOnlyDictionary<string, int> Vocabulary => _tokenizer.Vocabulary;

    /// <summary>
    /// Gets a special token ID by name.
    /// </summary>
    public int? GetSpecialTokenId(string tokenName)
    {
        return _specialTokens.TryGetValue(tokenName, out var id) ? id : null;
    }

    /// <summary>
    /// Gets all special tokens.
    /// </summary>
    public IReadOnlyDictionary<string, int> SpecialTokens => _specialTokens;

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
            inputIds[i] = PadTokenId;
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

    private static void DownloadFile(string url, string destinationPath)
    {
        using var client = new HttpClient();
        client.Timeout = TimeSpan.FromMinutes(10);

        Console.WriteLine($"Downloading {url}...");
        var response = client.GetAsync(url).Result;
        response.EnsureSuccessStatusCode();

        using var fileStream = File.Create(destinationPath);
        response.Content.CopyToAsync(fileStream).Wait();

        Console.WriteLine($"Downloaded to {destinationPath}");
    }

    private static async Task DownloadFileAsync(string url, string destinationPath, CancellationToken cancellationToken = default)
    {
        using var client = new HttpClient();
        client.Timeout = TimeSpan.FromMinutes(10);

        Console.WriteLine($"Downloading {url}...");
        var response = await client.GetAsync(url, cancellationToken);
        response.EnsureSuccessStatusCode();

        using var fileStream = File.Create(destinationPath);
        await response.Content.CopyToAsync(fileStream, cancellationToken);

        Console.WriteLine($"Downloaded to {destinationPath}");
    }
}