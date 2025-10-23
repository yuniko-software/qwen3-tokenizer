using Microsoft.ML.Tokenizers;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Tokenizer for Qwen3-Embedding models.
/// Based on byte-level BPE (Byte-Pair Encoding) similar to GPT-2.
/// </summary>
public class Qwen3Tokenizer
{
    private readonly Tokenizer _tokenizer;
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

        var bpe = new Bpe(vocabPath, mergesPath);
        _tokenizer = new Tokenizer(bpe);

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
    /// Encodes text into token IDs.
    /// </summary>
    /// <param name="text">Input text to tokenize</param>
    /// <param name="addEos">Whether to add EOS token at the end</param>
    /// <returns>Array of token IDs</returns>
    public int[] Encode(string text, bool addEos = true)
    {
        var result = _tokenizer.Encode(text);
        int[] ids = [.. result.Ids];

        if (addEos)
        {
            return [.. ids, EosTokenId];
        }

        return ids;
    }

    /// <summary>
    /// Encodes text and returns detailed encoding information.
    /// </summary>
    public EncodingResult EncodeDetailed(string text, bool addEos = true)
    {
        var result = _tokenizer.Encode(text);
        int[] ids = [.. result.Ids];
        string[] tokens = [.. result.Tokens];

        if (addEos)
        {
            ids = [.. ids, EosTokenId];
            tokens = [.. tokens, "<|endoftext|>"];
        }

        return new EncodingResult(ids, tokens, [.. result.Offsets]);
    }

    /// <summary>
    /// Encodes a batch of texts.
    /// </summary>
    public int[][] EncodeBatch(IEnumerable<string> texts, bool addEos = true)
    {
        return [.. texts.Select(text => Encode(text, addEos))];
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

        return _tokenizer.Decode(ids);
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
    public int VocabularySize
    {
        get
        {
            if (_tokenizer.Model is Bpe bpe)
            {
                return bpe.GetVocabSize();
            }
            return 0;
        }
    }

    /// <summary>
    /// Gets a special token ID by name.
    /// </summary>
    public int? GetSpecialTokenId(string tokenName)
    {
        return _specialTokens.TryGetValue(tokenName, out var id) ? id : null;
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

        if (ids.Length > maxLength)
        {
            ids = [.. ids.Take(maxLength)];
        }

        var inputIds = new long[maxLength];
        var attentionMask = new long[maxLength];

        for (int i = 0; i < ids.Length; i++)
        {
            inputIds[i] = ids[i];
            attentionMask[i] = 1;
        }

        for (int i = ids.Length; i < maxLength; i++)
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
}
