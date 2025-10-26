using System.Text.RegularExpressions;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Special tokens and constants for Qwen3 Embedding models.
/// </summary>
public static partial class Qwen3EmbeddingModelSpecialTokens
{
    // Token IDs
    public const int EndOfTextTokenId = 151643;
    public const int ImStartTokenId = 151644;
    public const int ImEndTokenId = 151645;
    public const int ObjectRefStartTokenId = 151646;
    public const int ObjectRefEndTokenId = 151647;
    public const int BoxStartTokenId = 151648;
    public const int BoxEndTokenId = 151649;
    public const int QuadStartTokenId = 151650;
    public const int QuadEndTokenId = 151651;
    public const int VisionStartTokenId = 151652;
    public const int VisionEndTokenId = 151653;
    public const int VisionPadTokenId = 151654;
    public const int ImagePadTokenId = 151655;
    public const int VideoPadTokenId = 151656;

    // Token strings
    public const string EndOfText = "<|endoftext|>";
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";
    public const string ObjectRefStart = "<|object_ref_start|>";
    public const string ObjectRefEnd = "<|object_ref_end|>";
    public const string BoxStart = "<|box_start|>";
    public const string BoxEnd = "<|box_end|>";
    public const string QuadStart = "<|quad_start|>";
    public const string QuadEnd = "<|quad_end|>";
    public const string VisionStart = "<|vision_start|>";
    public const string VisionEnd = "<|vision_end|>";
    public const string VisionPad = "<|vision_pad|>";
    public const string ImagePad = "<|image_pad|>";
    public const string VideoPad = "<|video_pad|>";

    /// <summary>
    /// Default pre-tokenizer regex pattern for Qwen3 tokenizer.
    /// </summary>
    public const string PreTokenizerPattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// <summary>
    /// Precompiled regex for the default pre-tokenizer pattern.
    /// </summary>
    [GeneratedRegex(PreTokenizerPattern, RegexOptions.Compiled)]
    public static partial Regex GetPreTokenizerRegex();

    /// <summary>
    /// Default special tokens dictionary for Qwen3 Embedding models.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, int> SpecialTokens = new Dictionary<string, int>
    {
        { EndOfText, EndOfTextTokenId },
        { ImStart, ImStartTokenId },
        { ImEnd, ImEndTokenId },
        { ObjectRefStart, ObjectRefStartTokenId },
        { ObjectRefEnd, ObjectRefEndTokenId },
        { BoxStart, BoxStartTokenId },
        { BoxEnd, BoxEndTokenId },
        { QuadStart, QuadStartTokenId },
        { QuadEnd, QuadEndTokenId },
        { VisionStart, VisionStartTokenId },
        { VisionEnd, VisionEndTokenId },
        { VisionPad, VisionPadTokenId },
        { ImagePad, ImagePadTokenId },
        { VideoPad, VideoPadTokenId }
    };
}