using System.Text.RegularExpressions;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Token constants and collections for Qwen3 models.
/// Contains added tokens, special token IDs, and other token-related constants.
/// </summary>
public static partial class Qwen3Tokens
{
    // Token IDs

    /// <summary>
    /// The <c>&lt;|endoftext|&gt;</c> token ID (151643).
    /// This is used as the PAD token in Qwen3 models.
    /// </summary>
    public const int EndOfTextTokenId = 151643;
    /// <summary>
    /// The <c>&lt;|im_start|&gt;</c> token ID (151644).
    /// Marks the start of an instant message or chat turn.
    /// </summary>
    public const int ImStartTokenId = 151644;
    /// <summary>
    /// The <c>&lt;|im_end|&gt;</c> token ID (151645).
    /// Marks the end of an instant message or chat turn.
    /// This is used as the EOS (End-Of-Sequence) token in Qwen3 models.
    /// </summary>
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
    public const int ToolCallTokenId = 151657;
    public const int ToolCallEndTokenId = 151658;
    public const int FimPrefixTokenId = 151659;
    public const int FimMiddleTokenId = 151660;
    public const int FimSuffixTokenId = 151661;
    public const int FimPadTokenId = 151662;
    public const int RepoNameTokenId = 151663;
    public const int FileSepTokenId = 151664;
    public const int ToolResponseTokenId = 151665;
    public const int ToolResponseEndTokenId = 151666;
    public const int ThinkTokenId = 151667;
    public const int ThinkEndTokenId = 151668;

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
    public const string ToolCall = "<tool_call>";
    public const string ToolCallEnd = "</tool_call>";
    public const string FimPrefix = "<|fim_prefix|>";
    public const string FimMiddle = "<|fim_middle|>";
    public const string FimSuffix = "<|fim_suffix|>";
    public const string FimPad = "<|fim_pad|>";
    public const string RepoName = "<|repo_name|>";
    public const string FileSep = "<|file_sep|>";
    public const string ToolResponse = "<tool_response>";
    public const string ToolResponseEnd = "</tool_response>";
    public const string Think = "<think>";
    public const string ThinkEnd = "</think>";

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
    /// All added tokens dictionary for Qwen3 models (all 26 tokens).
    /// These tokens are treated as atomic during pre-tokenization.
    /// In HuggingFace terminology, these are "added_tokens" - some are marked as "special": true, others as "special": false.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, int> AddedTokens = new Dictionary<string, int>
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
        { VideoPad, VideoPadTokenId },
        { ToolCall, ToolCallTokenId },
        { ToolCallEnd, ToolCallEndTokenId },
        { FimPrefix, FimPrefixTokenId },
        { FimMiddle, FimMiddleTokenId },
        { FimSuffix, FimSuffixTokenId },
        { FimPad, FimPadTokenId },
        { RepoName, RepoNameTokenId },
        { FileSep, FileSepTokenId },
        { ToolResponse, ToolResponseTokenId },
        { ToolResponseEnd, ToolResponseEndTokenId },
        { Think, ThinkTokenId },
        { ThinkEnd, ThinkEndTokenId }
    };

    /// <summary>
    /// Token IDs marked as "special": true in HuggingFace (14 tokens: 151643-151656).
    /// These are skipped during decoding when skip_special_tokens=True.
    /// The remaining added tokens (151657-151668) have "special": false and are NOT skipped.
    /// </summary>
    public static readonly IReadOnlySet<int> SpecialTokenIds = new HashSet<int>
    {
        EndOfTextTokenId,      // 151643
        ImStartTokenId,        // 151644
        ImEndTokenId,          // 151645
        ObjectRefStartTokenId, // 151646
        ObjectRefEndTokenId,   // 151647
        BoxStartTokenId,       // 151648
        BoxEndTokenId,         // 151649
        QuadStartTokenId,      // 151650
        QuadEndTokenId,        // 151651
        VisionStartTokenId,    // 151652
        VisionEndTokenId,      // 151653
        VisionPadTokenId,      // 151654
        ImagePadTokenId,       // 151655
        VideoPadTokenId        // 151656
    };
}