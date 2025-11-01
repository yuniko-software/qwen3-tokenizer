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
    /// <summary>
    /// The <c>&lt;|object_ref_start|&gt;</c> token ID (151646).
    /// </summary>
    public const int ObjectRefStartTokenId = 151646;
    /// <summary>
    /// The <c>&lt;|object_ref_end|&gt;</c> token ID (151647).
    /// </summary>
    public const int ObjectRefEndTokenId = 151647;
    /// <summary>
    /// The <c>&lt;|box_start|&gt;</c> token ID (151648).
    /// </summary>
    public const int BoxStartTokenId = 151648;
    /// <summary>
    /// The <c>&lt;|box_end|&gt;</c> token ID (151649).
    /// </summary>
    public const int BoxEndTokenId = 151649;
    /// <summary>
    /// The <c>&lt;|quad_start|&gt;</c> token ID (151650).
    /// </summary>
    public const int QuadStartTokenId = 151650;
    /// <summary>
    /// The <c>&lt;|quad_end|&gt;</c> token ID (151651).
    /// </summary>
    public const int QuadEndTokenId = 151651;
    /// <summary>
    /// The <c>&lt;|vision_start|&gt;</c> token ID (151652).
    /// </summary>
    public const int VisionStartTokenId = 151652;
    /// <summary>
    /// The <c>&lt;|vision_end|&gt;</c> token ID (151653).
    /// </summary>
    public const int VisionEndTokenId = 151653;
    /// <summary>
    /// The <c>&lt;|vision_pad|&gt;</c> token ID (151654).
    /// </summary>
    public const int VisionPadTokenId = 151654;
    /// <summary>
    /// The <c>&lt;|image_pad|&gt;</c> token ID (151655).
    /// </summary>
    public const int ImagePadTokenId = 151655;
    /// <summary>
    /// The <c>&lt;|video_pad|&gt;</c> token ID (151656).
    /// </summary>
    public const int VideoPadTokenId = 151656;
    /// <summary>
    /// The <c>&lt;tool_call&gt;</c> token ID (151657).
    /// </summary>
    public const int ToolCallTokenId = 151657;
    /// <summary>
    /// The <c>&lt;/tool_call&gt;</c> token ID (151658).
    /// </summary>
    public const int ToolCallEndTokenId = 151658;
    /// <summary>
    /// The <c>&lt;|fim_prefix|&gt;</c> token ID (151659).
    /// </summary>
    public const int FimPrefixTokenId = 151659;
    /// <summary>
    /// The <c>&lt;|fim_middle|&gt;</c> token ID (151660).
    /// </summary>
    public const int FimMiddleTokenId = 151660;
    /// <summary>
    /// The <c>&lt;|fim_suffix|&gt;</c> token ID (151661).
    /// </summary>
    public const int FimSuffixTokenId = 151661;
    /// <summary>
    /// The <c>&lt;|fim_pad|&gt;</c> token ID (151662).
    /// </summary>
    public const int FimPadTokenId = 151662;
    /// <summary>
    /// The <c>&lt;|repo_name|&gt;</c> token ID (151663).
    /// </summary>
    public const int RepoNameTokenId = 151663;
    /// <summary>
    /// The <c>&lt;|file_sep|&gt;</c> token ID (151664).
    /// </summary>
    public const int FileSepTokenId = 151664;
    /// <summary>
    /// The <c>&lt;tool_response&gt;</c> token ID (151665).
    /// </summary>
    public const int ToolResponseTokenId = 151665;
    /// <summary>
    /// The <c>&lt;/tool_response&gt;</c> token ID (151666).
    /// </summary>
    public const int ToolResponseEndTokenId = 151666;
    /// <summary>
    /// The <c>&lt;think&gt;</c> token ID (151667).
    /// </summary>
    public const int ThinkTokenId = 151667;
    /// <summary>
    /// The <c>&lt;/think&gt;</c> token ID (151668).
    /// </summary>
    public const int ThinkEndTokenId = 151668;

    // Token strings
    /// <summary>
    /// The <c>&lt;|endoftext|&gt;</c> token string.
    /// </summary>
    public const string EndOfText = "<|endoftext|>";
    /// <summary>
    /// The <c>&lt;|im_start|&gt;</c> token string.
    /// </summary>
    public const string ImStart = "<|im_start|>";
    /// <summary>
    /// The <c>&lt;|im_end|&gt;</c> token string.
    /// </summary>
    public const string ImEnd = "<|im_end|>";
    /// <summary>
    /// The <c>&lt;|object_ref_start|&gt;</c> token string.
    /// </summary>
    public const string ObjectRefStart = "<|object_ref_start|>";
    /// <summary>
    /// The <c>&lt;|object_ref_end|&gt;</c> token string.
    /// </summary>
    public const string ObjectRefEnd = "<|object_ref_end|>";
    /// <summary>
    /// The <c>&lt;|box_start|&gt;</c> token string.
    /// </summary>
    public const string BoxStart = "<|box_start|>";
    /// <summary>
    /// The <c>&lt;|box_end|&gt;</c> token string.
    /// </summary>
    public const string BoxEnd = "<|box_end|>";
    /// <summary>
    /// The <c>&lt;|quad_start|&gt;</c> token string.
    /// </summary>
    public const string QuadStart = "<|quad_start|>";
    /// <summary>
    /// The <c>&lt;|quad_end|&gt;</c> token string.
    /// </summary>
    public const string QuadEnd = "<|quad_end|>";
    /// <summary>
    /// The <c>&lt;|vision_start|&gt;</c> token string.
    /// </summary>
    public const string VisionStart = "<|vision_start|>";
    /// <summary>
    /// The <c>&lt;|vision_end|&gt;</c> token string.
    /// </summary>
    public const string VisionEnd = "<|vision_end|>";
    /// <summary>
    /// The <c>&lt;|vision_pad|&gt;</c> token string.
    /// </summary>
    public const string VisionPad = "<|vision_pad|>";
    /// <summary>
    /// The <c>&lt;|image_pad|&gt;</c> token string.
    /// </summary>
    public const string ImagePad = "<|image_pad|>";
    /// <summary>
    /// The <c>&lt;|video_pad|&gt;</c> token string.
    /// </summary>
    public const string VideoPad = "<|video_pad|>";
    /// <summary>
    /// The <c>&lt;tool_call&gt;</c> token string.
    /// </summary>
    public const string ToolCall = "<tool_call>";
    /// <summary>
    /// The <c>&lt;/tool_call&gt;</c> token string.
    /// </summary>
    public const string ToolCallEnd = "</tool_call>";
    /// <summary>
    /// The <c>&lt;|fim_prefix|&gt;</c> token string.
    /// </summary>
    public const string FimPrefix = "<|fim_prefix|>";
    /// <summary>
    /// The <c>&lt;|fim_middle|&gt;</c> token string.
    /// </summary>
    public const string FimMiddle = "<|fim_middle|>";
    /// <summary>
    /// The <c>&lt;|fim_suffix|&gt;</c> token string.
    /// </summary>
    public const string FimSuffix = "<|fim_suffix|>";
    /// <summary>
    /// The <c>&lt;|fim_pad|&gt;</c> token string.
    /// </summary>
    public const string FimPad = "<|fim_pad|>";
    /// <summary>
    /// The <c>&lt;|repo_name|&gt;</c> token string.
    /// </summary>
    public const string RepoName = "<|repo_name|>";
    /// <summary>
    /// The <c>&lt;|file_sep|&gt;</c> token string.
    /// </summary>
    public const string FileSep = "<|file_sep|>";
    /// <summary>
    /// The <c>&lt;tool_response&gt;</c> token string.
    /// </summary>
    public const string ToolResponse = "<tool_response>";
    /// <summary>
    /// The <c>&lt;/tool_response&gt;</c> token string.
    /// </summary>
    public const string ToolResponseEnd = "</tool_response>";
    /// <summary>
    /// The <c>&lt;think&gt;</c> token string.
    /// </summary>
    public const string Think = "<think>";
    /// <summary>
    /// The <c>&lt;/think&gt;</c> token string.
    /// </summary>
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
        { ThinkEnd, ThinkEndTokenId },
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
        VideoPadTokenId,        // 151656
    };
}
