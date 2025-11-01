using System.Text;

namespace Yuniko.Software.Qwen3Tokenizer.Tests;

public class NfcNormalizerTests
{
    private readonly NfcNormalizer _normalizer = new();

    [Fact]
    public void Normalize_EmptyString_ReturnsEmptyString()
    {
        var result = _normalizer.Normalize("");

        Assert.Equal("", result);
    }

    [Fact]
    public void Normalize_NullString_ReturnsNull()
    {
        var result = _normalizer.Normalize(null!);

        Assert.Null(result);
    }

    [Fact]
    public void Normalize_EmptySpan_ReturnsEmptyString()
    {
        var result = _normalizer.Normalize([]);

        Assert.Equal("", result);
    }

    [Theory]
    [InlineData("e\u0301", "é")]
    [InlineData("a\u0308", "ä")]
    [InlineData("n\u0303", "ñ")]
    [InlineData("o\u0302", "ô")]
    public void Normalize_CombiningCharacters_ProducesPrecomposedForm(string input, string expected)
    {
        var result = _normalizer.Normalize(input);

        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("café")]
    [InlineData("naïve")]
    [InlineData("Hello")]
    [InlineData("你好")]
    [InlineData("こんにちは")]
    public void Normalize_AlreadyNfc_ReturnsUnchanged(string input)
    {
        var result = _normalizer.Normalize(input);

        Assert.Equal(input, result);
    }

    [Fact]
    public void Normalize_MixedContent_NormalizesCorrectly()
    {
        const string input = "cafe\u0301 123";
        const string expected = "café 123";

        var result = _normalizer.Normalize(input);

        Assert.Equal(expected, result);
    }

    [Fact]
    public void Normalize_Span_ProducesCorrectNfcForm()
    {
        var input = "e\u0301".AsSpan();
        const string expected = "é";

        var result = _normalizer.Normalize(input);

        Assert.Equal(expected, result);
    }

    [Fact]
    public void Normalize_VerifiesNfcFormProperty()
    {
        const string input = "e\u0301";

        var result = _normalizer.Normalize(input);

        Assert.True(result.IsNormalized(NormalizationForm.FormC));
    }

    [Theory]
    [InlineData("Å")]
    [InlineData("ﬁ")]
    public void Normalize_SpecialUnicodeCharacters_HandlesCorrectly(string input)
    {
        var result = _normalizer.Normalize(input);

        Assert.True(result.IsNormalized(NormalizationForm.FormC));
    }

    [Fact]
    public void Normalize_LongString_WorksCorrectly()
    {
        var input = string.Concat(Enumerable.Repeat("cafe\u0301 ", 1000));

        var result = _normalizer.Normalize(input);

        Assert.Contains("café", result, StringComparison.InvariantCulture);
        Assert.True(result.IsNormalized(NormalizationForm.FormC));
    }
}
