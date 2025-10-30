using System.Text.Json;
using System.Text.Json.Serialization;

namespace Yuniko.Software.Qwen3Tokenizer.Tests.PythonComparison;

public record PythonGeneratedTestDataFile(
    [property: JsonPropertyName("model_name")] string ModelName,
    [property: JsonPropertyName("vocabulary_size")] int VocabularySize,
    [property: JsonPropertyName("eos_token_id")] int EosTokenId,
    [property: JsonPropertyName("pad_token_id")] int PadTokenId,
    [property: JsonPropertyName("test_cases")] List<PythonTestCase> TestCases
);

public record PythonTestCase(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("input")] string Input,
    [property: JsonPropertyName("add_eos")] bool AddEos,
    [property: JsonPropertyName("expected")] PythonExpectedResult Expected
);

public record PythonExpectedResult(
    [property: JsonPropertyName("ids")] List<int> Ids,
    [property: JsonPropertyName("tokens")] List<string> Tokens,
    [property: JsonPropertyName("offsets")] List<List<int>> Offsets,
    [property: JsonPropertyName("decoded")] string Decoded,
    [property: JsonPropertyName("decoded_skip_special_tokens")] string DecodedSkipSpecialTokens,
    [property: JsonPropertyName("token_count")] int TokenCount
);

public static class PythonTestDataProvider
{
    private static readonly Dictionary<string, Qwen3Tokenizer> _tokenizerCache = [];
    private static readonly Dictionary<string, PythonGeneratedTestDataFile> _testDataCache = [];
    private static readonly object _tokenizerLock = new();
    private static readonly object _testDataLock = new();

    public static readonly (string ModelName, string TestDataFileName)[] Models =
    [
        ("Qwen/Qwen3-0.6B", "TestData/test_data_qwen3_0.6b.json"),
        ("Qwen/Qwen3-Embedding-0.6B", "TestData/test_data_qwen3_embedding_0.6b.json"),
        ("Qwen/Qwen3-Next-80B-A3B-Instruct", "TestData/test_data_qwen3_next_80b_a3b_instruct.json"),
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", "TestData/test_data_qwen3_vl_30b_a3b_instruct.json"),
        ("Qwen/Qwen3-Reranker-0.6B", "TestData/test_data_qwen3_reranker_0.6b.json"),
    ];

    public static PythonGeneratedTestDataFile GetTestData(string fileName)
    {
        lock (_testDataLock)
        {
            if (_testDataCache.TryGetValue(fileName, out var cached))
            {
                return cached;
            }

            var path = Path.Combine(AppContext.BaseDirectory, fileName);
            var json = File.ReadAllText(path);

#pragma warning disable CA1869 // Cache and reuse 'JsonSerializerOptions' instances
            var testData = JsonSerializer.Deserialize<PythonGeneratedTestDataFile>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
            }) ?? throw new InvalidOperationException($"Failed to load test data from {fileName}");
#pragma warning restore CA1869 // Cache and reuse 'JsonSerializerOptions' instances

            _testDataCache[fileName] = testData;
            return testData;
        }
    }

    public static Qwen3Tokenizer GetTokenizer(string modelName)
    {
        lock (_tokenizerLock)
        {
            if (_tokenizerCache.TryGetValue(modelName, out var cached))
            {
                return cached;
            }

            var tokenizer = Qwen3Tokenizer.FromHuggingFace(modelName);
            _tokenizerCache[modelName] = tokenizer;
            return tokenizer;
        }
    }

    public static IEnumerable<(string ModelName, string TestDataFileName)> GetAvailableModels()
    {
        foreach (var (modelName, fileName) in Models)
        {
            var path = Path.Combine(AppContext.BaseDirectory, fileName);
            if (File.Exists(path))
            {
                yield return (modelName, fileName);
            }
        }
    }
}
