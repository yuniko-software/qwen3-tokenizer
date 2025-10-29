namespace Yuniko.Software.Qwen3Tokenizer.Tests.PythonComparison;

public class PythonComparisonTests
{
    [Theory]
    [ClassData(typeof(ModelData))]
    public void VocabularySize_MatchesPython(string modelName, string testDataFileName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);

        Assert.Equal(testData.VocabularySize, tokenizer.VocabularySize);
    }

    [Theory]
    [ClassData(typeof(TestCaseData))]
    public void Encode_ProducesExpectedTokenIds(string modelName, string testDataFileName, string testCaseName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);
        var testCase = testData.TestCases.First(tc => string.Equals(tc.Name, testCaseName, StringComparison.InvariantCulture));

        var actual = tokenizer.Encode(testCase.Input, testCase.AddEos);

        Assert.Equal([.. testCase.Expected.Ids], actual);
    }

    [Theory]
    [ClassData(typeof(TestCaseData))]
    public void EncodeDetailed_ProducesExpectedTokenIds(string modelName, string testDataFileName, string testCaseName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);
        var testCase = testData.TestCases.First(tc => string.Equals(tc.Name, testCaseName, StringComparison.InvariantCulture));

        var actual = tokenizer.EncodeDetailed(testCase.Input, testCase.AddEos);

        Assert.Equal([.. testCase.Expected.Ids], actual.Ids);
    }

    [Theory]
    [ClassData(typeof(TestCaseData))]
    public void EncodeDetailed_ProducesExpectedTokenStrings(string modelName, string testDataFileName, string testCaseName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);
        var testCase = testData.TestCases.First(tc => string.Equals(tc.Name, testCaseName, StringComparison.InvariantCulture));

        var actual = tokenizer.EncodeDetailed(testCase.Input, testCase.AddEos);

        Assert.Equal(testCase.Expected.Tokens.ToArray(), actual.Tokens);
    }

    [Theory]
    [ClassData(typeof(TestCaseData))]
    public void CountTokens_ProducesExpectedCount(string modelName, string testDataFileName, string testCaseName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);
        var testCase = testData.TestCases.First(tc => string.Equals(tc.Name, testCaseName, StringComparison.InvariantCulture));

        var actual = tokenizer.CountTokens(testCase.Input, testCase.AddEos);

        Assert.Equal(testCase.Expected.TokenCount, actual);
    }

    [Theory]
    [ClassData(typeof(TestCaseData))]
    public void Decode_WithSpecialTokens_ProducesExpectedText(string modelName, string testDataFileName, string testCaseName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);
        var testCase = testData.TestCases.First(tc => string.Equals(tc.Name, testCaseName, StringComparison.InvariantCulture));

        var actual = tokenizer.Decode([.. testCase.Expected.Ids], skipSpecialTokens: false);

        Assert.Equal(testCase.Expected.Decoded, actual);
    }

    [Theory]
    [ClassData(typeof(TestCaseData))]
    public void Decode_SkipSpecialTokens_ProducesExpectedText(string modelName, string testDataFileName, string testCaseName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);
        var testCase = testData.TestCases.First(tc => string.Equals(tc.Name, testCaseName, StringComparison.InvariantCulture));

        var actual = tokenizer.Decode([.. testCase.Expected.Ids], skipSpecialTokens: true);

        Assert.Equal(testCase.Expected.DecodedSkipSpecialTokens, actual);
    }

    public class ModelData : TheoryData<string, string>
    {
        public ModelData()
        {
            foreach (var (modelName, fileName) in PythonTestDataProvider.GetAvailableModels())
            {
                Add(modelName, fileName);
            }
        }
    }

    public class TestCaseData : TheoryData<string, string, string>
    {
        public TestCaseData()
        {
            foreach (var (modelName, fileName) in PythonTestDataProvider.GetAvailableModels())
            {
                var testData = PythonTestDataProvider.GetTestData(fileName);
                foreach (var testCase in testData.TestCases)
                {
                    Add(modelName, fileName, testCase.Name);
                }
            }
        }
    }
}
