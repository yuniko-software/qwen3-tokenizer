namespace Yuniko.Software.Qwen3Tokenizer.Tests.PythonComparison;

public class BatchOperationTests
{
    [Theory]
    [ClassData(typeof(ModelData))]
    public void EncodeBatch_MatchesIndividualEncodes(string modelName, string testDataFileName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);
        var inputs = testData.TestCases.Take(5).Select(tc => tc.Input).ToArray();
        const bool addEos = true;

        var batchResults = tokenizer.EncodeBatch(inputs, addEos);
        var individualResults = inputs.Select(input => tokenizer.Encode(input, addEos)).ToArray();

        Assert.Equal(inputs.Length, batchResults.Length);
        for (int i = 0; i < inputs.Length; i++)
        {
            Assert.Equal(individualResults[i], batchResults[i]);
        }
    }

    [Theory]
    [ClassData(typeof(ModelData))]
    public void DecodeBatch_MatchesIndividualDecodes(string modelName, string testDataFileName)
    {
        var testData = PythonTestDataProvider.GetTestData(testDataFileName);
        var tokenizer = PythonTestDataProvider.GetTokenizer(modelName);
        var idsBatch = testData.TestCases.Take(5).Select(tc => tc.Expected.Ids.ToArray()).ToArray();
        const bool skipSpecialTokens = true;

        var batchResults = tokenizer.DecodeBatch(idsBatch, skipSpecialTokens);
        var individualResults = idsBatch.Select(ids => tokenizer.Decode(ids, skipSpecialTokens)).ToArray();

        Assert.Equal(idsBatch.Length, batchResults.Length);
        for (int i = 0; i < idsBatch.Length; i++)
        {
            Assert.Equal(individualResults[i], batchResults[i]);
        }
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
}
