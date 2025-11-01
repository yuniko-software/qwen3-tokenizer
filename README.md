# Yuniko.Software.Qwen3Tokenizer

> ⚠️ **Preview Release**: This package is currently in preview as it depends on `Microsoft.ML.Tokenizers` (preview). It's ready for production use, but the preview tag will remain until dependencies stabilize.

Native .NET tokenizer implementation for Qwen3 models. Lightweight byte-pair encoding with HuggingFace integration.

## Features

- **Byte-Pair Encoding (BPE)**: Compatible with Qwen3 model tokenization
- **Special Token Support**: Handles EOS, PAD, and custom added tokens
- **NFC Normalization**: Built-in Unicode normalization
- **HuggingFace Integration**: Load tokenizers directly from HuggingFace models
- **Efficient**: Optimized for performance with minimal dependencies

## Installation

```bash
dotnet add package Yuniko.Software.Qwen3Tokenizer
```

Or via Package Manager:

```powershell
Install-Package Yuniko.Software.Qwen3Tokenizer
```

## Quick Start

```csharp
using Yuniko.Software.Qwen3Tokenizer;

// Load from HuggingFace model
var tokenizer = await Qwen3Tokenizer.FromHuggingFaceAsync(
    "Qwen/Qwen3-Embedding-8B");

// Encode text
var result = tokenizer.Encode("Hello, world!");
Console.WriteLine($"Token IDs: {string.Join(", ", result.Ids)}");

// Decode tokens
var text = tokenizer.Decode(result.Ids);
Console.WriteLine($"Decoded: {text}");
```

## Supported Models

Works with all Qwen3 model variants:
- Qwen3-Embedding models
- Qwen3 General models
- Qwen3-Reranker models
- Qwen3-VL (Vision-Language) models

## Requirements

- .NET 10.0 or later

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/yuniko-software/qwen3-tokenizer/blob/main/LICENSE) file for details.

## Contributing

Contributions are welcome! Please visit the [GitHub repository](https://github.com/yuniko-software/qwen3-tokenizer) for more information.

## Support

For issues, questions, or suggestions, please open an issue on [GitHub](https://github.com/yuniko-software/qwen3-tokenizer/issues).
