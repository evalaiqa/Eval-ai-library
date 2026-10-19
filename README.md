# Eval AI Library

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/eval-ai-library)](https://pypi.org/project/eval-ai-library/)

> Based on [firstlinesoftware/eval-ai-library](https://github.com/firstlinesoftware/eval-ai-library). This is an independently maintained version with additional features and PyPI distribution.

Comprehensive AI model evaluation framework for RAG systems and AI agents. Supports 35+ evaluation metrics, 30+ LLM providers (OpenAI, Anthropic, Google, Azure, AWS Bedrock, Vertex AI, Mistral, Groq, xAI, DeepSeek, Cohere, OpenRouter, Together, Fireworks, Ollama, MLX, and more via LiteLLM), built-in test data generation from documents, an interactive web dashboard for visualization and analysis, and a first-class tracing subsystem with framework integrations for LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, Semantic Kernel, Claude Agent SDK, smolagents, phidata, and OpenAI Assistants (plus an OpenTelemetry exporter). Implements advanced techniques including G-Eval probability-weighted scoring and Temperature-Controlled Verdict Aggregation via Generalized Power Mean.

## Installation

```bash
pip install eval-ai-library
```

Full version with document parsing and OCR support:

```bash
pip install eval-ai-library[full]
```

Lite version (core evaluation only):

```bash
pip install eval-ai-library[lite]
```

## Quick Start

```python
from eval_lib import EvalAI

evaluator = EvalAI(model="gpt-4o")

result = evaluator.evaluate(
    input="What is Python?",
    actual_output="Python is a programming language.",
    expected_output="Python is a high-level programming language.",
    metrics=["answer_relevancy", "faithfulness"]
)

print(result.score)
```

## Documentation

Full documentation is available at [library.eval-ai.com](https://library.eval-ai.com).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:
```bibtex
@software{eval_ai_library,
  author = {Meshkov, Aleksandr},
  title = {Eval AI Library: Comprehensive AI Model Evaluation Framework},
  year = {2025},
  url = {https://github.com/meshkovQA/Eval-ai-library.git}
}
```

### References

This library implements techniques from:
```bibtex
@misc{meshkov2026adaptiverigoraievaluation,
      title={Adaptive Rigor in AI System Evaluation using Temperature-Controlled Verdict Aggregation via Generalized Power Mean}, 
      author={Aleksandr Meshkov},
      year={2026},
      eprint={2604.08595},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.08595}, 
}
```
```bibtex
@inproceedings{liu2023geval,
  title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
  author={Liu, Yang and Iter, Dan and Xu, Yichong and Wang, Shuohang and Xu, Ruochen and Zhu, Chenguang},
  booktitle={Proceedings of EMNLP},
  year={2023}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/meshkovQA/Eval-ai-library/issues)
- Documentation: [library.eval-ai.com](https://library.eval-ai.com)
