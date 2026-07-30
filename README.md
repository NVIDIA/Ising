# NVIDIA Ising

**Open models, tools, and cookbooks for quantum computing — from device calibration to error correction.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Models-FFD21E?logo=huggingface)](https://huggingface.co/collections/nvidia/nvidia-ising)

> This repository is the central landing page for the NVIDIA Ising family of quantum computing tools, models, and cookbooks. For code, see the repositories below.

---

## Table of Contents

- [Models](#models)
- [NIMs](#nims)
- [Model API Endpoints](#model-api-endpoint)
- [Datasets](#datasets)
- [Repositories](#repositories)
- [Cookbooks](#cookbooks)
- [Issues & Support](#issues--support)
- [License](#license)

---

## Models

The Ising model family is available on Hugging Face:

| Model | Description |
|-------|-------------|
| [**Ising-Calibration-1.5-31B-BF16**](https://huggingface.co/nvidia/Ising-Calibration-1.5-31B-BF16) | 31B vision-language model for quantum calibration plot understanding (BF16) |
| [**Ising-Calibration-1.5-31B-NVFP4**](https://huggingface.co/nvidia/Ising-Calibration-1.5-31B-NVFP4) | 31B vision-language model for quantum calibration plot understanding (NVFP4) |
| [**Ising-Decoder-ColorCode-1-Fast**](https://huggingface.co/nvidia/Ising-Decoder-ColorCode-1-Fast) | AI predecoder for color code quantum error correction (2.94M params) |
| [**Ising-Decoder-SurfaceCode-1-Accurate**](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Accurate) | AI predecoder for surface code quantum error correction (1.79M params) | 
| [**Ising-Decoder-SurfaceCode-1-Fast**](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast) | AI predecoder for surface code quantum error correction (0.91M params) |
| [**Ising-Calibration-1-35B-A3B**](https://huggingface.co/nvidia/Ising-Calibration-1-35B-A3B) | Vision-language model for quantum device calibration |

Browse the full collection: [NVIDIA Ising Collection on Hugging Face](https://huggingface.co/collections/nvidia/nvidia-ising)

---
## NIMs

| NIM | Description |
|------------|-------------|
| [**Ising-Calibration-1.5-31B**](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/ising-calibration-1-5-31b) | Ready-to-deploy NIM for Ising-Calibration-1.5 |

---

## Model API Endpoint

| API | Description |
|------------|-------------|
| [**Ising-Calibration-1.5-31B**](https://build.nvidia.com/nvidia/ising-calibration-1.5-31b) | Hosted API endpoint for Ising-Calibration-1.5 |


## Datasets

| Dataset | Description |
|---------|-------------|
| [**QCalEval**](https://huggingface.co/datasets/nvidia/QCalEval) | Evaluation dataset for quantum calibration agents |

---

## Repositories

| Repository | Description |
|------------|-------------|
| [Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding) | Training Samples for AI Decoding |
| [Quantum Calibration Agent Blueprint](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint/) | Agentic workflow blueprint for Quantum Comptuer Calibration VLMs |
| [QCalEval](https://github.com/NVIDIA/QCalEval) | Open benchmark for assessing VLM capabilities on quantum calibration experiment analysis |

---

## Cookbooks

Practical guides and tutorials organized by repository:

### [Ising-Decoding](https://github.com/NVIDIA/ising-decoding)

| Guide | Description |
|-------|-------------|
|[Ising-Decoder-Tutorial](https://github.com/NVIDIA/Ising-Decoding/blob/main/cookbook/predecoder.ipynb) | Training, Optimized Inference, Quantizing an AI Pre-decoder |
|[Quantum Calibration Agent Documentation](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint/tree/main/docs) | Documentation to get started with an agentic workflow for quantum computer calibration |
---

## Issues & Support

This repository is the central place to file issues across the NVIDIA Ising family:

- **Bug reports** — [Open an issue](https://github.com/NVIDIA/ising/issues/new?template=bug.yml)
- **Feature requests** — [Open an issue](https://github.com/NVIDIA/ising/issues/new?template=feature.yml)
- **Questions & Discussion** — [Start a discussion](https://github.com/NVIDIA/ising/discussions)

For security vulnerabilities, do not file a public issue — see [SECURITY.md](SECURITY.md).

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

**Open models, tools, and cookbooks for quantum computing — from device calibration to error correction.**
