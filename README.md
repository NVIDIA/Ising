# NVIDIA Ising

**Open models, tools, and cookbooks for quantum computing — from device calibration to error correction.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Models-FFD21E?logo=huggingface)](https://huggingface.co/collections/nvidia/nvidia-ising)

> This repository is the central landing page for the NVIDIA Ising family of quantum computing tools, models, and cookbooks. For code, see the repositories below.

---

## Table of Contents

- [Models](#models)
- [Datasets](#datasets)
- [Repositories](#repositories)
- [Cookbooks](#cookbooks)
- [Issues & Support](#issues--support)
- [Contributing](#contributing)
- [License](#license)

---

## Models

The Ising model family is available on Hugging Face:

| Model | Description | Link |
|-------|-------------|------|
| **Ising-Calibration-1-35B-A3B** | Vision-language model for quantum device calibration | [HuggingFace](https://huggingface.co/nvidia/Ising-Calibration-1-35B-A3B) |
| **Ising-Decoder-SurfaceCode-1-1.79M** | AI predecoder for surface code quantum error correction (1.79M params) | [HuggingFace](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-1.79M) |
| **Ising-Decoder-SurfaceCode-1-712k** | AI predecoder for surface code quantum error correction (712k params) | [HuggingFace](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-712k) |

Browse the full collection: [NVIDIA Ising Collection on Hugging Face](https://huggingface.co/collections/nvidia/nvidia-ising)

---

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **QCalEval** | Evaluation dataset for quantum calibration agents | [HuggingFace](https://huggingface.co/datasets/nvidia/QCalEval) |

---

## Repositories

| Repository | Description |
|------------|-------------|
| [ising-calibration](https://github.com/NVIDIA/ising-calibration) | Cookbook and Quantum Calibration Agent Blueprint — AI-powered agent for discovering, executing, and analyzing quantum calibration experiments |
| [Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding) | AI predecoder for quantum error correction |

---

## Cookbooks

Practical guides and tutorials organized by repository:

### [ising-calibration](https://github.com/NVIDIA/ising-calibration)

| Guide | Description |
|-------|-------------|
| [Quick Start](https://github.com/NVIDIA/ising-calibration/tree/main/cookbook/quick-start) | Installation and first run |
| [Tutorial](https://github.com/NVIDIA/ising-calibration/blob/main/cookbook/tutorial.md) | End-to-end walkthrough: experiments, workflows, and the knowledge base |
| [User Guide](https://github.com/NVIDIA/ising-calibration/tree/main/cookbook/user-guide) | CLI reference, experiment storage, workflows, and knowledge base |
| [Developer Guide](https://github.com/NVIDIA/ising-calibration/tree/main/cookbook/developer-guide) | Architecture, contributing, and extending the agent |
| [vLLM Local Model Setup](https://github.com/NVIDIA/ising-calibration/blob/main/cookbook/vllm-local-model-setup.md) | Run the VLM locally with vLLM |

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

**NVIDIA Ising** — Open quantum computing tools for the next generation of quantum hardware.
