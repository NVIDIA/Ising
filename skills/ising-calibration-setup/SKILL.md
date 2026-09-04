---
name: ising-calibration-setup
description: >-
  Get a verified working connection to the NVIDIA Ising Calibration model,
  either through the hosted build.nvidia.com API or a self-hosted NIM
  container (on your own hardware or on a cloud GPU instance). Covers auth,
  keys, and connectivity verification only, not model usage. Use when the
  user says "set up Ising calibration", "connect to the Ising calibration
  model", "get an API key for Ising Calibration", "run the Ising Calibration
  NIM", "self-host Ising Calibration", or "/ising-calibration-setup".
license: Apache-2.0
metadata:
  author: "NVIDIA"
  version: "0.1.0"
  github-url: "https://github.com/NVIDIA/Ising"
  tags: "nvidia ising quantum calibration nim api setup"
---

## Purpose

Get a user from zero to a **verified working connection** to the NVIDIA
Ising Calibration model (Ising-Calibration-1.5-31B), via either of two paths:

1. **Hosted Model API** on build.nvidia.com -- fastest path, no GPU required.
2. **Self-hosted NIM** container -- for private/offline/high-volume use, requires
   a supported NVIDIA GPU.

This skill is plumbing-scoped: getting a key, getting a `200` back, and
confirming that connection actually works. It stops there -- it does not
teach how to use the model (prompting it for calibration analysis, building
an agent workflow around it, etc.). See "Next steps" below for where that
guidance lives.

## Instructions

Ask the user which path they want (or infer it: "I don't have a GPU" ->
hosted API; "I need this on-prem / air-gapped / high volume" -> self-hosted
NIM). Then follow the matching workflow below top-to-bottom. Full command
sequences live in `references/`; this file only handles routing, prerequisites,
and the connectivity checklist.

### When to use

- "How do I connect to Ising Calibration?"
- "Set up the Ising Calibration model"
- "Get an API key for build.nvidia.com / Ising"
- "Run / self-host the Ising Calibration NIM" (own hardware or a cloud GPU instance)
- "My Ising Calibration API calls are failing, check my setup"

### Prerequisites (both paths)

- An NVIDIA account (create one at build.nvidia.com or ngc.nvidia.com if the
  user does not have one).
- `curl` available locally for connectivity checks.
- `jq` available locally for the hello-world multimodal request (Path 1).

## Path 1: Hosted Model API (build.nvidia.com)

No GPU needed; requests are billed/rate-limited against the hosted endpoint.

1. Sign in at `https://build.nvidia.com`.
2. Generate a personal API key: profile menu -> Settings -> API keys ->
   **Generate API Key** (or go directly to `https://build.nvidia.com/settings/api-keys`).
   You will be asked to name the key and set an expiration. The key starts
   with `nvapi-` and is shown only once -- store it immediately without
   letting it land in shell history (see
   [`references/hosted-api-quickstart.md`](references/hosted-api-quickstart.md)
   for a `read -rs`-based method).
3. Confirm the model page for reference: `https://build.nvidia.com/nvidia/ising-calibration-1.5-31b`.
4. Follow [`references/hosted-api-quickstart.md`](references/hosted-api-quickstart.md)
   for the full request format, suggested parameters, and a working `curl`
   example.

Governed by the NVIDIA API Trial Terms of Service.

## Path 2: Self-hosted NIM

Requires an NVIDIA Ada Lovelace, Hopper, or Blackwell GPU with enough memory
for the chosen precision (see the reference doc for the tested minimums).

1. Generate a separate **NGC personal key** -- this is NOT the same credential
   as the `nvapi-` key from Path 1. Go to `https://org.ngc.nvidia.com/setup/personal-keys`
   (or: account icon -> Setup -> Generate Personal Key) and store it without
   letting it land in shell history (see
   [`references/self-hosted-nim-quickstart.md`](references/self-hosted-nim-quickstart.md)
   for a `read -rs`-based method).
2. Log Docker in to the NVIDIA container registry:
   `echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin`
3. Pull and run the container, then confirm it is serving. Full command,
   GPU/memory requirements, health check, and example request are in
   [`references/self-hosted-nim-quickstart.md`](references/self-hosted-nim-quickstart.md).

## Connectivity checklist

Run this regardless of path to confirm the setup actually works, and to tell
an auth failure apart from a network/model-not-ready failure. Full checklist
with copy-pasteable commands: [`references/connectivity-checklist.md`](references/connectivity-checklist.md).

Summary:

1. Confirm the right key env var is set and non-empty (`NVIDIA_API_KEY` for
   Path 1, `NGC_API_KEY` for Path 2).
2. Send one lightweight request against the endpoint.
3. A `401`/`403` means the key is missing, wrong, or expired -- regenerate it.
4. A connection-refused/timeout on the self-hosted path means the container
   is not up yet or the port mapping is wrong -- check `docker ps` and
   container logs before assuming an auth problem.

## Hello world: your first real request

Once the connectivity checklist returns a `200`, prove the connection works
for the model's actual job (reading a calibration plot) before moving on.
Full verified walkthrough:
[`references/hosted-api-quickstart.md`](references/hosted-api-quickstart.md)
(Path 1). It pulls a real sample image from the public QCalEval benchmark
dataset and asks the model to describe it -- no local test image needed.

## Next steps

Once you have gotten a real answer back from the hello-world request above,
this skill's job is done -- you have a working, authenticated connection.
What to do with it is out of scope here; go to:

- **Building an agentic calibration workflow?** Start with the
  [Quantum Calibration Agent Blueprint](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint/)
  -- an end-to-end workflow built on this model.
- **Want the other models in this family (decoders, other calibration
  sizes)?** Browse the
  [Hugging Face collection](https://huggingface.co/collections/nvidia/nvidia-ising).
- **Training or optimizing a predecoder?** See the
  [Ising-Decoding cookbook](https://github.com/NVIDIA/Ising-Decoding).

## Limitations

- Self-hosted NIM does not support tool calling.
- Hosted API rate limits and pricing are not published in this skill; check
  the build.nvidia.com model page for current terms.
- GPU memory requirements vary by precision (BF16 vs NVFP4) and GPU family --
  see `references/self-hosted-nim-quickstart.md` for the tested minimums.

## Troubleshooting

- **Error**: `401 Unauthorized` on the hosted API. **Cause**: missing/expired
  `NVIDIA_API_KEY`, or using the NGC key instead of the `nvapi-` key.
  **Solution**: regenerate at `build.nvidia.com/settings/api-keys`.
- **Error**: `docker login` or `docker pull` fails with auth error. **Cause**:
  missing/expired `NGC_API_KEY`, or using the `nvapi-` key instead of the NGC
  key. **Solution**: regenerate at `org.ngc.nvidia.com/setup/personal-keys`.
- **Error**: connection refused on `localhost:8000`. **Cause**: NIM container
  still loading weights, crashed, or the port mapping does not match.
  **Solution**: check `docker ps` and `docker logs <container>`; wait for the
  container to report ready before sending requests.
- **Error**: `Insufficient GPU memory` / OOM on container start. **Cause**:
  GPU does not meet the minimum for the chosen precision. **Solution**: use a
  smaller precision (NVFP4) or a GPU with more memory; see
  `references/self-hosted-nim-quickstart.md`.
