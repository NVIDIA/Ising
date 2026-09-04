# Self-hosted NIM quickstart

Container: `nvcr.io/nim/nvidia/ising-calibration-1-5-31b:latest`
Catalog page: `https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/ising-calibration-1-5-31b`

## Get an NGC personal key

This is a **separate credential** from the `nvapi-` key used for the hosted
API (see `hosted-api-quickstart.md`) -- do not reuse one for the other.

1. Sign in at `https://org.ngc.nvidia.com`.
2. Go to `https://org.ngc.nvidia.com/setup/personal-keys` (or: account icon ->
   Setup -> Generate Personal Key).
3. Generate a key and load it into your shell without it appearing in your
   command history or terminal scrollback:

   ```bash
   read -rs NGC_API_KEY
   export NGC_API_KEY
   ```

   Paste the key at the prompt (input will not echo), then press Enter.
   `export NGC_API_KEY=...` typed as a literal command instead would leave
   the raw key in `~/.bash_history` / `~/.zsh_history` -- avoid that form.

### If you will be deploying regularly

If you re-run `docker login`/`docker run` against this NIM often (rather
than a one-off), pull `NGC_API_KEY` from an OS-level secret store at the
moment it is needed instead of a plaintext file -- e.g. `secret-tool` or
`pass` on Linux, Keychain via the `security` CLI on macOS, or a password
manager CLI (1Password `op`, Bitwarden `bw`). A `.env` file loaded per-project
(e.g. via `direnv`) is an acceptable middle ground if `chmod 600` and
excluded via `.gitignore` -- do not source it from a global shell rc file.

If the container is deployed as part of an automated pipeline rather than
started by a person at a terminal, use a real secrets manager (Vault, or
your cloud provider's Secrets Manager) and inject `NGC_API_KEY` as an
environment variable at deploy time. Never embed it in a Dockerfile, image
layer, or anything committed to a repo.

## Log Docker in to nvcr.io

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

`$oauthtoken` is a literal, fixed username used for API-key auth -- it is not
a placeholder for your own username.

## Pull and run

```bash
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-~/.cache/nim}
mkdir -p "$LOCAL_NIM_CACHE"

docker pull nvcr.io/nim/nvidia/ising-calibration-1-5-31b:latest

docker run --gpus all -it --rm \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  -v "$LOCAL_NIM_CACHE:/root/.cache/nim" \
  nvcr.io/nim/nvidia/ising-calibration-1-5-31b:latest
```

The volume mount persists downloaded model weights across restarts so the
container does not re-download them every run.

## GPU requirements

Requires an NVIDIA Ada Lovelace, Hopper, or Blackwell architecture GPU.
Minimum tested configurations:

| GPU family        | Memory / count                        |
|--------------------|----------------------------------------|
| L40S               | 48GB, 2 GPUs (BF16 / NVFP4)             |
| H200 SXM           | 141GB, 1-2 GPUs                         |
| GH200 / B200 / GB200 | 1-2 GPUs                              |
| GB300+             | single GPU                              |

## Calling the running container

Once up, the container serves an OpenAI-compatible HTTP API on port 8000 for
chat completions and multimodal (image) calibration-plot analysis. Tool
calling is not supported.

This documentation does not yet cover:

- The health-check endpoint path. NIM containers commonly expose
  `/v1/health/ready`, but this is not confirmed for this specific image.
- A real end-to-end request/response example. The request shape may differ
  slightly from the hosted API's.
- Container startup time and the log messages that indicate the container is
  ready to serve.
