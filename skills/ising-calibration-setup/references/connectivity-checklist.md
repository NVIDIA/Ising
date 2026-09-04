# Connectivity checklist

Run this after setup to confirm the connection actually works, and to tell an
auth problem apart from a network/model-not-ready problem.

## Step 1: confirm the key is set

Hosted API:

```bash
test -n "$NVIDIA_API_KEY" && echo "NVIDIA_API_KEY is set" || echo "NVIDIA_API_KEY is EMPTY"
```

Self-hosted NIM:

```bash
test -n "$NGC_API_KEY" && echo "NGC_API_KEY is set" || echo "NGC_API_KEY is EMPTY"
```

If empty, go back to the relevant quickstart doc and generate the key -- do
not proceed to step 2.

## Step 2: send one lightweight request

Hosted API:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/ising-calibration-1.5-31b","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

Self-hosted NIM (adjust host/port -- `localhost:8000` only applies if you are
running the container on the same machine as this check; for a cloud/remote
GPU instance, use its address, e.g. its public IP or an SSH tunnel, and
confirm the port is actually open/reachable from where curl runs):

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

## Step 3: read the status code

| Code | Meaning | Path 1 (hosted) fix | Path 2 (self-hosted) fix |
|------|---------|----------------------|----------------------------|
| `200` | Working. | -- | -- |
| `401` / `403` | Auth failure. | Regenerate `NVIDIA_API_KEY` at `build.nvidia.com/settings/api-keys`; make sure you did not accidentally export the NGC key instead. | Self-hosted NIM does not usually require a request-time key once running; a 401 here more likely means the container is proxied behind something else -- check your setup. |
| `000` / connection refused / timeout | Network or service-not-ready failure, not auth. | Check general internet connectivity and that `integrate.api.nvidia.com` is reachable. | Container is not up yet or the port mapping is wrong. Run `docker ps` to confirm the container is running and `-p 8000:8000` is mapped, then `docker logs <container>` to check startup progress. On a cloud/remote instance, also check that the port is actually exposed to you (firewall / security group rules), not just bound inside the instance. |
| `429` | Rate limited. | Wait and retry; check the trial terms for limits. | N/A for self-hosted. |
| `5xx` | Server-side error. | Retry; if persistent, the hosted service may be degraded. | Check `docker logs <container>` for a crash or OOM. |

Do not assume a connection-refused error on the self-hosted path means the
key is wrong -- confirm the container is actually running and ready first.
