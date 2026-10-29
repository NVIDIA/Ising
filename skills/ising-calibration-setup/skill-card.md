## Description: <br>
Use when connecting to the NVIDIA Ising Calibration model via the hosted API or a self-hosted NIM. Covers auth, keys and connectivity checks only. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and researchers use this skill to establish and verify a working connection to the NVIDIA Ising Calibration model (Ising-Calibration-1.5-31B), choosing between the hosted build.nvidia.com API and a self-hosted NIM container, obtaining the correct credential for each, and diagnosing failed connections. Prompting the model and building calibration workflows are out of scope. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>  

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. See skill body for more details. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Hosted API quickstart](references/hosted-api-quickstart.md) <br>
- [Self-hosted NIM quickstart](references/self-hosted-nim-quickstart.md) <br>
- [Connectivity checklist](references/connectivity-checklist.md) <br>
- [Ising Calibration model on build.nvidia.com](https://build.nvidia.com/nvidia/ising-calibration-1.5-31b) <br>
- [Ising Calibration NIM container on NGC](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/ising-calibration-1-5-31b) <br>
- [NVIDIA Ising repository](https://github.com/NVIDIA/Ising) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reports which connectivity checks passed or failed and the next diagnostic step; never includes credential values.] <br>

## Evaluation Tasks: <br>
Nine authored evaluation cases in evals/evals.json: six positive cases covering hosted setup, self-hosted setup, first requests, and connectivity diagnosis, and three negative controls where the skill must not activate. Live agent evaluation through NVSkills CI is pending. <br>

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
