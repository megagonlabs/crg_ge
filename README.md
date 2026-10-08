# Confidence Reasoning Graphs

[![License: BSD 3-Clause with Attribution](https://img.shields.io/badge/License-BSD_3--Clause_with_Attribution-blue.svg)](LICENSE)

Code for **Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents**
by Brendan King, Farima Fatahi Bayat, Jean-Flavien Bussotti, Pouya Pezeshkpour,
and Estevam Hruschka. CRGs estimate task success from a single agent trajectory
by decomposing the task into claims, grounding them in trajectory evidence,
and aggregating confidence estimates.

The manuscript is [available on OpenReview](https://openreview.net/forum?id=Frn1IZARzJ)
and was submitted to ICLR 2027.

## Installation

Use Python 3.14 or newer and [uv](https://docs.astral.sh/uv/).
Run the commands below from the repository root. The lockfile pins official
OpenHands packages to version 1.31.0. It uses packages published on PyPI.

```bash
git clone https://github.com/megagonlabs/crg_ge.git
cd crg_ge
uv venv --python 3.14
uv sync --locked
```

The project installs numerical and machine learning dependencies; model serving
is configured separately as described below.

Run the local tests without downloading datasets or calling model services:

```bash
uv run pytest -m "not integration and not slow"
```

Tests marked `integration` require the referenced Hugging Face dataset and its
trajectory archives under `data/`. Tests that depend on recorded trajectory fixtures
require local copies and are skipped when those fixtures are absent; benchmark
archives and derived transcripts are not included in this release.

## Gathering Datasets

The following Hugging Face datasets provide task metadata and recorded outcomes
for the paper's evaluation on three benchmarks:

- [Brendan/openhands_ce_data_enterprise_ops](https://huggingface.co/datasets/Brendan/openhands_ce_data_enterprise_ops): 975 trajectories from 325 problems in [Enterprise-Ops Gym](https://enterpriseops-gym.github.io/), produced using GPT 5.5, Minimax-M3, and Gemini 3.5 Flash.
- [Brendan/openhands_ce_data_swebench](https://huggingface.co/datasets/Brendan/openhands_ce_data_swebench): 900 trajectories from the 306 hardest problems in [SWE-Bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified). Trajectories sourced from the [OpenHands Index](https://index.openhands.dev/), using the same 3 agent models.
- [Brendan/openhands_ce_data_skills_bench](https://huggingface.co/datasets/Brendan/openhands_ce_data_skills_bench): 219 trajectories from the 81 hardest problems in [SkillsBench](https://github.com/benchflow-ai/skillsbench), for the same 3 agent models.
- [Brendan/openhands_ce_data_dev](https://huggingface.co/datasets/Brendan/openhands_ce_data_dev): development set used for ablations. Contains problems from SWE-Smith and Enterprise-Ops Gym (non-overlapping problems with test set above).

These datasets contain metadata and paths to the recorded conversations; the
full trajectory archives are not included in the Hugging Face downloads.

### Using your own trajectories

You can evaluate CRGs on trajectories you generate with
[OpenHands](https://github.com/OpenHands/software-agent-sdk), using the benchmarks
above or your own tasks. Prepare the resulting executions as follows:

1. Save each completed OpenHands conversation as a `.tar.gz` archive containing
   its `base_state.json` and `events/` directory, preserving the SDK's event filenames.
2. Create a dataset with one row per execution containing `instance_id`, `model`,
   `problem_statement`, `conversation_archive_path`, and `resolved` (a boolean
   indicating the measured success or failure of that execution). Include `benchmark`
   when available.
3. Set `dataset.path` and `dataset.split` in a run YAML to your dataset. Store each
   archive at `data/<dataset.path>/<dataset.split>/<conversation_archive_path>`.
   Archive paths in the dataset should be relative and cannot contain `..`.
4. Configure the confidence-estimation models and run the experiment as described
   in [Running Experiments](#running-experiments).

The dataset may be hosted on Hugging Face or loaded from a relative local dataset
directory.
To use another archive location, set `DATA_BASE_PATH` to replace the `data/` prefix.
To check locally supplied OpenHands archives, replace the dataset ID below with
your own dataset ID or local directory:

```bash
uv run python scripts/validate_dataset.py --dataset-path your-org/your-trajectory-dataset
```

Pass `--base-path` as well if you use a custom `DATA_BASE_PATH`.

Use the task outcomes measured for your own executions when evaluating confidence.
Newly generated trajectories may yield different numerical results from those
reported in the paper; matching the original evaluation requires its recorded
trajectories and outcomes.

## Running Experiments

All experiments are defined with a YAML file under [runs/](./runs/).

### Model service setup

The checked-in Qwen configs expect an OpenAI-compatible server providing
`Qwen/Qwen3.8-27B-FP8` at `http://localhost:21046/v1`. The development
log-probability baseline uses `Qwen/Qwen3.8-27B` at
`http://localhost:21048/v1`. These ports refer to services you start yourself.

Install a compatible [vLLM release](https://docs.vllm.ai/en/stable/getting_started/installation/)
in a separate environment with the GPU resources required for the model and
context length. For example, the FP8 service can be started with:

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 21046 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
```

Check the [model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
and [vLLM tool calling documentation](https://docs.vllm.ai/en/stable/features/tool_calling/)
for chat template and parser support in your server version. Graph construction
requires tool calling; confidence population uses structured outputs. The
log-probability baseline additionally requires `prompt_logprobs` and
`return_token_ids` in the server response, as provided by vLLM. Its requests go
through the OpenAI/LiteLLM clients and do not require an OpenHands fork.

When using another endpoint, edit every applicable `agent.model_name` and
`agent.api_base` in the run YAML, including both `graph_generator` and
`graph_populator`. Changing model weights, quantization, or serving settings can
change experimental results.

For authenticated services, supply keys through environment variables. For
example, set `OPENAI_API_KEY` in your shell and add `api_key: "$OPENAI_API_KEY"`
to the relevant `agent` configuration. Keep credentials out of run files.

To run a single method on one dataset, you can call `uv run batch_run_ce` on that run file (**`--resume` flag recommended to maintain progress on incomplete runs**):

```bash
uv run batch_run_ce runs/dev_set/ours/qwen38_27b_agentic_fp8_max_depth_5.yaml --resume
```

#### Recommended: use batch_folder_runner.sh

A convenience script can be used to run each run file with `--resume` 3x, for all run-files under a target path:

```bash
bash batch_folder_runner.sh runs/dev_set
```

## Citation

Please cite the manuscript if you use this code or the accompanying datasets:

```bibtex
@misc{king2026confidence,
  title = {Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents},
  author = {King, Brendan and Fatahi Bayat, Farima and Bussotti, Jean-Flavien
            and Pezeshkpour, Pouya and Hruschka, Estevam},
  year = {2026},
  note = {Submitted to ICLR 2027},
  url = {https://openreview.net/forum?id=Frn1IZARzJ}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).


## Disclosures:

This software may include, incorporate, or access open source software (OSS) components,
datasets and other third party components, including those identified below. The license terms
respectively governing the datasets and third-party components continue to govern those
portions, and you agree to those license terms may limit any distribution, use, and copying.
You may use any OSS components under the terms of their respective licenses, which may
include BSD 3, Apache 2.0, and other licenses. In the event of conflicts between Megagon Labs,
Inc. (“Megagon”) license conditions and the OSS license conditions, the applicable OSS
conditions governing the corresponding OSS components shall prevail.
You agree not to, and are not permitted to, distribute actual datasets used with the OSS
components listed below. You agree and are limited to distribute only links to datasets from
known sources by listing them in the datasets overview table below. You agree that any right to
modify datasets originating from parties other than Megagon are governed by the respective
third party’s license conditions.
You agree that Megagon grants no license as to any of its intellectual property and patent rights.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS (INCLUDING
MEGAGON) “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. You agree to cease using,
incorporating, and distributing any part of the provided materials if you do not agree with the
terms or the lack of any warranty herein.
While Megagon makes commercially reasonable efforts to ensure that citations in this
document are complete and accurate, errors may occur. If you see any error or omission, please
help us improve this document by sending information to contact_oss@megagon.ai.

### Datasets

The dataset sources used by this project are listed below, including their copyright
holders and license information. Data is obtained from the linked sources or supplied
locally, as described in [Using your own trajectories](#using-your-own-trajectories).


For datasets with portions released under different licenses, refer to the linked
sources for the terms governing each portion.

| Source and credit | Used in | Upstream terms |
| --- | --- | --- |
| [EnterpriseOps-Gym, ServiceNow](https://github.com/ServiceNow/EnterpriseOps-Gym) | EnterpriseOps and development datasets | [Apache 2.0](https://github.com/ServiceNow/EnterpriseOps-Gym/blob/main/LICENSE) |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) and [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | SWE-bench dataset | [MIT for benchmark code](https://github.com/SWE-bench/SWE-bench/blob/main/LICENSE); original task repositories' licenses for their content |
| [OpenHands Index](https://index.openhands.dev/) and [OpenHands contributors](https://github.com/OpenHands/software-agent-sdk) | SWE-bench trajectories and conversation format | [MIT for the SDK](https://github.com/OpenHands/software-agent-sdk/blob/main/LICENSE); source benchmark and model terms for trajectories |
| [SkillsBench, BenchFlow](https://github.com/benchflow-ai/skillsbench) | SkillsBench dataset | [Apache 2.0](https://github.com/benchflow-ai/skillsbench/blob/main/LICENSE); referenced assets retain their source terms |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | Development tasks | [MIT for benchmark code](https://github.com/SWE-bench/SWE-smith/blob/main/LICENSE); original task repositories' licenses for their content |
| [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | Confidence estimation and surrogate models | Apache 2.0 for these model weights; consult each model card |

Evaluation uses AUROC, AUARC, and equal-width ECE with the conventions from CAGE-Cal:
[Counterfactual Graph for Multi-Agent LLM Calibration](https://arxiv.org/abs/2605.30653)
by Jiatan Huang, Mingchen Li, Ziming Li, Sunjae Kwon, Hong Yu, and Chuxu Zhang.

### Open Source Software (OSS) Components

The direct runtime dependencies and their licenses are listed below. Version
resolution, including transitive dependencies, is recorded in [uv.lock](uv.lock).
Dependencies and their bundled components retain their upstream licenses and
notices.

| Component and credit | License |
| --- | --- |
| [Datasets, Hugging Face](https://github.com/huggingface/datasets) | Apache 2.0 |
| [Jinja, Pallets](https://github.com/pallets/jinja) | BSD 3-Clause |
| [LiteLLM, BerriAI](https://github.com/BerriAI/litellm) | MIT |
| [Matplotlib contributors](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) | Matplotlib license |
| [NumPy developers](https://github.com/numpy/numpy) | BSD 3-Clause; additional licenses for bundled components |
| [OpenAI Python](https://github.com/openai/openai-python) | Apache 2.0 |
| [OpenHands SDK, tools, workspace, and agent server contributors](https://github.com/OpenHands/software-agent-sdk) | MIT |
| [pandas development team](https://github.com/pandas-dev/pandas) | BSD 3-Clause |
| [pgmpy contributors](https://github.com/pgmpy/pgmpy) | MIT |
| [Pydantic, pydantic-core, and pydantic-settings contributors](https://github.com/pydantic) | MIT |
| [PyYAML contributors](https://github.com/yaml/pyyaml) | MIT |
| [Rich, Will McGugan and contributors](https://github.com/Textualize/rich) | MIT |
| [scikit-learn developers](https://github.com/scikit-learn/scikit-learn) | BSD 3-Clause |
| [SciPy developers](https://github.com/scipy/scipy) | BSD 3-Clause; additional licenses for bundled components |
| [seaborn, Michael Waskom and contributors](https://github.com/mwaskom/seaborn) | BSD 3-Clause |
| [Tenacity contributors](https://github.com/jd/tenacity) | Apache 2.0 |
| [tqdm contributors](https://github.com/tqdm/tqdm) | MIT and MPL 2.0 |

Development tools include [dictdiffer](https://github.com/inveniosoftware/dictdiffer),
[pytest](https://github.com/pytest-dev/pytest),
[mypy](https://github.com/python/mypy), [Ruff](https://github.com/astral-sh/ruff),
[pre-commit](https://github.com/pre-commit/pre-commit), and
[tabulate](https://github.com/astanin/python-tabulate) (MIT),
[Jupyter](https://github.com/jupyter/jupyter) and
[pandas-stubs](https://github.com/pandas-dev/pandas-stubs) (BSD 3-Clause), and
the [typeshed](https://github.com/python/typeshed) packages `types-PyYAML` and
`types-tqdm` (Apache 2.0).
The build backend [uv-build, Astral](https://github.com/astral-sh/uv) is available
under MIT or Apache 2.0.

## Contact

For code questions, contact [Farima Fatahi Bayat](mailto:farima@megagon.ai) or
[Jean-Flavien Bussotti](mailto:jflavien@megagon.ai), or open an issue in this repository.
