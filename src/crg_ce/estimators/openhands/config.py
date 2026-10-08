import builtins
import os
from logging import Logger
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TypeVar

import yaml
from datasets import Dataset, load_dataset
from jinja2 import Template
from openhands.sdk import get_logger
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from crg_ce.estimators.calibration import CalibrationConfig
from crg_ce.estimators.openhands.replay_estimator import ReplayEstimatorConfig
from crg_ce.graph.edges import EvidenceEdgeLabel
from crg_ce.utils.general import resolve_template

SupportedAgentToolPreset = Literal[
    "default",
    "gsn_agentic_graph_construction",
    "gsn_agentic_graph_construction_interp",
    "gsn_agentic_gather_evidence_v1",
]

# These are sets of tools that the agent might have used in a trajectory we might attempt to resume
# or otherwise deserialize events from, NOT the tools we want to make available to a resumed agent.
SupportedAgentToolSet = Literal["default"]


class AgentConfig(BaseModel):
    model_name: str = Field(description="model name for the agent", default="")
    api_base: str | None = None
    api_key: SecretStr | None = Field(
        description="API key to use. Prefixes of $ resolve through environment variables (e.g. `$OPENAI_API_KEY`)",
        default=None,
    )
    top_p: float | None = None
    enable_encrypted_reasoning: bool = True
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max", "none"] = "none"
    allowed_openai_params: list[str] = Field(
        default_factory=list,
        description="LiteLLM OpenAI-compatible parameters to forward even when the model metadata does not list them.",
    )
    max_output_tokens: int | None = None
    max_concurrent_llm_calls: int | None = Field(default=None, gt=0)
    timeout: int | None = Field(default=600, ge=0)
    tools_preset: SupportedAgentToolPreset = "default"
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)


class ConfidenceVerbalizationConfig(BaseModel):
    scale_min: float = 0
    scale_max: float = 10
    scale_suffix: str = ""


class DomainSuccessCriteriaConfig(BaseModel):
    default: Template = Field(
        default_factory=lambda: resolve_template("prompts/domains/swe/agent_success_criteria.txt")
    )
    by_benchmark: dict[str, Template] = Field(default_factory=dict)

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("default", mode="before")
    @classmethod
    def resolve_default_template(cls, value: object) -> object:
        return resolve_template(value) if isinstance(value, str) else value

    @field_validator("by_benchmark", mode="before")
    @classmethod
    def resolve_benchmark_templates(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise TypeError(f"Expected benchmark-template mapping, got {type(value).__name__}")
        return {
            benchmark: resolve_template(template_path) if isinstance(template_path, str) else template_path
            for benchmark, template_path in value.items()
        }

    def render(self, benchmark: str | None = None) -> str:
        if benchmark is None:
            return self.default.render().strip()
        if benchmark not in self.by_benchmark:
            raise ValueError(f"No domain success criteria configured for benchmark {benchmark!r}")
        return self.by_benchmark[benchmark].render().strip()


type GSNNodeKind = Literal[
    "GSNGoalNode",
    "EvidenceNodeV2",
]

GSN_CORE_NODE_FIELDS: dict[GSNNodeKind, set[str]] = {
    "GSNGoalNode": {"goal_name", "auditable_claim"},
    "EvidenceNodeV2": {"evidence", "auditable_claim"},
}
GSN_OPTIONAL_NODE_FIELDS: dict[GSNNodeKind, set[str]] = {
    "GSNGoalNode": {"reasoning"},
    "EvidenceNodeV2": {"step_numbers", "contribution"},
}


def _default_gsn_node_fields() -> dict[GSNNodeKind, list[str]]:
    return {
        "GSNGoalNode": ["reasoning"],
        "EvidenceNodeV2": ["step_numbers", "contribution"],
    }


class GSNGraphVerbalizationConfig(BaseModel):
    node_fields: dict[GSNNodeKind, list[str]] = Field(default_factory=_default_gsn_node_fields)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @field_validator("node_fields")
    @classmethod
    def validate_node_fields(cls, value: dict[GSNNodeKind, list[str]]) -> dict[GSNNodeKind, list[str]]:
        for node_kind, fields in value.items():
            if len(fields) != len(set(fields)):
                raise ValueError(f"Duplicate configured fields for {node_kind}: {fields}")
            automatic_fields = GSN_CORE_NODE_FIELDS[node_kind]
            configured_automatic_fields = automatic_fields.intersection(fields)
            if configured_automatic_fields:
                raise ValueError(
                    f"Core fields are always rendered for {node_kind}: {sorted(configured_automatic_fields)}"
                )
            unknown_fields = set(fields) - GSN_OPTIONAL_NODE_FIELDS[node_kind]
            if unknown_fields:
                raise ValueError(f"Unsupported fields for {node_kind}: {sorted(unknown_fields)}")
        return value


class LiteLLMVerbalEstimatorConfig(BaseSettings):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    query_type: Literal["structured", "ask_and_parse", "true_or_false", "sampled_true_or_false"] = "structured"
    instruction_template: str = "prompts/confidence_estimation/litellm/base_verbalizer.j2"
    ask_and_parse_output_instruction: str = "prompts/confidence_estimation/litellm/ask_and_parse_output_instruction.txt"
    verbalization: ConfidenceVerbalizationConfig = Field(default_factory=ConfidenceVerbalizationConfig)
    domain_success_criteria: DomainSuccessCriteriaConfig | None = None
    true_or_false_samples: int = Field(default=10, ge=3)

    def render_domain_success_criteria(self, benchmark: str | None) -> str | None:
        if not self.domain_success_criteria:
            return None
        if benchmark is None:
            raise ValueError("LiteLLM verbal estimation requires a benchmark to select domain success criteria")
        return self.domain_success_criteria.render(benchmark)


SupportedGatherEvidenceMethod = Literal["all_at_once"]
SupportedGSNGraphGeneratorType = Literal["agentic"]
SupportedGSNGraphPopulatorType = Literal["litellm"]


def _default_evidence_edge_labels() -> list[EvidenceEdgeLabel]:
    return ["proves", "supports", "refutes", "undermines"]


class GSNPromptConfig(BaseModel):
    system_prompt_instruction: Template = Field(default_factory=lambda: resolve_template("prompts/not_used.j2"))
    confidence_estimation_system_prompt_instruction: Template = Field(
        default_factory=lambda: resolve_template("prompts/not_used.j2")
    )
    domain_success_criteria: DomainSuccessCriteriaConfig | None = Field(default_factory=DomainSuccessCriteriaConfig)
    gather_evidence_method: SupportedGatherEvidenceMethod = Field(default="all_at_once")
    evidence_edge_labels: list[EvidenceEdgeLabel] = Field(default_factory=_default_evidence_edge_labels)
    decompose_goal_included_fields: list[str] = Field(default_factory=lambda: ["trajectory"])
    gather_evidence_included_fields: list[str] = Field(default_factory=lambda: ["trajectory"])
    skip_system_user_messages_in_trajectory: bool = False
    goal_zero_task_name: str = Field(default="Agent's Overall Task")
    goal_zero_auditable_claim: str = Field(default="The assistant achieved the user's task successfully")
    goal_zero_reasoning: str = Field(default="")
    goal_decompose_instruction: Template
    gather_evidence_instruction: Template
    assign_confidence_to_goal_instruction: Template
    assign_confidence_to_evidence_instruction: Template

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("*", mode="before")
    @classmethod
    def resolve_instruction_template(cls, value: object, info: ValidationInfo) -> object:
        if info.field_name and info.field_name.endswith("instruction") and isinstance(value, str):
            return resolve_template(value)
        if info.field_name == "domain_success_criteria" and isinstance(value, str | Template):
            return {"default": value}
        return value

    def render_domain_success_criteria(self, benchmark: str | None = None) -> str | None:
        if self.domain_success_criteria is None:
            return None
        return self.domain_success_criteria.render(benchmark)

    @model_validator(mode="after")
    def validate_skipped_trajectory_context(self) -> "GSNPromptConfig":
        skips_initial_messages = self.skip_system_user_messages_in_trajectory
        includes_problem_statement = "problem_statement" in self.decompose_goal_included_fields
        if skips_initial_messages and not includes_problem_statement:
            raise ValueError(
                "skip_system_user_messages_in_trajectory requires problem_statement in decompose_goal_included_fields"
            )
        return self


class GQAPromptConfig(BaseModel):
    system_prompt: Template = Field(default_factory=lambda: resolve_template("prompts/gqa/global/v0/system_prompt.j2"))
    coverage: Template = Field(default_factory=lambda: resolve_template("prompts/gqa/global/v0/coverage.j2"))
    argument_coherence: Template = Field(
        default_factory=lambda: resolve_template("prompts/gqa/global/v0/argument_coherence.j2")
    )
    non_redundancy: Template = Field(
        default_factory=lambda: resolve_template("prompts/gqa/global/v0/non_redundancy.j2")
    )
    granularity: Template = Field(default_factory=lambda: resolve_template("prompts/gqa/global/v0/granularity.j2"))
    domain_success_criteria: Template = Field(
        default_factory=lambda: resolve_template("prompts/domains/swe/agent_success_criteria.txt")
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def resolve_prompt_template(cls, value: object) -> object:
        return resolve_template(value) if isinstance(value, str) else value


class EntailmenntPromptConfig(BaseModel):
    system_prompt: Template = Field(
        default_factory=lambda: resolve_template("prompts/gqa/local/entailment/v0/system_prompt.j2")
    )
    entailment_task_prompt: Template = Field(
        default_factory=lambda: resolve_template("prompts/gqa/local/entailment/v0/task_prompt.j2")
    )
    examples: Path | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def resolve_prompt_template(cls, value: object, info: ValidationInfo) -> object:
        if info.field_name in {"system_prompt", "entailment_task_prompt"} and isinstance(value, str):
            return resolve_template(value)
        return value


class LocalGQAMetricsConfig(BaseModel):
    joint_sufficiency: bool = True
    child_necessity: bool = True
    non_redundant_siblings: bool = False
    quality_of_particularization: bool = True
    confidence_aggregation_mse: bool = False

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class CondenseConfig(BaseModel):
    mode: Literal["full", "summarize_uncited"] = "full"
    goal_confidence_context: Literal[
        "inherit_evidence_citations",
        "all_action_summaries",
    ] = Field(
        default="inherit_evidence_citations",
        description=(
            "Whether goal confidence replays preserve actions cited by descendant evidence or summarize every action"
        ),
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class GSNGraphGeneratorConfig(BaseSettings):
    generator_type: SupportedGSNGraphGeneratorType
    agent: AgentConfig
    prompts: GSNPromptConfig
    max_steps: int = Field(default=5, gt=0)


class GSNGraphPopulatorConfig(BaseSettings):
    generator_type: SupportedGSNGraphPopulatorType
    agent: AgentConfig
    prompts: GSNPromptConfig
    verbalization: ConfidenceVerbalizationConfig = Field(default_factory=ConfidenceVerbalizationConfig)
    condense: CondenseConfig = Field(default_factory=CondenseConfig)
    confidence_population_mode: Literal[
        "all_nodes",
        "goal_leaves_product",
        "product_interp_verbalized",
        "product_interp_prior",
    ] = "all_nodes"
    interpolation_prior: float = Field(default=0.5, ge=0, le=1)


class BasicLiteLLMVerbalEstimatorConfig(BaseModel):
    estimator_type: Literal["litellm_verbal"] = "litellm_verbal"
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    litellm: LiteLLMVerbalEstimatorConfig

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class LogProbsEstimatorConfig(BaseModel):
    estimator_type: Literal["log_probs_estimator"] = "log_probs_estimator"
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    agent: AgentConfig
    completion_client: Literal["litellm", "openai"] = Field(
        default="litellm",
        description="Client used for the prompt-log-probability request.",
    )
    replay_from: Path | None = Field(
        default=None,
        description="Run-config YAML whose saved log-probability prompt and features should be re-aggregated.",
    )
    aggregation_type: Literal[
        "mean_log_prob",
        "last_action_mean_log_prob",
        "seq_prob_last",
        "seq_prob_first",
        "seq_prob_mean",
        "seq_prob_min",
        "len_norm_seq_prob_last",
        "len_norm_seq_prob_first",
        "len_norm_seq_prob_mean",
        "len_norm_seq_prob_min",
    ] = "mean_log_prob"

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class GSNPlainVerbalizedEstimatorConfig(BaseModel):
    estimator_type: Literal["gsn_plain_verbalized"] = "gsn_plain_verbalized"
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    load_graphs_from_path: Path
    instruction_template: str = "prompts/confidence_estimation/gsn/plain_verbalizer.j2"
    ask_and_parse_output_instruction: str = "prompts/confidence_estimation/litellm/ask_and_parse_output_instruction.txt"
    graph_verbalization: GSNGraphVerbalizationConfig = Field(default_factory=GSNGraphVerbalizationConfig)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class OHGSNEstimatorConfig(BaseModel):
    estimator_type: Literal["oh_gsn"] = "oh_gsn"
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    graph_generator: GSNGraphGeneratorConfig
    graph_populator: GSNGraphPopulatorConfig

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_graph_sources(self) -> "OHGSNEstimatorConfig":
        uses_interp_construction = (
            self.graph_generator.agent.tools_preset == "gsn_agentic_graph_construction_interp"
        )
        uses_interp_population = self.graph_populator.confidence_population_mode in {
            "product_interp_verbalized",
            "product_interp_prior",
        }
        assert not uses_interp_population or uses_interp_construction, (
            "interpolated population modes require gsn_agentic_graph_construction_interp"
        )
        return self


class GSNPostHocAggregateConfig(BaseModel):
    """Configure deterministic aggregation over the goal tree of a saved GSN graph."""

    estimator_type: Literal["gsn_aggregate"] = "gsn_aggregate"
    replay_from: Path = Field(description="Run-config YAML whose graph.json artifacts should be aggregated.")
    aggregation_type: Literal[
        "product",
        "product_leaf_adaptive_claim_dropout",
        "product_leaf_claim_dropout",
        "product_leaf_temperature_scaled",
        "product_all_temperature_scaled",
        "temperature_scale_final",
        "geometric_mean",
        "geometric_mean_leaves",
        "arithmetic_mean",
        "minimum",
        "maximum",
        "union_bound",
        "simple_bp_goal_leaf_v1",
        "simple_bp_goal_leaf_v2",
        "product_interp_prior",
    ]
    temperature: float | None = Field(default=None, gt=0)
    d: float | None = Field(default=None, ge=0, le=1)
    k_expected_leaves: float | None = Field(default=None, ge=0)
    interpolation_prior: float | None = Field(default=None, ge=0, le=1)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_aggregation_parameters(self) -> "GSNPostHocAggregateConfig":
        temperature_scaled_aggregation_types = {
            "product_leaf_temperature_scaled",
            "product_all_temperature_scaled",
            "temperature_scale_final",
        }
        if self.aggregation_type in temperature_scaled_aggregation_types and self.temperature is None:
            raise ValueError(f"aggregation_type={self.aggregation_type!r} requires temperature")
        if self.aggregation_type not in temperature_scaled_aggregation_types and self.temperature is not None:
            raise ValueError(f"aggregation_type={self.aggregation_type!r} does not use temperature")
        if self.aggregation_type == "product_leaf_claim_dropout" and self.d is None:
            raise ValueError("aggregation_type='product_leaf_claim_dropout' requires d")
        if self.aggregation_type != "product_leaf_claim_dropout" and self.d is not None:
            raise ValueError(f"aggregation_type={self.aggregation_type!r} does not use d")
        if self.aggregation_type == "product_leaf_adaptive_claim_dropout" and self.k_expected_leaves is None:
            raise ValueError("aggregation_type='product_leaf_adaptive_claim_dropout' requires k_expected_leaves")
        if self.aggregation_type != "product_leaf_adaptive_claim_dropout" and self.k_expected_leaves is not None:
            raise ValueError(f"aggregation_type={self.aggregation_type!r} does not use k_expected_leaves")
        if self.aggregation_type == "product_interp_prior" and self.interpolation_prior is None:
            raise ValueError("aggregation_type='product_interp_prior' requires interpolation_prior")
        if self.aggregation_type != "product_interp_prior" and self.interpolation_prior is not None:
            raise ValueError(f"aggregation_type={self.aggregation_type!r} does not use interpolation_prior")
        return self


SupportedConfidenceEstimatorConfig = Annotated[
    BasicLiteLLMVerbalEstimatorConfig  # verbal baselines
    | LogProbsEstimatorConfig  # surrogate
    | GSNPlainVerbalizedEstimatorConfig  # reason-with-graph
    | OHGSNEstimatorConfig  # ours
    | GSNPostHocAggregateConfig  # ours - ablate propagation method
    | ReplayEstimatorConfig,  # ours/others: replay or vary some part
    Field(discriminator="estimator_type"),
]


def agent_configs_for_estimator(cfg: SupportedConfidenceEstimatorConfig) -> tuple[AgentConfig, ...]:
    """Return the agent configurations which can issue LLM calls for an estimator."""
    if isinstance(cfg, BasicLiteLLMVerbalEstimatorConfig):
        return (cfg.litellm.agent,)
    if isinstance(cfg, LogProbsEstimatorConfig):
        return () if cfg.replay_from is not None else (cfg.agent,)
    if isinstance(cfg, GSNPlainVerbalizedEstimatorConfig):
        return (cfg.agent,)
    if isinstance(cfg, OHGSNEstimatorConfig):
        return (cfg.graph_generator.agent, cfg.graph_populator.agent)
    if isinstance(cfg, (GSNPostHocAggregateConfig, ReplayEstimatorConfig)):
        return ()
    raise ValueError(f"Unsupported estimator config: {cfg}")


SupportedTrajFormat = Literal["openhands"]


class _SliceMaker:
    def __getitem__(self, item):
        return item


_slice_maker = _SliceMaker()


def parse_slice(slice_str: str) -> slice:
    """Parse '[:10]', '10:20', '::2', etc. into a slice object, reusing Python's own slice syntax."""
    s = slice_str.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    result = eval(f"_slice_maker[{s}]")  # noqa: S307
    if not isinstance(result, builtins.slice):
        raise ValueError(f"Not a slice: {slice_str!r}")
    return result


def _parse_slice_field_for_config(v: object) -> slice | None:
    if v is None or isinstance(v, builtins.slice):
        return v
    if isinstance(v, str):
        return parse_slice(v)
    raise TypeError(f"Invalid type for slice field: {type(v)}")


class TrajectoryDataConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    trajectory_format: SupportedTrajFormat = Field(description="format to load trajectories from", default="openhands")
    base_path: Path = Field(description="base path to load trajectories from")
    slice: builtins.slice | None = Field(description="slice of the data to use", default=None)

    @field_validator("slice", mode="before")
    @classmethod
    def _parse_slice_field(cls, v: object) -> slice | None:  # type: ignore
        return _parse_slice_field_for_config(v)


class OutputConfig(BaseModel):
    output_dir: Path | None = Field(
        description="output dir to save results (None will lead to an inferred output dir)", default=None
    )


class BaseRunConfig(BaseSettings):
    output: OutputConfig = Field(default_factory=OutputConfig)
    max_workers: int = Field(default=8, ge=1, description="maximum number of parallel workers to use on this run")


RunConfigT = TypeVar("RunConfigT", bound=BaseRunConfig)


class HuggingFaceDatasetConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    logger: Logger = Field(default_factory=lambda: get_logger(__name__), exclude=True)
    path: str = Field(
        default="Brendan/openhands_ce_data",
        description="Hugging Face dataset repository id or local dataset loading path.",
    )
    # Note: name is a huggingface concept and could be used to define a meaningful sub-set such as 'swesmith-2k', etc.
    name: str | None = Field(default=None, description="Optional Hugging Face dataset config name.")
    split: str = Field(default="train", description="Dataset split to load before local slicing.")
    revision: str | None = Field(default=None, description="Optional Hugging Face dataset revision.")
    slice: builtins.slice | None = Field(default=None, description="Exact Python-style slice to apply to the split.")
    trajectory_path_root: Path | None = Field(
        default=None,
        description="Optional root used to resolve relative trajectory paths from the dataset.",
    )
    shuffle: bool = Field(default=False, description="Shuffle the selected dataset before sampling.")
    sample: int | None = Field(default=None, ge=1, description="Number of rows to retain after shuffling.")
    seed: int = Field(default=42, description="Seed used for deterministic dataset shuffling.")
    sub_sample_n: int = Field(ge=0, default=0, description="sub-sample a fixed number from this slice (seed=42)")

    @field_validator("slice", mode="before")
    @classmethod
    def _parse_slice_field(cls, v: object) -> slice | None:  # type: ignore
        return _parse_slice_field_for_config(v)

    @model_validator(mode="after")
    def _validate_sampling_config(self) -> "HuggingFaceDatasetConfig":
        if self.sample is not None and self.sub_sample_n:
            raise ValueError("sample and sub_sample_n cannot both be set")
        return self

    def load_dataset(self) -> Dataset:
        dataset = load_dataset(path=self.path, name=self.name, split=self.split, revision=self.revision)
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected a single Dataset for split={self.split!r}, got {type(dataset)}")

        selected_data = dataset
        if self.slice:
            indices = _slice_indices(self.slice, len(dataset))
            self.logger.info("Sliced %s/%s with %s to %d rows", self.path, self.split, self.slice, len(indices))
            selected_data = dataset.select(indices)
        if self.shuffle:
            self.logger.info("Shuffling %s/%s with seed=%d", self.path, self.split, self.seed)
            selected_data = selected_data.shuffle(seed=self.seed)
        if self.sample is not None:
            assert self.sample <= len(selected_data), f"sample={self.sample} > len(data)={len(selected_data)}"
            self.logger.info("Sampling %d points from dataset", self.sample)
            selected_data = selected_data.select(range(self.sample))
        if self.sub_sample_n:
            assert self.sub_sample_n <= len(selected_data), (
                f"sub_sample_n={self.sub_sample_n} > len(data)={len(selected_data)}"
            )
            self.logger.info("sub-sampling %d points from dataset w/ seed=42", self.sub_sample_n)
            selected_data = selected_data.shuffle(seed=42).select(range(self.sub_sample_n))
        return selected_data

    @property
    def base_path(self) -> Path:
        return Path(os.environ.get("DATA_BASE_PATH", "data")) / self.path / self.split


class GQARunConfig(BaseRunConfig):
    agent: AgentConfig
    graph_run_config: Path = Field(description="batch_run_ce config whose output graphs should be evaluated")
    dataset: HuggingFaceDatasetConfig = Field(default_factory=HuggingFaceDatasetConfig)
    prompts: GQAPromptConfig = Field(default_factory=GQAPromptConfig)
    criterion_workers: int = Field(default=4, ge=1, description="parallel criterion evaluations per graph")
    output_filename: str = "gqa_metrics.jsonl"

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(extra="forbid")


class LocalGQARunConfig(BaseRunConfig):
    agent: AgentConfig
    graph_run_config: Path = Field(description="batch_run_ce config whose output graphs should be evaluated")
    dataset: HuggingFaceDatasetConfig = Field(default_factory=HuggingFaceDatasetConfig)
    prompts: EntailmenntPromptConfig = Field(default_factory=EntailmenntPromptConfig)
    metrics: LocalGQAMetricsConfig = Field(default_factory=LocalGQAMetricsConfig)
    sample_n_pairs: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of pairs sampled independently from each enabled entailment metric per graph.",
    )
    output_filename: str = "local_gqa_metrics.jsonl"

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(extra="forbid")


def _slice_indices(slice_: builtins.slice, dataset_len: int) -> list[int]:
    range_indices = range(dataset_len)
    indices = list(range_indices[slice_])
    if slice_.start is not None and (slice_.start < -dataset_len or slice_.start > dataset_len):
        raise IndexError(f"Slice start {slice_.start} is outside dataset length {dataset_len}")
    if slice_.stop is not None and (slice_.stop < -dataset_len or slice_.stop > dataset_len):
        raise IndexError(f"Slice stop {slice_.stop} is outside dataset length {dataset_len}")
    return indices


def _default_output_dir(config_file_path: Path) -> Path:
    run_path = config_file_path.with_suffix("")
    if run_path.is_absolute():
        run_path = Path(*run_path.parts[1:])
    return Path("outputs") / run_path


def output_dir_for_run_config(config_file_path: Path) -> Path:
    """Return the conventional output directory corresponding to a run-config path."""
    return _default_output_dir(config_file_path)


def load_run_config[RunConfigT: BaseRunConfig](cfg_path: str | Path, run_cfg_class: type[RunConfigT]) -> RunConfigT:
    config_file_path = Path(cfg_path)
    config_text: str = config_file_path.read_text()
    cfg_data = yaml.safe_load(config_text)
    cfg = run_cfg_class.model_validate(cfg_data)
    if cfg.output.output_dir is None:
        cfg.output.output_dir = _default_output_dir(config_file_path)
        cfg.output.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output.output_dir / "run_config.yaml").write_text(config_text)
    return cfg
