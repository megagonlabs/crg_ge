import argparse
import itertools
import json
import logging
import math
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar, NamedTuple, cast

import litellm
import numpy as np
import pandas as pd
from datasets import Dataset
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from crg_ce.datasets.oh_benchmarks.data_types import OpenHandsCEDataPoint
from crg_ce.estimators.openhands.config import LocalGQARunConfig, load_run_config, output_dir_for_run_config
from crg_ce.evaluators.contextual_entailment import (
    ContextualEntailmentEvaluator,
    EntailmentAssessment,
    EntailmentLabel,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.nodes import GSNGoalNode
from crg_ce.utils.litellm_utils import LiteLLMCallStats

logger = logging.getLogger(__name__)

ANALYSIS_FILENAME = "analysis.json"
REDO_ASPECTS = (
    "joint_sufficiency",
    "child_necessity",
    "non_redundant_siblings",
    "quality_of_particularization",
    "confidence_aggregation_mse",
)


class GraphInput(BaseModel):
    instance_id: str
    model: str
    task_description: str
    graph_path: Path

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class DecompositionAssessment(BaseModel):
    parent_id: str
    child_ids: list[str]
    premise: str
    hypothesis: str
    assessment: EntailmentAssessment

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class EntailmentClaimPair(NamedTuple):
    premise: str
    hypothesis: str
    parent_id: str
    child_ids: list[str]


class EntailmentMetricResult(BaseModel):
    score: float | None
    entailment_count: int = Field(ge=0)
    contradiction_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    total_predictions: int = Field(ge=0)
    assessments: list[DecompositionAssessment]

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class TotalEntailmentsMetric(BaseModel):
    score: float | None
    observed_count: int = Field(ge=0)
    expected_count: int = Field(ge=0)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ConfidenceAggregationComparison(BaseModel):
    parent_id: str
    child_ids: list[str]
    parent_confidence: float
    product_confidence: float
    geometric_mean_confidence: float
    arithmetic_mean_confidence: float

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ParticularizationConfidenceComparison(BaseModel):
    child_id: str
    parent_id: str
    child_confidence: float
    parent_confidence: float

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ConfidenceAggregationMSE(BaseModel):
    product_mse: float | None
    geometric_mean_mse: float | None
    arithmetic_mean_mse: float | None
    total_decompositions: int = Field(ge=0)
    comparisons: list[ConfidenceAggregationComparison]
    particularization_mse: float | None
    total_particularizations: int = Field(ge=0)
    particularization_comparisons: list[ParticularizationConfidenceComparison]

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class LocalGQAAnalysisArtifact(BaseModel):
    instance_id: str
    model: str
    evaluator_model: str
    graph_path: Path
    task_description: str
    joint_sufficiency: EntailmentMetricResult | None
    child_necessity: EntailmentMetricResult | None
    non_redundant_siblings: EntailmentMetricResult | None
    quality_of_particularization: EntailmentMetricResult | None
    total_entailments: TotalEntailmentsMetric
    confidence_aggregation_mse: ConfidenceAggregationMSE | None
    decomposition_count: int = Field(ge=0)
    particularization_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    cost: float = Field(ge=0)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class LocalGQAFailure(BaseModel):
    instance_id: str
    model: str
    graph_path: Path
    error: str


class AggregatedEntailmentMetric(BaseModel):
    n: int = Field(ge=0)
    score: float | None
    avg_entailment_count: float
    avg_contradiction_count: float
    avg_neutral_count: float
    avg_total_predictions: float


class AggregatedConfidenceAggregationMSE(BaseModel):
    n: int = Field(ge=0)
    product_mse: float | None
    geometric_mean_mse: float | None
    arithmetic_mean_mse: float | None
    avg_total_decompositions: float
    particularization_mse: float | None
    avg_total_particularizations: float


class AggregatedLocalGQAMetrics(BaseModel):
    n: int = Field(ge=0)
    joint_sufficiency: AggregatedEntailmentMetric | None
    child_necessity: AggregatedEntailmentMetric | None
    non_redundant_siblings: AggregatedEntailmentMetric | None
    quality_of_particularization: AggregatedEntailmentMetric | None
    total_entailments: TotalEntailmentsMetric
    confidence_aggregation_mse: AggregatedConfidenceAggregationMSE | None
    avg_decomposition_count: float
    avg_particularization_count: float
    avg_total_tokens: float
    avg_generated_tokens: float
    avg_cost: float


def load_graph_inputs(cfg: LocalGQARunConfig) -> list[GraphInput]:
    graph_dir = output_dir_for_run_config(cfg.graph_run_config)
    dataset: Dataset = cfg.dataset.load_dataset()
    dataset_items = cast(list[OpenHandsCEDataPoint], dataset.to_list())
    inputs = []
    seen_keys: set[tuple[str, str]] = set()
    for item in dataset_items:
        key = (item["instance_id"], item["model"])
        if key in seen_keys:
            raise ValueError(f"Duplicate instance_id/model in configured dataset: {key}")
        seen_keys.add(key)
        graph_path = graph_dir / item["instance_id"] / item["model"] / "graph.json"
        if not graph_path.is_file():
            logger.warning("Missing graph for instance_id=%s model=%s at %s", *key, graph_path)
            continue
        inputs.append(
            GraphInput(
                instance_id=item["instance_id"],
                model=item["model"],
                task_description=item["problem_statement"],
                graph_path=graph_path,
            )
        )
    return inputs


def decomposition_claims(graph: ConfidenceGraph) -> list[tuple[GSNGoalNode, list[GSNGoalNode]]]:
    nodes_by_id = {node.id: node for node in graph.nodes}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.relationship_type == "decomposes_from":
            children_by_parent[edge.target].append(edge.source)

    decompositions = []
    for parent_id, child_ids in children_by_parent.items():
        parent = nodes_by_id[parent_id]
        if not isinstance(parent, GSNGoalNode):
            raise TypeError(f"Decomposition parent must be a GSNGoalNode: {parent_id}")
        children = []
        for child_id in child_ids:
            child = nodes_by_id[child_id]
            if not isinstance(child, GSNGoalNode):
                raise TypeError(f"Decomposition child must be a GSNGoalNode: {child_id}")
            children.append(child)
        decompositions.append((parent, children))
    return decompositions


def particularization_claims(graph: ConfidenceGraph) -> list[tuple[GSNGoalNode, GSNGoalNode]]:
    nodes_by_id = {node.id: node for node in graph.nodes}
    particularizations = []
    for edge in graph.edges:
        if edge.relationship_type != "particularizes":
            continue
        child = nodes_by_id[edge.source]
        parent = nodes_by_id[edge.target]
        if not isinstance(child, GSNGoalNode):
            raise TypeError(f"Particularization child must be a GSNGoalNode: {child.id}")
        if not isinstance(parent, GSNGoalNode):
            raise TypeError(f"Particularization parent must be a GSNGoalNode: {parent.id}")
        particularizations.append((child, parent))
    return particularizations


def entailment_claim_pairs(
    decompositions: list[tuple[GSNGoalNode, list[GSNGoalNode]]],
    particularizations: list[tuple[GSNGoalNode, GSNGoalNode]],
) -> dict[str, list[EntailmentClaimPair]]:
    return {
        "joint_sufficiency": [
            EntailmentClaimPair(
                premise=" AND ".join(child.auditable_claim for child in children),
                hypothesis=parent.auditable_claim,
                parent_id=parent.id,
                child_ids=[child.id for child in children],
            )
            for parent, children in decompositions
        ],
        "child_necessity": [
            EntailmentClaimPair(
                premise=parent.auditable_claim,
                hypothesis=child.auditable_claim,
                parent_id=parent.id,
                child_ids=[child.id],
            )
            for parent, children in decompositions
            for child in children
        ],
        "non_redundant_siblings": [
            EntailmentClaimPair(
                premise=premise_child.auditable_claim,
                hypothesis=hypothesis_child.auditable_claim,
                parent_id=parent.id,
                child_ids=[first_child.id, second_child.id],
            )
            for parent, children in decompositions
            for first_child, second_child in itertools.combinations(children, 2)
            for premise_child, hypothesis_child in (
                (first_child, second_child),
                (second_child, first_child),
            )
        ],
        "quality_of_particularization": [
            EntailmentClaimPair(
                premise=premise.auditable_claim,
                hypothesis=hypothesis.auditable_claim,
                parent_id=parent.id,
                child_ids=[child.id],
            )
            for child, parent in particularizations
            for premise, hypothesis in ((child, parent), (parent, child))
        ],
    }


def sample_claim_pairs(
    claim_pairs: dict[str, list[EntailmentClaimPair]],
    sample_n_pairs: int | None,
    *,
    graph_key: str,
) -> dict[str, list[EntailmentClaimPair]]:
    if sample_n_pairs is None:
        return claim_pairs
    sampled = dict(claim_pairs)
    sibling_directions = claim_pairs["non_redundant_siblings"]
    if len(sibling_directions) % 2:
        raise ValueError("Non-redundancy directions must occur in pairs")
    sibling_pairs = [sibling_directions[index : index + 2] for index in range(0, len(sibling_directions), 2)]
    rng = random.Random(f"{graph_key}:non_redundant_siblings")
    indexes = sorted(rng.sample(range(len(sibling_pairs)), min(sample_n_pairs, len(sibling_pairs))))
    sampled["non_redundant_siblings"] = [direction for index in indexes for direction in sibling_pairs[index]]
    return sampled


def paired_check_score(metric: EntailmentMetricResult, *, success_label: EntailmentLabel) -> float | None:
    if len(metric.assessments) % 2:
        raise ValueError("Bidirectional checks must contain two assessments per logical check")
    checks = [metric.assessments[index : index + 2] for index in range(0, len(metric.assessments), 2)]
    if not checks:
        return None
    if success_label == "entailment":
        successes = sum(all(item.assessment.label == "entailment" for item in check) for check in checks)
    else:
        successes = sum(all(item.assessment.label != "entailment" for item in check) for check in checks)
    return successes / len(checks)


def total_entailments_metric(
    joint_sufficiency: EntailmentMetricResult | None,
    child_necessity: EntailmentMetricResult | None,
    quality_of_particularization: EntailmentMetricResult | None,
) -> TotalEntailmentsMetric:
    metrics = [
        metric for metric in (joint_sufficiency, child_necessity, quality_of_particularization) if metric is not None
    ]
    observed_count = sum(metric.entailment_count for metric in metrics)
    expected_count = sum(metric.total_predictions for metric in metrics)
    return TotalEntailmentsMetric(
        score=observed_count / expected_count if expected_count else None,
        observed_count=observed_count,
        expected_count=expected_count,
    )


class LocalGraphQualityAnalyzer:
    def __init__(self, cfg: LocalGQARunConfig) -> None:
        self.cfg = cfg
        self.evaluator = ContextualEntailmentEvaluator(cfg.agent, cfg.prompts)

    def analyze(
        self,
        graph_input: GraphInput,
        output_dir: Path,
        *,
        save_prompts: bool = False,
        existing_artifact: LocalGQAAnalysisArtifact | None = None,
        redo_aspects: set[str] | None = None,
    ) -> LocalGQAAnalysisArtifact:
        redo_aspects = redo_aspects or set()
        graph = ConfidenceGraph.model_validate_json(graph_input.graph_path.read_text())
        stats = LiteLLMCallStats()
        decompositions = decomposition_claims(graph)
        particularizations = particularization_claims(graph)
        claim_pairs = sample_claim_pairs(
            entailment_claim_pairs(decompositions, particularizations),
            self.cfg.sample_n_pairs,
            graph_key=f"{graph_input.instance_id}:{graph_input.model}",
        )
        joint_sufficiency = existing_artifact.joint_sufficiency if existing_artifact is not None else None
        if self.cfg.metrics.joint_sufficiency and (
            existing_artifact is None or "joint_sufficiency" in redo_aspects
        ):
            joint_sufficiency = self._evaluate_metric(
                "joint_sufficiency",
                claim_pairs["joint_sufficiency"],
                task_description=graph_input.task_description,
                output_dir=output_dir,
                save_prompts=save_prompts,
                stats=stats,
            )
        child_necessity = existing_artifact.child_necessity if existing_artifact is not None else None
        if self.cfg.metrics.child_necessity and (existing_artifact is None or "child_necessity" in redo_aspects):
            child_necessity = self._evaluate_metric(
                "child_necessity",
                claim_pairs["child_necessity"],
                task_description=graph_input.task_description,
                output_dir=output_dir,
                save_prompts=save_prompts,
                stats=stats,
            )
        non_redundant_siblings = existing_artifact.non_redundant_siblings if existing_artifact is not None else None
        if self.cfg.metrics.non_redundant_siblings and (
            existing_artifact is None or "non_redundant_siblings" in redo_aspects
        ):
            non_redundant_siblings = self._evaluate_metric(
                "non_redundant_siblings",
                claim_pairs["non_redundant_siblings"],
                success_label="neutral",
                task_description=graph_input.task_description,
                output_dir=output_dir,
                save_prompts=save_prompts,
                stats=stats,
            )
            non_redundant_siblings.score = paired_check_score(non_redundant_siblings, success_label="neutral")
        quality_of_particularization = (
            existing_artifact.quality_of_particularization if existing_artifact is not None else None
        )
        if self.cfg.metrics.quality_of_particularization and (
            existing_artifact is None or "quality_of_particularization" in redo_aspects
        ):
            quality_of_particularization = self._evaluate_metric(
                "quality_of_particularization",
                claim_pairs["quality_of_particularization"],
                success_label="entailment",
                task_description=graph_input.task_description,
                output_dir=output_dir,
                save_prompts=save_prompts,
                stats=stats,
            )
            quality_of_particularization.score = paired_check_score(
                quality_of_particularization, success_label="entailment"
            )
        confidence_aggregation_mse = (
            existing_artifact.confidence_aggregation_mse if existing_artifact is not None else None
        )
        if self.cfg.metrics.confidence_aggregation_mse and (
            existing_artifact is None or "confidence_aggregation_mse" in redo_aspects
        ):
            confidence_aggregation_mse = self._confidence_aggregation_mse(decompositions, particularizations)
        artifact = LocalGQAAnalysisArtifact(
            instance_id=graph_input.instance_id,
            model=graph_input.model,
            evaluator_model=self.cfg.agent.model_name,
            graph_path=graph_input.graph_path,
            task_description=graph_input.task_description,
            joint_sufficiency=joint_sufficiency,
            child_necessity=child_necessity,
            non_redundant_siblings=non_redundant_siblings,
            quality_of_particularization=quality_of_particularization,
            total_entailments=total_entailments_metric(
                joint_sufficiency,
                child_necessity,
                quality_of_particularization,
            ),
            confidence_aggregation_mse=confidence_aggregation_mse,
            decomposition_count=len(decompositions),
            particularization_count=len(particularizations),
            total_tokens=(existing_artifact.total_tokens if existing_artifact is not None else 0) + stats.total_tokens,
            generated_tokens=(existing_artifact.generated_tokens if existing_artifact is not None else 0)
            + stats.completion_tokens,
            cost=(existing_artifact.cost if existing_artifact is not None else 0) + stats.cost,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / ANALYSIS_FILENAME
        temporary_artifact_path = output_dir / f".{ANALYSIS_FILENAME}.tmp"
        temporary_artifact_path.write_text(artifact.model_dump_json(indent=2))
        temporary_artifact_path.replace(artifact_path)
        return artifact

    def _evaluate_metric(
        self,
        metric_name: str,
        claim_pairs: list[EntailmentClaimPair],
        *,
        success_label: EntailmentLabel = "entailment",
        task_description: str,
        output_dir: Path,
        save_prompts: bool,
        stats: LiteLLMCallStats,
    ) -> EntailmentMetricResult:
        assessments = []
        total_pairs = len(claim_pairs)
        logger.info("Evaluating %s: %d pairs", metric_name, total_pairs)
        for pair_index, claim_pair in enumerate(claim_pairs, start=1):
            if save_prompts:
                output_dir.mkdir(parents=True, exist_ok=True)
                prompt = self.evaluator.task_prompt.render(
                    premise=claim_pair.premise,
                    hypothesis=claim_pair.hypothesis,
                    examples=self.evaluator.examples,
                    agent_task_input=task_description,
                    trajectory=None,
                )
                (output_dir / f"{metric_name}_{claim_pair.parent_id}_{'_'.join(claim_pair.child_ids)}.txt").write_text(
                    prompt
                )
            assessment = self.evaluator.evaluate(
                claim_pair.premise,
                claim_pair.hypothesis,
                agent_task_input=task_description,
                stats=stats,
            )
            assessments.append(
                DecompositionAssessment(
                    parent_id=claim_pair.parent_id,
                    child_ids=claim_pair.child_ids,
                    premise=claim_pair.premise,
                    hypothesis=claim_pair.hypothesis,
                    assessment=assessment,
                )
            )
            if pair_index == 1 or pair_index % 10 == 0 or pair_index == total_pairs:
                logger.info("Entailment progress: metric=%s pairs=%d/%d", metric_name, pair_index, total_pairs)
        counts = Counter(item.assessment.label for item in assessments)
        total_predictions = len(assessments)
        return EntailmentMetricResult(
            score=counts[success_label] / total_predictions if total_predictions else None,
            entailment_count=counts["entailment"],
            contradiction_count=counts["contradiction"],
            neutral_count=counts["neutral"],
            total_predictions=total_predictions,
            assessments=assessments,
        )

    def _confidence_aggregation_mse(
        self,
        decompositions: list[tuple[GSNGoalNode, list[GSNGoalNode]]],
        particularizations: list[tuple[GSNGoalNode, GSNGoalNode]],
    ) -> ConfidenceAggregationMSE:
        comparisons = []
        for parent, children in decompositions:
            if parent.confidence == -1:
                raise ValueError(f"Decomposition parent has no confidence: {parent.id}")
            child_confidences = [child.confidence for child in children]
            missing_child_ids = [child.id for child in children if child.confidence == -1]
            if missing_child_ids:
                raise ValueError(f"Decomposition children have no confidence: {missing_child_ids}")
            product_confidence = math.prod(child_confidences)
            geometric_mean_confidence = product_confidence ** (1 / len(child_confidences))
            arithmetic_mean_confidence = float(np.mean(child_confidences))
            comparisons.append(
                ConfidenceAggregationComparison(
                    parent_id=parent.id,
                    child_ids=[child.id for child in children],
                    parent_confidence=parent.confidence,
                    product_confidence=product_confidence,
                    geometric_mean_confidence=geometric_mean_confidence,
                    arithmetic_mean_confidence=arithmetic_mean_confidence,
                )
            )

        product_squared_errors = [(item.parent_confidence - item.product_confidence) ** 2 for item in comparisons]
        geometric_mean_squared_errors = [
            (item.parent_confidence - item.geometric_mean_confidence) ** 2 for item in comparisons
        ]
        arithmetic_mean_squared_errors = [
            (item.parent_confidence - item.arithmetic_mean_confidence) ** 2 for item in comparisons
        ]
        particularization_comparisons = []
        for child, parent in particularizations:
            if child.confidence == -1:
                raise ValueError(f"Particularization child has no confidence: {child.id}")
            if parent.confidence == -1:
                raise ValueError(f"Particularization parent has no confidence: {parent.id}")
            particularization_comparisons.append(
                ParticularizationConfidenceComparison(
                    child_id=child.id,
                    parent_id=parent.id,
                    child_confidence=child.confidence,
                    parent_confidence=parent.confidence,
                )
            )
        particularization_squared_errors = [
            (item.child_confidence - item.parent_confidence) ** 2 for item in particularization_comparisons
        ]
        return ConfidenceAggregationMSE(
            product_mse=float(np.mean(product_squared_errors)) if comparisons else None,
            geometric_mean_mse=float(np.mean(geometric_mean_squared_errors)) if comparisons else None,
            arithmetic_mean_mse=float(np.mean(arithmetic_mean_squared_errors)) if comparisons else None,
            total_decompositions=len(comparisons),
            comparisons=comparisons,
            particularization_mse=(
                float(np.mean(particularization_squared_errors)) if particularization_comparisons else None
            ),
            total_particularizations=len(particularization_comparisons),
            particularization_comparisons=particularization_comparisons,
        )


def analyze_graphs(
    cfg: LocalGQARunConfig,
    *,
    resume: bool = True,
    save_prompts: bool = False,
    redo_aspects: set[str] | None = None,
) -> list[LocalGQAAnalysisArtifact]:
    redo_aspects = redo_aspects or set()
    output_dir = cfg.output.output_dir
    if output_dir is None:
        raise ValueError("Local GQA output directory was not resolved")
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = LocalGraphQualityAnalyzer(cfg)
    failures: list[LocalGQAFailure] = []
    lock = threading.Lock()

    def analyze(graph_input: GraphInput) -> LocalGQAAnalysisArtifact | None:
        item_output_dir = output_dir / graph_input.instance_id / graph_input.model
        artifact_path = item_output_dir / ANALYSIS_FILENAME
        try:
            if resume and artifact_path.is_file():
                artifact = LocalGQAAnalysisArtifact.model_validate_json(artifact_path.read_text())
                if redo_aspects:
                    artifact = analyzer.analyze(
                        graph_input,
                        item_output_dir,
                        save_prompts=save_prompts,
                        existing_artifact=artifact,
                        redo_aspects=redo_aspects,
                    )
            else:
                artifact = analyzer.analyze(graph_input, item_output_dir, save_prompts=save_prompts)
            logger.info(
                "Local GQA result: instance_id=%s model=%s joint_sufficiency=%s child_necessity=%s "
                "non_redundant_siblings=%s quality_of_particularization=%s confidence_aggregation_mse=%s",
                artifact.instance_id,
                artifact.model,
                artifact.joint_sufficiency.model_dump() if artifact.joint_sufficiency is not None else None,
                artifact.child_necessity.model_dump() if artifact.child_necessity is not None else None,
                artifact.non_redundant_siblings.model_dump() if artifact.non_redundant_siblings is not None else None,
                (
                    artifact.quality_of_particularization.model_dump()
                    if artifact.quality_of_particularization is not None
                    else None
                ),
                (
                    artifact.confidence_aggregation_mse.model_dump()
                    if artifact.confidence_aggregation_mse is not None
                    else None
                ),
            )
            return artifact
        except Exception as error:
            logger.exception("Failed instance_id=%s model=%s", graph_input.instance_id, graph_input.model)
            with lock:
                failures.append(
                    LocalGQAFailure(
                        instance_id=graph_input.instance_id,
                        model=graph_input.model,
                        graph_path=graph_input.graph_path,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            return None

    graph_inputs = load_graph_inputs(cfg)
    logger.info(
        "Starting local GQA: graphs=%d workers=%d resume=%s redo=%s",
        len(graph_inputs),
        cfg.max_workers,
        resume,
        sorted(redo_aspects),
    )
    artifacts_by_index: list[LocalGQAAnalysisArtifact | None] = [None] * len(graph_inputs)
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(analyze, graph_input): index for index, graph_input in enumerate(graph_inputs)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Local GQA", unit="graph"):
            artifacts_by_index[futures[future]] = future.result()
    artifacts = [artifact for artifact in artifacts_by_index if artifact is not None]
    (output_dir / "failures.json").write_text(
        json.dumps([failure.model_dump(mode="json") for failure in failures], indent=2)
    )
    return artifacts


def count_entailment_pairs(cfg: LocalGQARunConfig) -> dict[str, int]:
    metric_names = (
        "joint_sufficiency",
        "child_necessity",
        "non_redundant_siblings",
        "quality_of_particularization",
    )
    counts = {metric_name: 0 for metric_name in metric_names}
    graph_inputs = load_graph_inputs(cfg)
    for graph_input in graph_inputs:
        graph = ConfidenceGraph.model_validate_json(graph_input.graph_path.read_text())
        pairs_by_metric = sample_claim_pairs(
            entailment_claim_pairs(decomposition_claims(graph), particularization_claims(graph)),
            cfg.sample_n_pairs,
            graph_key=f"{graph_input.instance_id}:{graph_input.model}",
        )
        for metric_name in metric_names:
            if getattr(cfg.metrics, metric_name):
                counts[metric_name] += len(pairs_by_metric[metric_name])
    counts["graphs"] = len(graph_inputs)
    counts["total_entailments"] = sum(
        counts[metric_name] for metric_name in ("joint_sufficiency", "child_necessity", "quality_of_particularization")
    )
    counts["total_evaluator_pairs"] = sum(counts[metric_name] for metric_name in metric_names)
    return counts


def count_resume_work(cfg: LocalGQARunConfig, *, resume: bool, redo_aspects: set[str]) -> dict[str, int]:
    output_dir = cfg.output.output_dir
    if output_dir is None:
        raise ValueError("Local GQA output directory was not resolved")
    resumed_graphs = 0
    graphs_to_analyze = 0
    resumed_pairs = 0
    evaluator_pairs_to_run = 0
    for graph_input in load_graph_inputs(cfg):
        graph = ConfidenceGraph.model_validate_json(graph_input.graph_path.read_text())
        pairs_by_metric = sample_claim_pairs(
            entailment_claim_pairs(decomposition_claims(graph), particularization_claims(graph)),
            cfg.sample_n_pairs,
            graph_key=f"{graph_input.instance_id}:{graph_input.model}",
        )
        enabled_pair_counts = {
            metric_name: len(pairs_by_metric[metric_name]) if getattr(cfg.metrics, metric_name) else 0
            for metric_name in REDO_ASPECTS
            if metric_name != "confidence_aggregation_mse"
        }
        artifact_exists = (output_dir / graph_input.instance_id / graph_input.model / ANALYSIS_FILENAME).is_file()
        if resume and artifact_exists:
            resumed_graphs += 1
            rerun_pairs = sum(enabled_pair_counts[name] for name in redo_aspects if name in enabled_pair_counts)
            evaluator_pairs_to_run += rerun_pairs
            resumed_pairs += sum(enabled_pair_counts.values()) - rerun_pairs
            if redo_aspects:
                graphs_to_analyze += 1
        else:
            graphs_to_analyze += 1
            evaluator_pairs_to_run += sum(enabled_pair_counts.values())
    return {
        "resumed_graphs": resumed_graphs,
        "graphs_to_analyze": graphs_to_analyze,
        "resumed_evaluator_pairs": resumed_pairs,
        "evaluator_pairs_to_run": evaluator_pairs_to_run,
    }


def print_dry_run(cfg: LocalGQARunConfig, *, resume: bool = True, redo_aspects: set[str] | None = None) -> None:
    redo_aspects = redo_aspects or set()
    counts = count_entailment_pairs(cfg)
    resume_counts = count_resume_work(cfg, resume=resume, redo_aspects=redo_aspects)
    print(
        "Local GQA dry run:\n"
        f"  graphs: {counts['graphs']}\n"
        f"  joint_sufficiency: {counts['joint_sufficiency']}\n"
        f"  child_necessity: {counts['child_necessity']}\n"
        f"  non_redundant_siblings: {counts['non_redundant_siblings']}\n"
        f"  quality_of_particularization: {counts['quality_of_particularization']}\n"
        f"  total_entailments: {counts['total_entailments']}\n"
        f"  total_evaluator_pairs: {counts['total_evaluator_pairs']}\n"
        f"  resumed_graphs: {resume_counts['resumed_graphs']}\n"
        f"  graphs_to_analyze: {resume_counts['graphs_to_analyze']}\n"
        f"  resumed_evaluator_pairs: {resume_counts['resumed_evaluator_pairs']}\n"
        f"  evaluator_pairs_to_run: {resume_counts['evaluator_pairs_to_run']}"
    )


def aggregate_entailment_metric(metrics: list[EntailmentMetricResult]) -> AggregatedEntailmentMetric:
    scores = [metric.score for metric in metrics if metric.score is not None]
    return AggregatedEntailmentMetric(
        n=len(metrics),
        score=float(np.mean(scores)) if scores else None,
        avg_entailment_count=float(np.mean([metric.entailment_count for metric in metrics])) if metrics else 0,
        avg_contradiction_count=float(np.mean([metric.contradiction_count for metric in metrics])) if metrics else 0,
        avg_neutral_count=float(np.mean([metric.neutral_count for metric in metrics])) if metrics else 0,
        avg_total_predictions=float(np.mean([metric.total_predictions for metric in metrics])) if metrics else 0,
    )


def aggregate_confidence_aggregation_mse(
    metrics: list[ConfidenceAggregationMSE],
) -> AggregatedConfidenceAggregationMSE:
    product_mses = [metric.product_mse for metric in metrics if metric.product_mse is not None]
    geometric_mean_mses = [metric.geometric_mean_mse for metric in metrics if metric.geometric_mean_mse is not None]
    arithmetic_mean_mses = [metric.arithmetic_mean_mse for metric in metrics if metric.arithmetic_mean_mse is not None]
    particularization_mses = [
        metric.particularization_mse for metric in metrics if metric.particularization_mse is not None
    ]
    return AggregatedConfidenceAggregationMSE(
        n=len(metrics),
        product_mse=float(np.mean(product_mses)) if product_mses else None,
        geometric_mean_mse=float(np.mean(geometric_mean_mses)) if geometric_mean_mses else None,
        arithmetic_mean_mse=float(np.mean(arithmetic_mean_mses)) if arithmetic_mean_mses else None,
        avg_total_decompositions=float(np.mean([metric.total_decompositions for metric in metrics])) if metrics else 0,
        particularization_mse=float(np.mean(particularization_mses)) if particularization_mses else None,
        avg_total_particularizations=(
            float(np.mean([metric.total_particularizations for metric in metrics])) if metrics else 0
        ),
    )


def aggregate_local_gqa_metrics(artifacts: list[LocalGQAAnalysisArtifact]) -> AggregatedLocalGQAMetrics:
    joint_sufficiency_metrics = [item.joint_sufficiency for item in artifacts if item.joint_sufficiency is not None]
    child_necessity_metrics = [item.child_necessity for item in artifacts if item.child_necessity is not None]
    non_redundant_siblings_metrics = [
        item.non_redundant_siblings for item in artifacts if item.non_redundant_siblings is not None
    ]
    quality_of_particularization_metrics = [
        item.quality_of_particularization for item in artifacts if item.quality_of_particularization is not None
    ]
    confidence_aggregation_mse_metrics = [
        item.confidence_aggregation_mse for item in artifacts if item.confidence_aggregation_mse is not None
    ]
    observed_entailments = sum(item.total_entailments.observed_count for item in artifacts)
    expected_entailments = sum(item.total_entailments.expected_count for item in artifacts)
    return AggregatedLocalGQAMetrics(
        n=len(artifacts),
        joint_sufficiency=(
            aggregate_entailment_metric(joint_sufficiency_metrics) if joint_sufficiency_metrics else None
        ),
        child_necessity=(aggregate_entailment_metric(child_necessity_metrics) if child_necessity_metrics else None),
        non_redundant_siblings=(
            aggregate_entailment_metric(non_redundant_siblings_metrics) if non_redundant_siblings_metrics else None
        ),
        quality_of_particularization=(
            aggregate_entailment_metric(quality_of_particularization_metrics)
            if quality_of_particularization_metrics
            else None
        ),
        total_entailments=TotalEntailmentsMetric(
            score=observed_entailments / expected_entailments if expected_entailments else None,
            observed_count=observed_entailments,
            expected_count=expected_entailments,
        ),
        confidence_aggregation_mse=(
            aggregate_confidence_aggregation_mse(confidence_aggregation_mse_metrics)
            if confidence_aggregation_mse_metrics
            else None
        ),
        avg_decomposition_count=float(np.mean([item.decomposition_count for item in artifacts])) if artifacts else 0,
        avg_particularization_count=(
            float(np.mean([item.particularization_count for item in artifacts])) if artifacts else 0
        ),
        avg_total_tokens=float(np.mean([item.total_tokens for item in artifacts])) if artifacts else 0,
        avg_generated_tokens=float(np.mean([item.generated_tokens for item in artifacts])) if artifacts else 0,
        avg_cost=float(np.mean([item.cost for item in artifacts])) if artifacts else 0,
    )


def save_batch_outputs(cfg: LocalGQARunConfig, artifacts: list[LocalGQAAnalysisArtifact]) -> AggregatedLocalGQAMetrics:
    output_dir = cfg.output.output_dir
    if output_dir is None:
        raise ValueError("Local GQA output directory was not resolved")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / cfg.output_filename).write_text("\n".join(item.model_dump_json() for item in artifacts) + "\n")
    rows = []
    for item in artifacts:
        row = {
            "instance_id": item.instance_id,
            "model": item.model,
            "evaluator_model": item.evaluator_model,
            "graph_path": item.graph_path,
            "decomposition_count": item.decomposition_count,
            "particularization_count": item.particularization_count,
            "total_tokens": item.total_tokens,
            "generated_tokens": item.generated_tokens,
            "cost": item.cost,
            "total_entailments_score": item.total_entailments.score,
            "total_entailments_observed_count": item.total_entailments.observed_count,
            "total_entailments_expected_count": item.total_entailments.expected_count,
        }
        for metric_name in (
            "joint_sufficiency",
            "child_necessity",
            "non_redundant_siblings",
            "quality_of_particularization",
        ):
            metric = getattr(item, metric_name)
            if metric is not None:
                metric_values = metric.model_dump(exclude={"assessments"})
                row.update({f"{metric_name}_{name}": value for name, value in metric_values.items()})
        if item.confidence_aggregation_mse is not None:
            confidence_mse_values = item.confidence_aggregation_mse.model_dump(
                exclude={"comparisons", "particularization_comparisons"}
            )
            row.update({f"confidence_aggregation_mse_{name}": value for name, value in confidence_mse_values.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    metrics = aggregate_local_gqa_metrics(artifacts)
    (output_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    logger.info("Local GQA aggregate metrics: %s", metrics.model_dump_json())
    return metrics


def main(
    cfg: LocalGQARunConfig,
    *,
    resume: bool = True,
    save_prompts: bool = False,
    redo_aspects: set[str] | None = None,
) -> None:
    metrics = save_batch_outputs(
        cfg,
        analyze_graphs(
            cfg,
            resume=resume,
            save_prompts=save_prompts,
            redo_aspects=redo_aspects,
        ),
    )
    print(
        "Local GQA scores:\n"
        f"  joint_sufficiency: {_format_score(metrics.joint_sufficiency, 'score')}\n"
        f"  child_necessity: {_format_score(metrics.child_necessity, 'score')}\n"
        f"  non_redundant_siblings: {_format_score(metrics.non_redundant_siblings, 'score')}\n"
        f"  quality_of_particularization: {_format_score(metrics.quality_of_particularization, 'score')}\n"
        f"  total_entailments: {_format_score(metrics.total_entailments, 'score')} "
        f"({metrics.total_entailments.observed_count}/{metrics.total_entailments.expected_count})\n"
        f"  confidence_product_mse: {_format_score(metrics.confidence_aggregation_mse, 'product_mse')}\n"
        f"  confidence_geometric_mean_mse: "
        f"{_format_score(metrics.confidence_aggregation_mse, 'geometric_mean_mse')}\n"
        f"  confidence_arithmetic_mean_mse: "
        f"{_format_score(metrics.confidence_aggregation_mse, 'arithmetic_mean_mse')}\n"
        f"  particularization_mse: {_format_score(metrics.confidence_aggregation_mse, 'particularization_mse')}"
        f"\nReference stats:\n"
        f"  avg_decompositions_per_graph: {metrics.avg_decomposition_count:.4f}\n"
        f"  avg_particularizations_per_graph: {metrics.avg_particularization_count:.4f}"
    )


def _format_score(metrics: BaseModel | None, score_name: str) -> str:
    if metrics is None:
        return "N/A"
    score = getattr(metrics, score_name)
    return f"{score:.4f}" if score is not None else "N/A"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local graph decomposition sufficiency.")
    parser.add_argument("config", type=Path, help="Path to the local GQA run configuration YAML file")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing per-instance analysis.json artifacts (default: enabled)",
    )
    parser.add_argument("--save-prompts", action="store_true", help="Save each rendered entailment prompt")
    parser.add_argument(
        "--redo",
        action="append",
        choices=REDO_ASPECTS,
        default=[],
        metavar="ASPECT",
        help="Recompute this aspect in resumed artifacts; repeat to redo multiple aspects",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count graphs and configured entailment pairs without LLM calls",
    )
    return parser.parse_args(argv)


def cli() -> None:
    litellm.drop_params = True
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_run_config(args.config, LocalGQARunConfig)
    redo_aspects = set(args.redo)
    disabled_aspects = sorted(aspect for aspect in redo_aspects if not getattr(cfg.metrics, aspect))
    if disabled_aspects:
        raise ValueError(f"Cannot redo disabled GQA aspects: {disabled_aspects}")
    if args.dry_run:
        print_dry_run(cfg, resume=args.resume, redo_aspects=redo_aspects)
        return
    # Completed instances are reused via analysis.json.
    main(cfg, resume=args.resume, save_prompts=args.save_prompts, redo_aspects=redo_aspects)


if __name__ == "__main__":
    cli()
