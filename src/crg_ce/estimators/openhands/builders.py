from crg_ce.estimators.base_estimator import BaseConfidenceEstimator
from crg_ce.estimators.calibration import CalibratedConfidenceEstimator, build_calibrator
from crg_ce.estimators.openhands.config import (
    BasicLiteLLMVerbalEstimatorConfig,
    GSNPlainVerbalizedEstimatorConfig,
    GSNPostHocAggregateConfig,
    LogProbsEstimatorConfig,
    OHGSNEstimatorConfig,
    SupportedConfidenceEstimatorConfig,
)
from crg_ce.estimators.openhands.gsn_plain_verbalized_estimator import GSNPlainVerbalizedEstimator
from crg_ce.estimators.openhands.litellm_verbal_estimator import LiteLLMVerbalEstimator
from crg_ce.estimators.openhands.log_probs_estimator import LogProbsEstimator
from crg_ce.estimators.openhands.oh_gsn_estimator import (
    GSNPostHocAggregateEstimator,
    OHGSNConfidenceEstimator,
)
from crg_ce.estimators.openhands.replay_estimator import ReplayConfidenceEstimator, ReplayEstimatorConfig
from crg_ce.llm_concurrency import LLMCallLimiter


def build_openhands_estimator(
    cfg: SupportedConfidenceEstimatorConfig,
    *,
    llm_limiter: LLMCallLimiter | None = None,
    llm_limiters: dict[int, LLMCallLimiter] | None = None,
) -> BaseConfidenceEstimator:
    if llm_limiter is not None and llm_limiters is not None:
        raise ValueError("Specify either llm_limiter or llm_limiters, not both")

    def limiter_for(agent_config) -> LLMCallLimiter | None:
        if llm_limiters is None:
            return llm_limiter
        return llm_limiters[id(agent_config)]

    if isinstance(cfg, BasicLiteLLMVerbalEstimatorConfig):
        estimator: BaseConfidenceEstimator = LiteLLMVerbalEstimator(
            cfg.litellm, llm_limiter=limiter_for(cfg.litellm.agent)
        )
    elif isinstance(cfg, LogProbsEstimatorConfig):
        estimator = LogProbsEstimator(cfg, llm_limiter=limiter_for(cfg.agent))
    elif isinstance(cfg, GSNPlainVerbalizedEstimatorConfig):
        estimator = GSNPlainVerbalizedEstimator(cfg, llm_limiter=limiter_for(cfg.agent))
    elif isinstance(cfg, OHGSNEstimatorConfig):
        estimator = OHGSNConfidenceEstimator(cfg, llm_limiter=llm_limiter, llm_limiters=llm_limiters)
    elif isinstance(cfg, GSNPostHocAggregateConfig):
        estimator = GSNPostHocAggregateEstimator(cfg)
    elif isinstance(cfg, ReplayEstimatorConfig):
        estimator = ReplayConfidenceEstimator(cfg)
    else:
        raise ValueError(f"Unsupported OpenHands estimator config: {cfg}")

    calibrator = build_calibrator(cfg.calibration)
    if calibrator is None:
        return estimator
    return CalibratedConfidenceEstimator(estimator, calibrator)
