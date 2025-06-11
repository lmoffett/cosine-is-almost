from typing import List

from .early_stopping import ProjectPatienceEarlyStopping  # noqa F401
from .scheduling import (  # noqa F401
    ClassificationBackpropPhase,
    ClassificationInferencePhase,
    ProjectPhase,
    PrunePrototypesPhase,
    RSampleInitPhase,
    TrainLayersUsingProtoPNetNames,
)
from .types import (  # noqa F401
    EarlyStopping,
    IterativePhase,
    Phase,
    PhaseType,
    PostPhaseSummary,
    PrePhaseSummary,
    SetTrainingLayers,
    StepContext,
    TrainingSchedule,
)


def repeated(phases: List[Phase], iterations: int) -> IterativePhase:
    return IterativePhase(phases=phases, iterations=iterations)
