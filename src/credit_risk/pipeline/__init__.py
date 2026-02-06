from .runner import PipelineRunner, Stage, StageResult, topological_order
from .stages import Context, build_stages, evaluate_model


def run_pipeline(cfg, until: str | None = None, force=False):
    """Build the stage graph for ``cfg`` and run it."""
    ctx = Context(cfg)
    runner = PipelineRunner(build_stages(ctx), ctx, root=ctx.root, cache_dir=ctx.path("cache"))
    return runner.run(until=until, force=force)


__all__ = [
    "Context",
    "PipelineRunner",
    "Stage",
    "StageResult",
    "build_stages",
    "evaluate_model",
    "run_pipeline",
    "topological_order",
]
