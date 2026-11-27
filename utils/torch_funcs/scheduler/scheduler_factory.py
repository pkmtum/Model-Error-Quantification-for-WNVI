from typing import Callable
import torch
from utils.torch_funcs.scheduler.function_custom_scheduler import make_lr_schedule


def scheduler_factory(
    name: str | None, options: dict
) -> Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None:
    """
    Returns a function that creates a scheduler for a given optimizer.
    If name is None, returns None.
    If name starts with 'custom_', uses custom logic (e.g. make_lr_schedule).
    Otherwise, fetches the torch scheduler by name and passes options.
    """
    if name is None:
        return None

    if name.startswith("custom_"):
        # Example: custom_lambda
        if name == "custom_lambda":
            schedule_fn = make_lr_schedule(
                options["warmup_steps"],
                options["base_lr"],
                options["target_lr"],
                options["decay_rate"],
            )

            def custom_lambda_creator(
                optimizer: torch.optim.Optimizer,
            ) -> torch.optim.lr_scheduler.LRScheduler:
                return torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=schedule_fn
                )

            return custom_lambda_creator
        # Add more custom schedulers here as needed
        raise ValueError(f"Unknown custom scheduler: {name}")

    # Standard torch schedulers
    try:
        scheduler_cls = getattr(torch.optim.lr_scheduler, name)
    except AttributeError:
        raise ValueError(f"Unknown scheduler name: {name}")

    def creator(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return scheduler_cls(optimizer, **options)

    return creator
