from typing import Callable


def make_lr_schedule(
    warmup_steps: int, base_lr: float, target_lr: float, decay_rate: float
) -> Callable[[int], float]:
    """
    Creates a learning rate schedule function for use with PyTorch's LambdaLR.

    The schedule consists of:
    - An exponential warmup from `base_lr` to `target_lr` over `warmup_steps` iterations.
    - An exponential decay applied after `warmup_steps`, with decay factor `decay_rate`.

    Args:
        warmup_steps (int): Number of steps for exponential warmup.
        base_lr (float): Initial learning rate (must match optimizer's lr).
        target_lr (float): Learning rate after warmup.
        decay_rate (float): Per-step decay factor after warmup (e.g. 0.9999).

    Returns:
        Callable[[int], float]: A function mapping global step index to LR multiplier.
                                This can be passed to torch.optim.lr_scheduler.LambdaLR.
    """

    if warmup_steps <= 0:

        def lr_schedule(step: int) -> float:
            if step == 0:
                return base_lr
            else:
                # Exponential decay from base_lr to target_lr
                decay_steps = step - 1
                factor = (target_lr / base_lr) * (decay_rate**decay_steps)
                return factor
    else:

        def lr_schedule(step: int) -> float:
            if step <= warmup_steps:
                # Exponential increase from base_lr to target_lr
                factor = (target_lr / base_lr) ** (step / warmup_steps)
                return factor
            else:
                # Exponential decay from target_lr after warmup
                decay_steps = step - warmup_steps
                factor = (target_lr / base_lr) * (decay_rate**decay_steps)
                return factor

    return lr_schedule


# # Example usage:
# # Define config
# config = {
#     "warmup_steps": 10000,
#     "base_lr": 1e-6,
#     "target_lr": 1e-4,
#     "decay_rate": 0.9999
# }

# # Optimizer with base_lr
# optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"])

# # Scheduler
# schedule_fn = make_lr_schedule(**config)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_fn)
