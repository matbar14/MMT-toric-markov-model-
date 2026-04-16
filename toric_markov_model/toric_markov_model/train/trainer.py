"""Training loop and trainer class."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import grad_clip


@dataclass
class TrainState:
    step: int = 0
    loss: float = 0.0


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_steps: int,
        grad_clip_norm: float = 1.0,
        log_interval: int = 100,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval

    def train(self) -> TrainState:
        self.model.train()
        state = TrainState()
        start = time.time()
        running_loss = 0.0

        while state.step < self.max_steps:
            for batch in self.dataloader:
                input_ids = batch.to(self.device)
                logits, targets = self.model(input_ids)

                loss = F.cross_entropy(
                    logits.reshape(-1, self.model.vocab_size),
                    targets.reshape(-1),
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                state.step += 1
                state.loss = float(loss.item())
                running_loss += state.loss

                if state.step % self.log_interval == 0:
                    elapsed = time.time() - start
                    avg = running_loss / self.log_interval
                    print(
                        f"step={state.step} loss={avg:.4f} "
                        f"time={elapsed:.1f}s"
                    )
                    running_loss = 0.0

                if state.step >= self.max_steps:
                    break

        return state


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    grad_clip_norm: float = 1.0,
    log_interval: int = 100,
) -> TrainState:
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        max_steps=max_steps,
        grad_clip_norm=grad_clip_norm,
        log_interval=log_interval,
    )
    return trainer.train()
