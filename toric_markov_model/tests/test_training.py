import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from toric_markov_model.train.trading import build_pos_weight, compute_loss, configure_stage_trainability, run_epoch
from tests.test_model import tiny_model


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_last_pattern_weight_is_not_treated_as_hold(self):
        labels = torch.zeros(10, 16)
        labels[0] = 1
        weights = build_pos_weight(labels, 15, "ratio")
        torch.testing.assert_close(weights, torch.full((16,), 9.0))

    def test_stage2_zero_loss_supports_backward(self):
        model = tiny_model()
        configure_stage_trainability(model, 2)
        outputs = model(torch.randn(2, 8, 28))
        labels = torch.zeros(2, 17)
        labels[:, -1] = 1
        loss, _ = compute_loss(model, outputs, labels, torch.zeros(2, 4),
                               torch.ones(16), torch.ones(1), stage=2)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(model.pattern_head[-1].weight.grad.abs().sum().item(), 0)

    def test_stage2_hold_only_batch_does_not_decay_weights(self):
        model = tiny_model()
        configure_stage_trainability(model, 2)
        previous = {name: value.clone() for name, value in model.state_dict().items()}
        labels = torch.zeros(3, 17)
        labels[:, -1] = 1
        dataset = TensorDataset(torch.randn(3, 8, 28), labels, torch.zeros(3, 4))
        optimizer = torch.optim.AdamW(model.pattern_head.parameters(), weight_decay=0.5)
        metrics = run_epoch(model, DataLoader(dataset, batch_size=2), "cpu", torch.ones(16),
                            torch.ones(1), stage=2, optimizer=optimizer)
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["optimizer_updates"], 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, previous[name], rtol=0, atol=0)

    def test_joint_training_updates_weights_and_keeps_last_batch(self):
        model = tiny_model()
        labels = torch.zeros(3, 17)
        labels[0, 0], labels[1, -1], labels[2, 1] = 1, 1, 1
        dataset = TensorDataset(torch.randn(3, 8, 28), labels, torch.zeros(3, 4))
        before = model.non_hold_gate_head[-1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters())
        metrics = run_epoch(model, DataLoader(dataset, batch_size=2), "cpu", torch.ones(16),
                            torch.ones(1), optimizer=optimizer)
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["optimizer_updates"], 2)
        self.assertFalse(torch.equal(before, model.non_hold_gate_head[-1].weight))

    def test_empty_loader_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty dataloader"):
            run_epoch(tiny_model(), [], "cpu", torch.ones(16), torch.ones(1))

    def test_validation_loss_does_not_depend_on_batch_partition(self):
        model = tiny_model()
        labels = torch.zeros(5, 17)
        labels[:, -1] = 1
        labels[0, -1], labels[4, -1] = 0, 0
        labels[0, 0], labels[4, 1] = 1, 1
        dataset = TensorDataset(torch.randn(5, 8, 28), labels, torch.randn(5, 4))
        small = run_epoch(model, DataLoader(dataset, batch_size=2), "cpu", torch.ones(16), torch.ones(1))
        large = run_epoch(model, DataLoader(dataset, batch_size=5), "cpu", torch.ones(16), torch.ones(1))
        for name in ("loss", "pattern_loss", "gate_loss", "aux_loss", "pattern_f1", "gate_f1"):
            self.assertAlmostEqual(small[name], large[name], places=6, msg=name)
