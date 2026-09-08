import unittest

import torch

from toric_markov_model.model.markov import DiscreteMarkovChain
from toric_markov_model.model.trading_model_v3 import ToricTradingModelV3
from toric_markov_model.train.trading import compute_loss


def tiny_model(**kwargs):
    return ToricTradingModelV3(num_features=28, dim_angles=16, max_len=8,
                               num_states=4, num_layers=1, dropout=0, **kwargs)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_markov_modes_are_identical(self):
        chain = DiscreteMarkovChain(8, 4)
        features, previous = torch.randn(2, 8), torch.ones(2, 4) / 4
        train = chain.train()(features, previous)[0]
        evaluation = chain.eval()(features, previous)[0]
        torch.testing.assert_close(train, evaluation, rtol=0, atol=0)
        torch.testing.assert_close(train.sum(1), torch.ones(2))
        with self.assertRaises(ValueError):
            chain.set_temperature(0)

    def test_whole_encoder_train_eval_match_without_dropout(self):
        model = tiny_model()
        features = torch.randn(2, 8, 28)
        train = model.train()(features)
        evaluation = model.eval()(features)
        for name in train:
            torch.testing.assert_close(train[name], evaluation[name], rtol=0, atol=0)

    def test_all_parameters_receive_gradients_and_attention_matters(self):
        model = tiny_model()
        features = torch.randn(2, 8, 28)
        labels = torch.zeros(2, 17)
        labels[0, 0], labels[1, -1] = 1, 1
        outputs = model(features)
        loss, _ = compute_loss(model, outputs, labels, torch.ones(2, 4), torch.ones(16), torch.ones(1))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        cell = model.toric_layers[0]
        self.assertGreater(abs(cell.attn_scale.grad.item()), 1e-8)
        model.eval()
        before = model(features)["pattern_logits"]
        with torch.no_grad():
            for context in cell.context:
                context.mul_(3)
        after = model(features)["pattern_logits"]
        self.assertGreater((before - after).abs().max().item(), 1e-4)
        self.assertFalse(hasattr(model, "regime_head"))
        self.assertFalse(hasattr(model, "confidence_head"))

    def test_gate_hold_and_threshold_contract(self):
        model = tiny_model(predict_return=False)
        logits = torch.full((2, 17), -10.0)
        logits[:, 0] = 10
        outputs = {"pattern_logits": logits, "non_hold_logit": torch.tensor([[-10.0], [1.0]])}
        decision = model.decode_outputs(outputs)
        self.assertEqual(decision["has_pattern"].tolist(), [False, True])
        self.assertEqual(decision["active_patterns"][:, -1].tolist(), [True, False])
        self.assertEqual(decision["strongest_pattern"].tolist(), [16, 0])
        strict = model.decode_outputs(outputs, confidence_threshold=0.99)
        self.assertFalse(strict["has_pattern"].any())
        torch.testing.assert_close(decision["hold_prob"], 1 - decision["non_hold_prob"])

    def test_regression_unbounded_and_inverse_scaling(self):
        model = tiny_model().eval()
        with torch.no_grad():
            for parameter in model.aux_head.parameters():
                parameter.zero_()
            model.aux_head[-1].bias.fill_(3)
        features = torch.zeros(1, 8, 28)
        self.assertEqual(model(features)["predicted_return"].item(), 3)
        with self.assertRaisesRegex(RuntimeError, "statistics"):
            model.detect_patterns(features)
        model.set_aux_target_stats({"aux_target_mean": [0.1, 1, 2, 3], "aux_target_std": [0.2, 2, 3, 4]})
        prediction = model.detect_patterns(features)
        self.assertAlmostEqual(prediction["predicted_return"].item(), 0.7, places=6)
        self.assertAlmostEqual(prediction["predicted_volume_change"].item(), 7)

    def test_input_and_inference_guards(self):
        model = tiny_model(predict_return=False)
        with self.assertRaises(RuntimeError):
            model.detect_patterns(torch.zeros(1, 8, 28))
        for features in (torch.zeros(1, 0, 28), torch.zeros(1, 9, 28), torch.zeros(1, 8, 29),
                         torch.full((1, 8, 28), float("nan"))):
            with self.assertRaises(ValueError):
                model(features)
