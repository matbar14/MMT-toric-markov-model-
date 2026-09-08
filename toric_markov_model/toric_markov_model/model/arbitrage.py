"""A distinct paired-return model; incompatible with directional classifier checkpoints."""

from .payoff import ToricPayoffModel


class ToricArbitrageModel(ToricPayoffModel):
    def __init__(self, num_features, max_len=32, dim_angles=16, num_states=8, dropout=0.1, num_outputs=1):
        if num_outputs != 1:
            raise ValueError("arbitrage predicts one net return on committed pair capital")
        super().__init__(num_features, max_len, dim_angles, num_states, dropout, num_outputs=1)

    def predict_pair_return(self, features):
        return self.predict_payoffs(features)
