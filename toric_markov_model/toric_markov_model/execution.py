"""Shared bar-based exit rules for training outcomes and historical execution."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExecutionConfig:
    horizon: int = 4
    stop_loss: float = 0.01
    take_profit: float = 0.02
    fee: float = 0.001
    slippage: float = 0.0002
    position_fraction: float = 0.2
    cooldown: int = 1

    def __post_init__(self):
        if (not isinstance(self.horizon, int) or self.horizon < 1 or
                not isinstance(self.cooldown, int) or self.cooldown < 0):
            raise ValueError("horizon and cooldown must be valid integer bar counts")
        for name in ("stop_loss", "take_profit", "fee", "slippage", "position_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"invalid {name}")
        if min(self.stop_loss, self.take_profit, self.position_fraction) <= 0:
            raise ValueError("stops and position fraction must be positive")


def exit_on_bar(side, entry_price, bar_open, bar_high, bar_low, bar_close,
                stop_loss, take_profit, bars_held, max_hold_bars, priority="stop_first"):
    if side not in (-1, 1) or priority not in ("stop_first", "take_first"):
        raise ValueError("invalid side or intrabar priority")
    stop = entry_price * (1 - side * stop_loss)
    take = entry_price * (1 + side * take_profit)
    hit_stop = bar_low <= stop if side == 1 else bar_high >= stop
    hit_take = bar_high >= take if side == 1 else bar_low <= take
    if bar_open is not None:
        if side * (bar_open - stop) <= 0:
            return True, float(bar_open), "STOP_LOSS"
        if side * (bar_open - take) >= 0:
            return True, float(bar_open), "TAKE_PROFIT"
    if hit_stop and (not hit_take or priority == "stop_first"):
        return True, float(stop), "STOP_LOSS"
    if hit_take:
        return True, float(take), "TAKE_PROFIT"
    if max_hold_bars > 0 and bars_held >= max_hold_bars:
        return True, float(bar_close), "MAX_HOLD"
    return False, float(bar_close), ""
