"""Five-port position-sequence input/output model.

The public entry point is ``simulate_positions``.  It accepts a sequence of
physical sensor-node positions (1..5) and returns one plotted M value per
position.

Example
-------
>>> simulate_positions([1, 2, 3, 4, 5, 4, 3, 2, 1], use_memory=True)
[0.3, 0.3, 0.3, 0.3, 0.65, 0.615008..., 0.516048..., 0.369207..., 0.3]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from five_port_requested_memory import RequestedMemorySOT


Line = Literal["blue", "red"]


@dataclass(frozen=True)
class FivePortConfig:
    voltage_abs_v: float = 34.0
    r1_ohm: float = 411.0
    r2_ohm: float = 703.0
    r3_ohm: float = 949.0
    r4_ohm: float = 1113.0
    r5_ohm: float = 1295.0
    current_scale_to_ma: float = 1000.0
    output_current_sign: float = -1.0

    @property
    def effective_resistances_ohm(self) -> dict[str, float]:
        """Series resistance from port 1 to each directed pair."""
        return {
            "R21": self.r1_ohm + self.r2_ohm,
            "R12": self.r1_ohm + self.r2_ohm,
            "R31": self.r1_ohm + self.r3_ohm,
            "R41": self.r1_ohm + self.r4_ohm,
            "R51": self.r1_ohm + self.r5_ohm,
        }


DEFAULT_CONFIG = FivePortConfig()

# Blue traverses R21 -> R31 -> R41 -> R51 -> R12 as position increases.
BLUE_LABEL_BY_POSITION = {1: "R21", 2: "R31", 3: "R41", 4: "R51", 5: "R12"}

# Red is the mirrored directed-pair traversal used in the existing figures.
RED_LABEL_BY_POSITION = {1: "R12", 2: "R51", 3: "R41", 4: "R31", 5: "R21"}


def _validate_positions(positions: Iterable[int]) -> list[int]:
    result = [int(position) for position in positions]
    if not result:
        raise ValueError("positions must contain at least one position")
    invalid = [position for position in result if position not in range(1, 6)]
    if invalid:
        raise ValueError(f"positions must be integers from 1 to 5; got {invalid}")
    return result


def _label_for_position(position: int, line: Line) -> str:
    if line == "blue":
        return BLUE_LABEL_BY_POSITION[position]
    if line == "red":
        return RED_LABEL_BY_POSITION[position]
    raise ValueError("line must be 'blue' or 'red'")


def _current_ma(label: str, config: FivePortConfig) -> float:
    voltage = -config.voltage_abs_v if label == "R12" else config.voltage_abs_v
    resistance = config.effective_resistances_ohm[label]
    return config.output_current_sign * voltage / resistance * config.current_scale_to_ma


def _memory_values(currents_ma: Sequence[float]) -> list[float]:
    """Stateful peak-memory model: weaker/equal same-branch inputs hold M."""
    model = RequestedMemorySOT()
    values: list[float] = []
    branch: str | None = None
    branch_peak: float | None = None

    for current in currents_ma:
        new_branch = "positive" if current > 0 else "negative"
        entering_branch = new_branch != branch
        stronger = (
            entering_branch
            or branch_peak is None
            or (new_branch == "positive" and current > branch_peak)
            or (new_branch == "negative" and current < branch_peak)
        )

        if stronger:
            _, value, _ = model.step(current)
            branch_peak = current
        else:
            # The stimulus occurred, but it did not exceed the stored peak.
            model.x_prev = current
            value = model.m

        branch = new_branch
        values.append(float(value))

    return values


def _stateless_values(currents_ma: Sequence[float]) -> list[float]:
    """No-memory model: reset to the initial state before every position."""
    values = []
    for current in currents_ma:
        model = RequestedMemorySOT()
        _, value, _ = model.step(current)
        values.append(float(value))
    return values


def simulate_positions(
    positions: Iterable[int],
    *,
    use_memory: bool = True,
    line: Line = "blue",
    config: FivePortConfig = DEFAULT_CONFIG,
    return_details: bool = False,
) -> list[float] | list[dict[str, float | int | str | bool]]:
    """Convert a physical position sequence into plotted M output values.

    Parameters
    ----------
    positions:
        Physical sensor-node positions, each in the inclusive range 1..5.
    use_memory:
        ``True`` carries state across the sequence and holds M when a stimulus
        does not exceed the stored same-branch peak. ``False`` resets the model
        before every position.
    line:
        ``"blue"`` uses the forward directed-pair mapping. ``"red"`` uses the
        mirrored mapping and returns negative plotted values, matching figures.
    config:
        Voltage and physical R1..R5 values. Effective pair resistances are sums
        R1+R2, R1+R3, R1+R4, and R1+R5.
    return_details:
        When true, return per-step dictionaries instead of only M values.
    """
    checked_positions = _validate_positions(positions)
    labels = [_label_for_position(position, line) for position in checked_positions]
    currents = [_current_ma(label, config) for label in labels]
    raw_values = _memory_values(currents) if use_memory else _stateless_values(currents)
    plot_sign = -1.0 if line == "red" else 1.0
    plotted_values = [plot_sign * value for value in raw_values]

    if not return_details:
        return plotted_values

    resistances = config.effective_resistances_ohm
    return [
        {
            "step": step,
            "position": position,
            "label": label,
            "resistance_ohm": resistances[label],
            "current_ma": current,
            "M": value,
            "use_memory": use_memory,
        }
        for step, (position, label, current, value) in enumerate(
            zip(checked_positions, labels, currents, plotted_values), start=1
        )
    ]


def _parse_positions(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, help="Comma-separated positions, e.g. 1,2,3,4,5,4,3,2,1")
    parser.add_argument("--line", choices=("blue", "red"), default="blue")
    parser.add_argument("--no-memory", action="store_true", help="Reset the model before every position")
    parser.add_argument("--details", action="store_true", help="Print per-step dictionaries")
    args = parser.parse_args()

    result = simulate_positions(
        _parse_positions(args.positions),
        use_memory=not args.no_memory,
        line=args.line,
        return_details=args.details,
    )
    print(result)


if __name__ == "__main__":
    main()
