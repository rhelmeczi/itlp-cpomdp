from typing import Sequence, Any, List
import json
import dataclasses
import numpy as np
from decimal import Decimal


class NumpyEncoder(json.JSONEncoder):
    """
    Inspired by `Stack Overflow <https://stackoverflow.com/a/47626762>`_.
    Provides the encoder for JSON to serialize an np.ndarray.
    """

    def default(self, obj):
        if isinstance(obj, np.longdouble):
            return float(obj)
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class JsonStorableDataClass:
    def save(self, fp):
        """
        Save this data.

        :param fp: A writeable file object.
        """
        json.dump(dataclasses.asdict(self), fp, cls=NumpyEncoder)

    @classmethod
    def load(cls, fp):
        """
        Load this data from a file path.

        :param fp: A readable file object.
        """
        return cls(**json.load(fp))


@dataclasses.dataclass
class FiniteConstrainedLpResult(JsonStorableDataClass):
    """
    Important information from an LP solution to a CPOMDP.
    """

    budget: float
    gridset: Sequence[float]
    deterministic: bool  # is this model deterministic
    optimal_x_tka: Any
    optimal_x_Nk: Any
    optimal_theta_tka: Any
    objective_value: float
    elapsed_time: float
    optimize_time: float = None
    v_hat: Sequence[float] = None

    def __post_init__(self):
        self.optimal_x_tka = np.array(self.optimal_x_tka)
        self.optimal_x_Nk = np.array(self.optimal_x_Nk)


def _round_sum_to_val_difference_remover(
    arr_rounded: Sequence[float], diff: Decimal
) -> List[Decimal]:
    arr_rounded_sum_to_val = arr_rounded
    if diff > 0:
        # addition is safe when diff is positive
        arr_rounded_sum_to_val[0] += diff
    elif diff < 0:
        # addition is safe iff array element is larger than diff when diff is negative
        diff_divisor = Decimal(
            "1"
        )  # try to divide the remaining_diff into parts with this divisor
        remaining_diff = diff
        while remaining_diff != Decimal("0"):
            arr_rounded_sum_to_val = arr_rounded  # reset arr_rounded_sum_to_val and remaining_diff for run with new divisor
            remaining_diff = diff
            diff_to_remove_per_item = diff / diff_divisor
            for i, v in enumerate(arr_rounded_sum_to_val):
                if v >= abs(
                    diff_to_remove_per_item
                ):  # subtract only when value at index is larger than diff_to_remove_per_item
                    arr_rounded_sum_to_val[i] += diff_to_remove_per_item
                    remaining_diff -= diff_to_remove_per_item

                if remaining_diff == Decimal("0"):
                    break

            diff_divisor *= Decimal(
                "2"
            )  # if current divisor did not work try with a larger one

    return arr_rounded_sum_to_val


def round_sum_to_val(
    arr: Sequence[float],
    round_to_num_of_digits: int = 12,
    sum_to_val: Decimal = Decimal("1"),
) -> List[float]:
    """
    Round an array to a set value. As the LP constraints are sensitive to floating point
    differences, this less-trivial normalizing algorithm ensures feasible LPs are generated.

    :param arr: The array to round.
    :param round_to_num_of_digists: The precision to round the results to.
    :param sum_to_val: What the sum of the final array should be.
    :return: The rounded array.
    """

    arr_rounded = [round(Decimal(v), round_to_num_of_digits) for v in arr]
    diff = sum_to_val - sum(arr_rounded)
    arr_rounded_sum_to_val = _round_sum_to_val_difference_remover(arr_rounded, diff)
    assert sum(arr_rounded_sum_to_val) == sum_to_val, (
        "Programming error arr_rounded_sum_to_val %s (sum=%s)\n does not sum up to %s\n"
        " original array %s (sum=%s)\n arr_rounded %s (sum=%s)\n diff %s"
        % (
            arr_rounded_sum_to_val,
            sum(arr_rounded_sum_to_val),
            sum_to_val,
            arr,
            sum(arr),
            arr_rounded,
            sum(arr_rounded),
            diff,
        )
    )
    return [float(v) for v in arr_rounded_sum_to_val]
