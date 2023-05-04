"""Command-line interface for solving constrained POMDPs with ITLP."""
from typing import Sequence, Optional
import datetime
import pathlib
import json
import pickle
import numpy as np
import pandas as pd
import typer
import tqdm
import itlp_cpomdp.linear_programming

app = typer.Typer()


def get_g_prime(
    prev_belief_state: np.ndarray,
    action_index: int,
    observation_index: int,
    transition_probs: np.ndarray,
    observation_probs: np.ndarray,
) -> np.ndarray:
    """Return the updated belief state."""
    num_states = len(prev_belief_state)
    g_prime = [None] * num_states
    for state_index_j in range(num_states):
        numerator = 0
        for state_index_i in range(num_states):
            numerator += (
                prev_belief_state[state_index_i]
                * transition_probs[action_index][state_index_i][state_index_j]
                * observation_probs[action_index][state_index_j][observation_index]
            )
        g_prime[state_index_j] = numerator

    denominator = sum(g_prime)

    # assert denominator!=0
    if denominator == 0:
        g_prime[state_index_j] = prev_belief_state[state_index_j]
    else:
        g_prime[state_index_j] = numerator / denominator

    g_prime = itlp_cpomdp.utils.round_sum_to_val(g_prime)
    return g_prime


def calculate_f_values(
    all_actions: np.ndarray,
    transition_probs: np.ndarray,
    all_observations: np.ndarray,
    observation_probs: np.ndarray,
    horizon: int,
    all_grid_points: Sequence[Sequence[float]],
    q_a_i: np.ndarray,
) -> np.ndarray:
    """Precompute the F values for setting up the LP."""
    max_range_val = horizon + 1

    # maximum reward over all actions a for each belief state k when t = 1
    terminal_rewards = np.amax(np.einsum("ki,ai->ka", all_grid_points, q_a_i), axis=1)
    prev_v_hat = terminal_rewards
    F_values = np.zeros(
        ((max_range_val), len(all_actions), len(all_grid_points), len(all_grid_points))
    )
    g_prime_dict = {}
    beta_calculator = itlp_cpomdp.linear_programming.StoredStateBetaCalculator(
        all_grid_points, itlp_cpomdp.linear_programming.BetaObjective.STANDARD
    )
    for time in tqdm.trange(2, max_range_val, desc="Computing F Values"):
        # calculate all beta for all possible action, observation, grid_value pair
        all_beta_values = np.zeros(
            (
                len(all_actions),
                len(all_observations),
                len(all_grid_points),
                len(all_grid_points),
            )
        )
        for belief_state_index, belief_state in enumerate(all_grid_points):
            for observation_index in range(len(all_observations)):
                for action_index in range(len(all_actions)):
                    g_prime_key = (
                        action_index,
                        observation_index,
                        belief_state_index,
                    )
                    if g_prime_key in g_prime_dict:
                        g_prime = g_prime_dict[g_prime_key]
                    else:
                        g_prime = get_g_prime(
                            belief_state,
                            action_index,
                            observation_index,
                            transition_probs,
                            observation_probs,
                        )
                        g_prime_dict[g_prime_key] = g_prime
                    all_beta_values[
                        action_index, observation_index, belief_state_index, :
                    ] = beta_calculator.get_backwards_induction_beta_values(
                        g_prime, prev_v_hat
                    )
        # m, n are grid indices, i, j are state indices, o is observation index,
        # a is action index
        F_values[time] = np.einsum(
            "mi,aij,ajo,aomn->amn",
            all_grid_points,
            transition_probs,
            observation_probs,
            all_beta_values,
            optimize="optimal",
        )
        prev_v_hat = np.amax(
            np.einsum("mi,ai->ma", all_grid_points, q_a_i, dtype=np.longdouble)
            + np.einsum("amn,n->ma", F_values[time], prev_v_hat, dtype=np.longdouble),
            axis=1,
        )
    return F_values


def run_itlp(
    horizon: int,
    all_grid_points: np.ndarray,
    budgets: Sequence[float],
    all_costs: np.ndarray,
    all_states: np.ndarray,
    all_actions: np.ndarray,
    all_observations: np.ndarray,
    transition_probs: np.ndarray,
    observation_probs: np.ndarray,
    q_a_i: np.ndarray,
    terminal_rewards: np.ndarray,
    results_path: pathlib.Path,
    delta: Optional[np.ndarray] = None,
):
    """Run the ITLP algorithm on a model."""
    f_values_timer_start = datetime.datetime.now()
    F_values = calculate_f_values(
        all_actions=all_actions,
        transition_probs=transition_probs,
        all_observations=all_observations,
        observation_probs=observation_probs,
        horizon=horizon,
        all_grid_points=all_grid_points,
        q_a_i=q_a_i,
    )
    f_values_elapsed_time = (
        datetime.datetime.now() - f_values_timer_start
    ).total_seconds()
    delta = itlp_cpomdp.utils.round_sum_to_val(
        [1 / len(all_grid_points)] * len(all_grid_points)
    )
    terminal_rewards = np.amax(np.einsum("ki,ai->ka", all_grid_points, q_a_i), axis=1)
    policies = itlp_cpomdp.linear_programming.lp_grid_upper_bound_dual(
        all_grid_points,
        horizon,
        len(all_actions),
        len(all_states),
        F_values,
        q_a_i,
        terminal_rewards,
        np.array(budgets),
        all_costs,
        delta,
    )

    summary_results = []
    for policy in policies:
        fname = f"budget-{policy.budget}"
        if policy.deterministic:
            fname += "_deterministic"
        fname += ".json"
        fpath = results_path / fname
        with open(fpath, "w", encoding="utf8") as fp:
            policy.save(fp)
        summary_results.append(
            {
                "budget": policy.budget,
                "deterministic": policy.deterministic,
                "V_LP": policy.objective_value,
                "f_values_elapsed_time": f_values_elapsed_time,
                "LP_time": policy.elapsed_time,
                "optimize_time": policy.optimize_time,
            }
        )
    df = pd.DataFrame(summary_results)
    print(df)
    df.to_excel(results_path / "summary_results.xlsx")


@app.command()
def train_constrained(
    data_path: pathlib.Path,
    results_path: pathlib.Path = pathlib.Path("results/tiger_model_test"),
    horizon: int = 20,
    budgets: list[float] = [21.0, 25.0, 50.0],
):
    """Train a constrained model using ITLP."""

    with open(data_path / "model_parameters.json", "r", encoding="utf8") as fp:
        model_parameters = json.load(fp)
    with open(data_path / "precomputed_values.pickle", "rb") as fp:
        precomputed_values = pickle.load(fp)
    results_path.mkdir(exist_ok=True, parents=True)
    run_itlp(
        horizon=horizon,
        budgets=budgets,
        all_costs=np.array(model_parameters["all_costs"]),
        all_states=np.array(model_parameters["all_states"]),
        all_actions=np.array(model_parameters["all_actions"]),
        all_observations=np.array(model_parameters["all_observations"]),
        transition_probs=np.array(model_parameters["transition_probs"]),
        observation_probs=np.array(model_parameters["observation_probs"]),
        q_a_i=np.array(precomputed_values["q_a_i"]),
        terminal_rewards=np.array(precomputed_values["terminal_rewards"]),
        all_grid_points=np.array(precomputed_values["all_grid_points"]),
        results_path=results_path,
    )


if __name__ == "__main__":
    app()
