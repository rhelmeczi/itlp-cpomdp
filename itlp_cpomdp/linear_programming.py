from typing import Sequence, Tuple, List, Optional
import logging
import enum
import time as timer_module
import numpy as np
import tqdm
from gurobipy import GRB, LinExpr, Constr, Model
import enum_tools.documentation
import itlp_cpomdp.utils

logger = logging.getLogger(__name__)


@enum_tools.documentation.document_enum
class BetaObjective(enum.Enum):
    """Indicates the objective function when calculating :math:`\\beta` values."""

    STANDARD = "standard"  # doc: Minimize :math:`\hat{v}`.
    MANHATTAN = "manhattan_distance"  # doc: Minimize the Manhattan distance.
    EUCLIDEAN = "euclidean_distance"  # doc: Minimize the Euclidean distance.


class StoredStateBetaCalculator:
    """
    A state-storing beta calculator which stores and updates the same model
    rather than recreating it from scratch each time the calculation is
    called.

    :param gridset: The set of belief states for the approximation.
    :type gridset: 2D array-like indexed by grid, state
    :param objective: The method for setting the objective function.
    :type objective: BetaObjective
    """

    def __init__(self, gridset: Sequence[Sequence[float]], objective: BetaObjective):
        if not isinstance(objective, BetaObjective):
            raise ValueError("unsupported objective")
        self._gridset = np.asarray(gridset)
        self._objective = objective
        self._model = None
        self._model_var_beta_values = None
        self._match_g_prime_constraints = None

    def get_backwards_induction_beta_values(
        self, g_prime: Sequence[float], v_hat: Optional[Sequence[float]] = None
    ) -> List[float]:
        """Compute the beta values.

        :param g_prime: A belief state.
        :param v_hat: The :math:`\hat{v}` values for the gridset.
        :return: The :math:`\\beta` values.
        """
        if self._model is None:
            assert g_prime is not None
            model = Model("backward_induction_beta_values")
            model.setParam("OutputFlag", 0)
            model.setParam("LogFile", "")
            model.setParam("LogToConsole", 0)
            model_var_beta_values = model.addVars(
                len(self._gridset),
                lb=0,
                ub=GRB.INFINITY,
                name="beta_values",
                obj=1.0,
                vtype=GRB.CONTINUOUS,
            )

            if self._objective is BetaObjective.MANHATTAN:
                expression = 0
                manhattan_distances = np.abs(self._gridset - g_prime).sum(axis=1)
                for idx in range(len(self._gridset)):
                    expression += manhattan_distances[idx] * model_var_beta_values[idx]
                model.setObjective(expression, GRB.MINIMIZE)
            elif self._objective is BetaObjective.STANDARD:
                expression = 0
                for idx in range(len(self._gridset)):
                    expression += v_hat[idx] * model_var_beta_values[idx]
                model.setObjective(expression, GRB.MINIMIZE)
            elif BetaObjective.EUCLIDEAN:
                expression = 0
                euclidean_distances = np.power(self._gridset - g_prime, 2).sum(axis=1)
                for idx in range(len(self._gridset)):
                    expression += euclidean_distances[idx] * model_var_beta_values[idx]
                model.setObjective(expression, GRB.MINIMIZE)
            else:
                raise ValueError("unsupported objective")

            # add constraint 1
            self._match_g_prime_constraints = []
            for i in range(len(g_prime)):
                expression = 0
                for l in range(len(self._gridset)):
                    expression += self._gridset[l][i] * model_var_beta_values[l]
                self._match_g_prime_constraints.append(
                    model.addConstr(expression == g_prime[i])
                )

            # add constraint 2
            expression = 0
            for l in range(len(self._gridset)):
                expression += model_var_beta_values[l]
            model.addConstr(expression == 1)
            self._model = model
            self._model_var_beta_values = model_var_beta_values
        else:
            # if we don't call reset, then the model starts from the previously
            # calculated beta values; calling reset makes this method consistent
            # with recreating the model every time
            self._model.reset()
            if self._objective is BetaObjective.MANHATTAN:
                manhattan_distances = np.abs(self._gridset - g_prime).sum(axis=1)
                for idx in range(len(self._gridset)):
                    self._model_var_beta_values[idx].obj = manhattan_distances[idx]
            elif self._objective is BetaObjective.STANDARD:
                for idx in range(len(self._gridset)):
                    self._model_var_beta_values[idx].obj = v_hat[idx]
            elif BetaObjective.EUCLIDEAN:
                euclidean_distances = np.power(self._gridset - g_prime, 2).sum(axis=1)
                for idx in range(len(self._gridset)):
                    self._model_var_beta_values[idx].obj = euclidean_distances[idx]
            else:
                raise ValueError("unsupported objective")
            for state_idx in range(len(self._match_g_prime_constraints)):
                self._match_g_prime_constraints[state_idx].rhs = g_prime[state_idx]
        self._model.optimize()
        if self._model.status != GRB.Status.OPTIMAL:
            raise ValueError(
                f"The model is not optimal. model.status = {self._model.status}"
            )
        results = list(self._model.getAttr("x", self._model_var_beta_values).values())
        return results


def verify_budgets(
    budget_values: Sequence[float],
    model: Model,
    budget_constraint_lhs: LinExpr,
    min_only=False,
) -> np.ndarray:
    """Filter out infeasibly small budgets.

    :param budgets: The budgets to filter.
    :param model: The LP model, with all other constraints already added except
        for deterministic constraints (if both deterministic and randomized cases
        are going to be considered).
    :param budget_constraint_lhs: The budget constraint. For a budget `b`, the LP
        has the constraint `budget_constraint_lhs < b`.
    :return: The filtered list of budgets.
    """
    model.update()  # ensure we get the correct objective function next
    model_objective = model.getObjective()

    # compute the minimum and maximum budgets
    model.setObjective(budget_constraint_lhs, GRB.MINIMIZE)
    model.optimize()
    if model.status == GRB.Status.OPTIMAL:
        smallest_possible_budget = model.objVal
    else:
        raise ValueError(
            f"Cannot solve for budget constraints; model.status = {model.status}"
        )

    if not min_only:
        model.setObjective(budget_constraint_lhs, GRB.MAXIMIZE)
        model.optimize()
        if model.status == GRB.Status.OPTIMAL:
            largest_possible_budget = model.objVal
        else:
            raise ValueError(
                f"Cannot solve for budget constraints; model.status = {model.status}"
            )

    # reset model objective to the LP objective rather than max/min budget
    model.setObjective(model_objective, GRB.MAXIMIZE)
    old_num_budgets = len(budget_values)
    budget_values = budget_values[budget_values >= smallest_possible_budget]
    new_num_budgets = len(budget_values)
    if new_num_budgets != old_num_budgets:
        logger.warning(
            f"Dropped a total of {new_num_budgets - old_num_budgets} for "
            "for being too small."
        )
    if len(budget_values) == 0:
        logger.error(
            "The provided range of budgets are all above the maximum "
            "possible budget. Adding the maximum requested budget..."
        )
        budget_values = np.array([smallest_possible_budget, largest_possible_budget])
    return budget_values


def solve_over_budgets(
    model: Model,
    all_grid_points: Sequence[Sequence[float]],
    model_var_x_tka,
    model_var_x_Nk,
    model_var_theta_tka,
    budget_values: Sequence[float],
    budget_constraint: Constr,
    setup_time: float,
    tka_dims: Tuple[int],
    Nk_dims: int,
    deterministic: bool,
) -> "List[itlp_cpomdp.utils.FiniteConstrainedLpResult]":
    """Solve the LP over an array of budgets.

    :param model (Model): The LP model with all constraints added.
    :param budget_values: The budgets to solve the problem for.
    :param budget_constraint: The budget constraint in `model`.
        For each budget, the rhs of this constraint is updated.
    :param deterministic: Does the model have deterministic constraints?

    :return: The LP solution for each budget.
    """
    result = []
    for idx, budget in enumerate(tqdm.tqdm(budget_values, desc="Optimizing budgets")):
        budget_constraint.rhs = budget
        logger.info(f"Solving for budget = {budget}, deterministic = {deterministic}")
        timer_start = timer_module.time()
        optimize_timer_start = timer_module.time()
        model.setParam("TimeLimit", 1000)
        model.optimize()
        optimize_time = timer_module.time() - optimize_timer_start
        # model.write('log/lp_grid_upper_bound_dual.lp')
        if model.status in (GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT):
            if model.status == GRB.Status.TIME_LIMIT:
                print(f"time limit reached for {budget}, det={deterministic}")
            model_values = model.getAttr("x", model_var_x_tka)
            optimal_x_tka = np.zeros(tka_dims)
            for key in model_values.keys():
                optimal_x_tka[key] = model_values[key]

            optimal_x_Nk = np.zeros(Nk_dims)
            model_values = model.getAttr("x", model_var_x_Nk)
            for key in sorted(model_values.keys()):
                optimal_x_Nk[key] = model_values[key]
            if deterministic:
                optimal_theta_tka = np.zeros(tka_dims)
                model_values = model.getAttr("x", model_var_theta_tka)
                for key in model_values.keys():
                    optimal_theta_tka[key] = model_values[key]
                result.append(
                    itlp_cpomdp.utils.FiniteConstrainedLpResult(
                        budget,
                        all_grid_points,
                        deterministic,
                        optimal_x_tka,
                        optimal_x_Nk,
                        optimal_theta_tka,
                        model.objVal,
                        elapsed_time=setup_time + timer_module.time() - timer_start,
                        optimize_time=optimize_time,
                    )
                )
            else:
                result.append(
                    itlp_cpomdp.utils.FiniteConstrainedLpResult(
                        budget,
                        all_grid_points,
                        deterministic,
                        optimal_x_tka,
                        optimal_x_Nk,
                        None,
                        model.objVal,
                        elapsed_time=setup_time + timer_module.time() - timer_start,
                        optimize_time=optimize_time,
                    )
                )
        else:
            logger.error(f"Could not optimize for budget = {budget}")
    return result


def lp_grid_upper_bound_dual(
    all_grid_points: Sequence[Sequence[float]],
    T: int,
    num_actions: int,
    num_states: int,
    all_F_values: np.ndarray,
    q_a_i: Sequence[Sequence[float]],
    all_final_expected_rewards: Sequence[float],
    budget_values: Sequence[float],
    all_costs: Sequence[Sequence[Sequence[float]]] = None,
    delta: Optional[Sequence[float]] = None,
) -> "List[itlp_cpomdp.utils.FiniteConstrainedLpResult]":
    """
    Compute the x_tka and x_Nk values for an array of budgets and for
    both deterministic and randomized policies.

    :param all_grid_points: The set of all belief states in our grid.
    :param T: The horizon for the problem.
    :param num_actions: The number of actions in the model.
    :param num_states: The number of belief states in the model.
    :param all_F_values: The F values,
            with the shape ((T + 1), num_actions, len(all_grid_points), len(all_grid_points)).
    :param q_a_i: The immediate rewards.
    :param all_final_expected_rewards: The terminal rewards
            for ending in a particular state.
    :param budget_values: The budgets to solve the problem for.
    :param all_costs: The cost at time t
            of taking the action at index a in the state at index i is
            all_costs[t][a][i]. Defaults to None.
    :param delta: The delta values.
            If None, set to uniform probability of all states. Defaults to None.
    :return: The results for each budget and deterministic value.
    """
    logger.info("Adding constraints to LP")
    setup_start = timer_module.time()
    if delta is None:
        delta = [1 / len(all_grid_points)] * len(all_grid_points)

    delta = itlp_cpomdp.utils.round_sum_to_val(delta)

    model = Model("lp_grid_upper_bound_dual")
    model.setParam("OutputFlag", 0)
    model.setParam("LogFile", "")
    model.setParam("LogToConsole", 0)

    grid_point_time_action_tuples = []
    for gp_index, gp in enumerate(all_grid_points):
        for t in range(T - 1):
            for action_index in range(num_actions):
                grid_point_time_action_tuples.append((t, gp_index, action_index))

    tka_dims = (T - 1, len(all_grid_points), num_actions)
    NK_dims = len(all_grid_points)
    model_var_x_tka = model.addVars(
        *(T - 1, len(all_grid_points), num_actions),
        lb=0,
        ub=GRB.INFINITY,
        name="x_tka",
        obj=1.0,
        vtype=GRB.CONTINUOUS,
    )
    model_var_x_Nk = model.addVars(
        len(all_grid_points),
        lb=0,
        ub=GRB.INFINITY,
        name="x_Nk",
        obj=1.0,
        vtype=GRB.CONTINUOUS,
    )

    # set objective
    expression = 0
    for time in range(T - 1):
        for action_index in range(num_actions):
            for grid_point_index, grid_point in enumerate(all_grid_points):
                for state_index in range(num_states):
                    expression += (
                        grid_point[state_index]
                        * q_a_i[action_index, state_index]
                        * model_var_x_tka[time, grid_point_index, action_index]
                    )

    for grid_point_index, grid_point in enumerate(all_grid_points):
        expression += (
            all_final_expected_rewards[grid_point_index]
            * model_var_x_Nk[grid_point_index]
        )

    model.setObjective(expression, GRB.MAXIMIZE)

    # add constraint 1
    for grid_point_index, grid_point in enumerate(all_grid_points):
        expression = 0
        for action_index in range(num_actions):
            expression += model_var_x_tka[0, grid_point_index, action_index]
        model.addConstr(expression == delta[grid_point_index])
    for grid_index_k, grid_k in enumerate(all_grid_points):
        for time in range(1, T - 1):
            expression = 0
            for action_index in range(num_actions):
                expression += model_var_x_tka[time, grid_index_k, action_index]
                for grid_index_l, grid_l in enumerate(all_grid_points):
                    f_val = all_F_values[
                        T - time + 1, action_index, grid_index_l, grid_index_k
                    ]
                    # adding a constraint is incredibly expensive; do not add
                    # if the coefficient is zero!
                    if f_val != 0:
                        expression -= (
                            f_val
                            * model_var_x_tka[time - 1, grid_index_l, action_index]
                        )
            model.addConstr(expression == 0)

    # add constraint 3
    for grid_index_k, grid_k in enumerate(all_grid_points):
        expression = model_var_x_Nk[grid_index_k]
        for action_index in range(num_actions):
            for grid_index_l, grid_l in enumerate(all_grid_points):
                f_val = all_F_values[2, action_index, grid_index_l, grid_index_k]
                if f_val != 0:
                    expression -= (
                        f_val * model_var_x_tka[T - 2, grid_index_l, action_index]
                    )
        model.addConstr(expression == 0)

    expression = 0
    for time in range(T - 1):
        for grid_point_index, grid_point in enumerate(all_grid_points):
            for action_index in range(num_actions):
                for state_index in range(num_states):
                    expression += (
                        grid_point[state_index]
                        * all_costs[time, action_index, state_index]
                        * model_var_x_tka[time, grid_point_index, action_index]
                    )
    logger.info("Verifying budgets")
    budget_values = verify_budgets(budget_values, model, expression)
    setup_time = timer_module.time() - setup_start
    budget_constraint = model.addConstr(expression <= budget_values[0])

    logger.info("Solving randomized case")
    result = solve_over_budgets(
        model,
        all_grid_points,
        model_var_x_tka,
        model_var_x_Nk,
        None,
        budget_values,
        budget_constraint,
        setup_time,
        tka_dims,
        NK_dims,
        deterministic=False,
    )
    return result
