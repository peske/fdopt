import cyipopt
import numpy as np
from numpy.typing import NDArray

from ._nlp_solvable import NLPSolvable


class IPOptProblemCore:
    """``NLPSolvable`` wrapper when gradient isn't implemented."""

    def __init__(self, solvable: NLPSolvable):
        self.solvable = solvable

        _, b = solvable.lin_inequality_constraints
        if b:
            self.lin_ineq_count = b.size
        else:
            self.lin_ineq_count = 0

        _, b = solvable.lin_equality_constraints
        if b:
            self.lin_eq_count = b.size
        else:
            self.lin_eq_count = 0

    def objective(self, x: NDArray) -> float:
        return self.solvable.objective(x)

    def constraints(self, x: NDArray) -> NDArray:
        cons = np.empty(self.lin_ineq_count + self.lin_eq_count +
                        self.solvable.nonlin_ineq_constraints_count +
                        self.solvable.nonlin_eq_constraints_count,
                        dtype='float64')

        start: int = 0

        # Linear inequality constraints:
        if self.lin_ineq_count:
            a, _ = self.solvable.lin_inequality_constraints
            np.matmul(a, x, out=cons[:self.lin_ineq_count])
            start += self.lin_ineq_count

        # Linear equality constraints:
        if self.lin_eq_count:
            a, _ = self.solvable.lin_equality_constraints
            np.matmul(a, x, out=cons[start: start + self.lin_eq_count])
            start += self.lin_eq_count

        # Nonlinear constraints:
        if self.solvable.nonlin_ineq_constraints_count > 0 or\
                self.solvable.nonlin_eq_constraints_count > 0:
            c_ineq, c_eq = self.solvable.nonlinear_constraints(x)

            # inequality:
            if self.solvable.nonlin_ineq_constraints_count:
                cons[start: start +
                     self.solvable.nonlin_ineq_constraints_count] = c_ineq
                start += self.solvable.nonlin_ineq_constraints_count

            # equality:
            if self.solvable.nonlin_eq_constraints_count:
                cons[start:] = c_eq

        return cons

    def intermediate(self, **kwargs) -> bool:
        return self.solvable.intermediate(**kwargs)


class IPOptProblemGrad(IPOptProblemCore):
    """``NLPSolvable`` wrapper when gradient is implemented, but Hessian isn't.
    """

    def __init__(self, solvable: NLPSolvable):
        super().__init__(solvable)

    def gradient(self, x: NDArray) -> NDArray:
        grad: NDArray = np.empty((self.lin_ineq_count +
                                  self.lin_eq_count +
                                  self.solvable.nonlin_ineq_constraints_count +
                                  self.solvable.nonlin_eq_constraints_count,
                                  self.solvable.x_count), dtype='float64')

        return grad


class IPOptProblemHess(IPOptProblemGrad):
    """``NLPSolvable`` wrapper when both gradient and Hessian are implemented.
    """

    def __init__(self, solvable: NLPSolvable):
        super().__init__(solvable)


def ipopt_solve(solvable: NLPSolvable, x0: NDArray):
    cyipopt.minimize_ipopt()
    pass
