"""Defines HS071 class."""
from typing import SupportsFloat, Tuple

import numpy as np
from numpy.typing import NDArray

from fdopt import NLPSolvable


# flake8: noqa: E731
# pylint: disable=line-too-long
class HS071(NLPSolvable):
    r"""Example implementation of ``NLPSolvable``.

    The purpose of the class is to provide an example of a proper
    :py:class:`NLPSolvable` implementation. The NLP problem implemented here is
    `Hock & Schittkowski Problem #71 <http://apmonitor.com/wiki/index.php/Apps/HockSchittkowski>`,
    that is often used for testing optimization software. The problem is
    defined as:

    Minimize:

    .. math::
       x_1 x_4\left ( x_1 + x_2 + x_3 \right ) + x_3

    subject to:

    .. math::
       x_1 x_2 x_3 x_4 \geqslant 25\\
       x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40\\
       1 \leqslant x_1, x_2, x_3, x_4 \leqslant 5

    The initial variable values are:

    .. math::
       x_0 = \left ( 1, 5, 5, 1 \right )

    The first equation from above (function that should be minimized) is
    actually *objective function*, and should be implemented in
    :py:func:`objective` method.

    Section *subject to* from above represents constraints:

    * The first line is a nonlinear inequality constraint;
    * the second line is a nonlinear equality constraint;
    * and the third line contains bounds, which are linear constraints that
      should be implemented in :py:attr:`x_lower` and :py:attr:`x_upper`
      properties.

    Nonlinear constraints from the list above should be implemented in
    :py:func:`nonlinear_constraints` method.
    """

    @property
    def x_count(self) -> int:
        """Get number of variables.

        Returns:
            The number of variables, which is ``4`` in this problem.
        """
        return 4

    @property
    def x_lower(self) -> NDArray[np.float_]:
        r"""Get lower bounds.

        Bounds are defined by:

        .. math::
           1 \leqslant x_1, x_2, x_3, x_4 \leqslant 5

        It means that lower bounds are:

        .. math::
           x_{lower} = \left ( 1, 1, 1, 1 \right )

        Returns:
            Vector ``(1, 1, 1, 1)``.
        """
        return np.array([1, 1, 1, 1])

    @property
    def x_upper(self) -> NDArray[np.float_]:
        r"""Get upper bounds.

        Bounds are defined by:

        .. math::
           1 \leqslant x_1, x_2, x_3, x_4 \leqslant 5

        It means that upper bounds are:

        .. math::
           x_{upper} = \left ( 5, 5, 5, 5 \right )

        Returns:
            Vector ``(5, 5, 5, 5)``.
        """
        return np.array([5, 5, 5, 5])

    @property
    def lin_inequality_constraints(self) \
            -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Get linear inequality constraints.

        The problem does not define any linear inequality constraints, except
        for bounds which are already covered in :py:attr:`x_lower` and
        :py:attr:`x_upper`, so this method will simply return two empty arrays.

        Returns:
            Two empty arrays.
        """
        return np.zeros((0, 0)), np.zeros((0, 0))

    @property
    def lin_equality_constraints(self) \
            -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Get linear equality constraints.

        The problem does not define any linear equality constraints, so this
        method will simply return two empty arrays.

        Returns:
            Two empty arrays.
        """
        return np.zeros((0, 0)), np.zeros((0, 0))

    @property
    def implements_gradient(self) -> bool:
        """Indicates if gradient is implemented.

        Returns:
            ``True`` since :py:func:`gradient` is implemented in this class.
        """
        return False

    @property
    def implements_hessian(self) -> bool:
        """Indicates if hessian is implemented.

        If the derived class overrides :py:func:`hessian` method, and
        implements it properly, this property should return ``True``.
        Otherwise, the property must return ``False``.

        Returns:
            ``True`` since :py:func:`hessian` is implemented in this class.
        """
        return False

    @property
    def nonlin_ineq_constraints_count(self) -> int:
        """Get nonlinear inequality constraints count.

        The problem defines one nonlinear inequality constraint, so this method
        will return ``1``.

        Returns:
            ``1``
        """
        return 1

    @property
    def nonlin_eq_constraints_count(self) -> int:
        """Get nonlinear equality constraints count.

        The problem defines one nonlinear equality constraint, so this method
        will return ``1``.

        Returns:
            ``1``
        """
        return 1

    def nonlinear_constraints(self, x: NDArray[np.float_]) \
            -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        r"""Get nonlinear constraint values at point ``x``.

        Nonlinear inequality constraint is:

        .. math::
           x_1 x_2 x_3 x_4 \geqslant 25

        We need to rewrite it so tha it is *less than or equal to zero*:

        .. math::
           25 - x_1 x_2 x_3 x_4 \leqslant 0

        So we can write nonlinear inequality constraint function as:

        .. math::
           c_{neq_1}(\vec{x}) = 25 - x_1 x_2 x_3 x_4

        Note that we've used index ``1`` above, meaning that it's just the
        first nonlinear inequality constraint. In this particular case we have
        only one defined anyway, so that's it. In general, we can have more
        than one, though.

        Now we can do the similar for nonlinear equality constraint defined by:

        .. math::
           x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40

        We need to rewrite it so tha it is *equal to zero*:

        .. math::
           x_1^2 + x_2^2 + x_3^2 + x_4^2 - 40 = 0

        So we can write nonlinear equality constraint function as:

        .. math::
           c_{eq_1}(\vec{x}) = x_1^2 + x_2^2 + x_3^2 + x_4^2 - 40

        Args:
            x: Vector of variable values.

        Returns:
            :Tuple that contains two vectors:

            - Vector of nonlinear inequality constraint values, with number of
              elements equal to the number of nonlinear inequality constraints
              defined. In the current problem it is only one element.
            - Vector of nonlinear equality constraint values, with number of
              elements equal to the number of nonlinear equality constraints
              defined. In the current problem it is only one element.
        """
        c_neq = np.array([25 - x[0] * x[1] * x[2] * x[3]])
        c_eq = np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40])
        return c_neq, c_eq

    def nonlinear_constraints_jacobian(self, x: NDArray[np.float_]) \
            -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        r"""Calculate Jacobian matrices of nonlinear constraints.

        It is already explained inn :py:func:`nonlinear_constraints`
        documentation that nonlinear inequality constraint function is:

        .. math::
           c_{neq_1}(\vec{x}) = 25 - x_1 x_2 x_3 x_4

        Here we need to calculate its Jacobian as:

        .. math::
           \begin{aligned}
           \nabla c_{neq_1} &= \begin{bmatrix}
             \frac{\partial c_{neq_1}}{\partial x_1} &
             \frac{\partial c_{neq_1}}{\partial x_2} &
             \frac{\partial c_{neq_1}}{\partial x_3} &
             \frac{\partial c_{neq_1}}{\partial x_4}
           \end{bmatrix} \\
           &= \begin{bmatrix}
           -x_2 x_3 x_4 & -x_1 x_3 x_4 & -x_1 x_2 x_4 & -x_1 x_2 x_3
           \end{bmatrix}\end{aligned}

        We should do the same for nonlinear equality constraint function
        (explained in :py:func:`nonlinear_constraints`):

        .. math::
           c_{eq_1}(\vec{x}) = x_1^2 + x_2^2 + x_3^2 + x_4^2 - 40

        Its Jacobian matrix is:

        .. math::
           \begin{aligned}
           \nabla c_{eq_1} &= \begin{bmatrix}
             \frac{\partial c_{eq_1}}{\partial x_1} &
             \frac{\partial c_{eq_1}}{\partial x_2} &
             \frac{\partial c_{eq_1}}{\partial x_3} &
             \frac{\partial c_{eq_1}}{\partial x_4}
           \end{bmatrix} \\
           &= \begin{bmatrix}
           2 x_1 & 2 x_2 & 2 x_3 & 2 x_4
           \end{bmatrix}\end{aligned}

        Args:
            x: Vector of variable values.

        Returns:
            :Tuple that contains two matrices:

            - Jacobian matrix of nonlinear inequality constraints, whose umber
              of rows is equal to the number of nonlinear inequality
              constraints defined (in our case one), and number of columns
              equal to :py:attr:`x_count`.
            - Jacobian matrix of nonlinear equality constraints, whose umber of
              rows is equal to the number of nonlinear equality constraints
              defined (in our case one), and number of columns equal to
              :py:attr:`x_count`.
        """
        jac_neq = np.array([
            - x[1] * x[2] * x[3],
            - x[0] * x[2] * x[3],
            - x[0] * x[1] * x[3],
            - x[0] * x[1] * x[2],
        ])
        jac_eq = np.array([2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]])
        return jac_neq, jac_eq

    def objective(self, x: NDArray[np.float_]) -> float:
        r"""Calculate the objective function value at point ``x``.

        The objective function is:

        .. math::
           f(x) = x_1 x_4\left ( x_1 + x_2 + x_3 \right ) + x_3

        Args:
            x: Vector of variable values.

        Returns:
            The objective function value at point ``x``.
        """
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def gradient(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        r"""Calculate gradient of the objective function at point ``x``.

        The objective function is:

        .. math::
           f(x) = x_1 x_4\left ( x_1 + x_2 + x_3 \right ) + x_3

        Its gradient is:

        .. math::
           \nabla f(x) = \begin{bmatrix}
             \frac {\partial f}{\partial x_1} \\
             \frac {\partial f}{\partial x_2} \\
             \frac {\partial f}{\partial x_3} \\
             \frac {\partial f}{\partial x_4}
           \end{bmatrix} = \begin{bmatrix}
             x_4 \left ( 2 x_1 + x_2 + x_3 \right ) \\
             x_1 x_4 \\
             x_1 x_4 + 1 \\
             x_1 \left ( x_1 + x_2 + x_3 \right )
           \end{bmatrix}

        Args:
            x: Vector of variable values.

        Returns:
            Objective function gradient at point ``x``. It's a vector with
            number of elements equal to :py:attr:`x_count`.
        """
        return np.array([
            [x[3] * (2 * x[0] + x[1] + x[2])],
            [x[0] * x[3]],
            [x[0] * x[3] + 1.0],
            [x[0] * (x[0] + x[1] + x[2])],
        ])

    def hessian(self,
                x: NDArray[np.float_],
                sigma: SupportsFloat,
                lambda_ineq: NDArray[np.float_],
                lambda_eq: NDArray[np.float_]
                ) -> NDArray[np.float_]:
        r"""Calculate Hessian of Lagrangian.

        Let's start from the Hessian of the objective function, which is
        actually matrix of second partial derivatives:

        .. math::
           h = \begin{bmatrix}
             \frac{\partial^2 f}{ \partial x_1 ^2} &
             \frac{\partial^2 f}{\partial x_1 \partial x_2} &
             \frac{\partial^2 f}{\partial x_1 \partial x_3} &
             \frac{\partial^2 f}{\partial x_1 \partial x_4}
             \\
             \frac{\partial^2 f}{\partial x_1 \partial x_2} &
             \frac{\partial^2 f}{ \partial x_2 ^2} &
             \frac{\partial^2 f}{\partial x_2 \partial x_3} &
             \frac{\partial^2 f}{\partial x_2 \partial x_4}
             \\
             \frac{\partial^2 f}{\partial x_1 \partial x_3} &
             \frac{\partial^2 f}{\partial x_2 \partial x_3} &
             \frac{\partial^2 f}{ \partial x_3 ^2} &
             \frac{\partial^2 f}{\partial x_3 \partial x_4}
             \\
             \frac{\partial^2 f}{\partial x_1 \partial x_4} &
             \frac{\partial^2 f}{\partial x_2 \partial x_4} &
             \frac{\partial^2 f}{\partial x_3 \partial x_4} &
             \frac{\partial^2 f}{ \partial x_4 ^2}
            \end{bmatrix}

        Since we already have the first partial derivatives calculated in
        :py:func:`gradient`, we can calculate the second order derivatives:

        .. math::
           h = \begin{bmatrix}
             2 x_4 & x_4 & x_4 & 2 x_1 + x_2 + x_3 \\
             x_4 & 0 & 0 & x_1 \\
             x_4 & 0 & 0 & x_1 \\
             2 x_1 + x_2 + x_3 & x_1 & x_1 & 0
            \end{bmatrix}

        This is Hessian of the objective function, but here we need Hessian of
        Lagrangian. For this reason we need to calculate the second partial
        derivatives of nonlinear inequality constraints, and nonlinear equality
        constraints. These are calculated in the same exact way as Hessian of
        objective above, except that instead of using objective function now we
        will use nonlinear constraint functions.

        Second partial derivatives of nonlinear inequality constraints:

        .. math::
           h_{ineq} = \begin{bmatrix}
             0 & - x_3 x_4 & - x_2 x_4 & - x_2 x_3 \\
             - x_3 x_4 & 0 & - x_1 x_4 & - x_1 x_3 \\
             - x_2 x_4 & - x_1 x_4 & 0 & - x_1 x_2 \\
             - x_2 x_3 & - x_1 x_3 & - x_1 x_2 & 0
           \end{bmatrix}

        Second partial derivatives of nonlinear equality constraints:

        .. math::
           h_{eq} = \begin{bmatrix}
             2 & 0 & 0 & 0 \\
             0 & 2 & 0 & 0 \\
             0 & 0 & 2 & 0 \\
             0 & 0 & 0 & 2 \\
           \end{bmatrix}

        The last step is to multiply these three matrices with proper
        coefficients, and sum them up. The first matrix should be multiplied by
        ``sigma``, the second by ``lambda_ineq``, and the third one by
        ``lambda_eq``.

        Finally note that all three matrices are symmetrical, meaning that
        their sum will also be symmetrical (as Hessian always is).

        Args:
            x: Vector of variable values.
            sigma: Coefficient for objective function Hessian.
            lambda_ineq: Vector of Lagrangian coefficients of nonlinear
                inequality constraints. Since there's only one nonlinear
                inequality constraint, this vector will have only one element.
            lambda_eq: Vector of Lagrangian coefficients of nonlinear equality
                constraints. Since there's only one nonlinear equality
                constraint, this vector will have only one element.

        Returns:
            Hessian of Lagrangian, in a form of a lower triangular square
            matrix with number of rows and columns equal to :py:attr:`x_count`.
        """
        hess = sigma * np.array([
            [2 * x[3], x[3], x[3], 2 * x[0] + x[1] + x[2]],
            [x[3], 0, 0, x[0]],
            [x[3], 0, 0, x[0]],
            [2 * x[0] + x[1] + x[2], x[0], x[0], 0]
        ])
        hess += lambda_ineq[0] * np.array([
            [0, - x[2] * x[3], - x[1] * x[3], - x[1] * x[2]],
            [- x[2] * x[3], 0, - x[0] * x[3], - x[0] * x[2]],
            [- x[1] * x[3], - x[0] * x[3], 0, - x[0] * x[1]],
            [- x[1] * x[2], - x[0] * x[2], - x[0] * x[1], 0]
        ])
        hess += lambda_eq[0] * np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2]
        ])
        return hess

    def intermediate(self, **kwargs) -> bool:
        """Process intermediate status.

        Simply prints ``**kwargs``.

        Args:
            **kwargs: Arguments provided by actual solvers used, and can be
                different for different solvers.

        Returns:
            ``True``
        """
        print(**kwargs)
        return True
