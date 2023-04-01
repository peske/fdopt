"""NLPSolvable abstract class."""
import abc
from typing import SupportsFloat, Tuple

import numpy as np
from numpy.typing import NDArray


class NLPSolvable(abc.ABC):
    r"""Abstract class that defines NLP (nonlinear programming) problems.

    The essence of NLP problem is to minimize some *objective function* in the
    variable scope defined by some *constraints*. Constraints can be *linear*
    and *nonlinear*.

    Linear constraints are represented in this class in the following way:

    - *Lower bounds*, exposed via :py:attr:`x_lower` property, defined as:

    .. math::
       x \geqslant x_{lower}

    - *Upper bounds*, exposed via :py:attr:`x_upper` property, defined as:

    .. math::
       x \leqslant x_{upper}

    - *Linear inequality constraints*, exposed via
      :py:attr:`lin_inequality_constraints` property, defined as:

    .. math::
       A_{lte} \times x \leqslant b_{lte}

    - *Linear equality constraints*, exposed via
      :py:attr:`lin_equality_constraints` property, defined as:

    .. math::
       A_{eq} \times x = b_{eq}

    Nonlinear constraints are exposed via :py:func:`nonlinear_constraints`
    method.

    The objective function is exposed via :py:func:`objective` method.

    Examples:
        .. code-block:: python

           from FDOpt import NLPSolvable


           class MyNLPProblem(NLPSolvable):
               # ...implementation goes here...
               pass
    """

    @property
    @abc.abstractmethod
    def x_count(self) -> int:
        """Get the number of variables / the length of ``x`` vector.

        Returns:
            Number of variables.
        """

    @property
    @abc.abstractmethod
    def x_lower(self) -> NDArray[np.float_]:
        r"""Get the vector of lower bounds (minimal values) of variables.

        This constraint is defined as:

        .. math::
           x \geqslant x_{lower}

        Returns:
            Vector with :attr:`x_count` elements which are representing minimal
            values of the corresponding variables, or an empty ``ndarray`` if
            no such a constraint is defined.
        """

    @property
    @abc.abstractmethod
    def x_upper(self) -> NDArray[np.float_]:
        r"""Get the vector of upper bounds (maximal values) of variables.

        This constraint is defined as:

        .. math::
           x \leqslant x_{upper}

        Returns:
            Vector with :attr:`x_count` elements which are representing maximal
            values of the corresponding variables, or an empty ``ndarray`` if
            no such a constraint is defined.
        """

    @property
    @abc.abstractmethod
    def lin_inequality_constraints(self) \
            -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        r"""Get linear inequality constraints.

        Property returns matrix ``A_lte`` and vector ``b_lte``. The number of
        rows in ``A_lte`` is equal to the number of elements in ``b_lte``, and
        it is the number of inequality constraints. The number of columns in
        ``A_lte`` is equal to :py:attr:`x_count`. These constraints are defined
        as:

        .. math::
           A_{lte} \times x \leqslant b_{lte}

        Returns:
            :Tuple that contains two elements:

            - matrix ``A_lte`` or empty ``ndarray`` if linear inequality
              constraints are not defined;
            - vector ``b_lte`` or empty ``ndarray`` if linear inequality
              constraints are not defined.
        """

    @property
    @abc.abstractmethod
    def lin_equality_constraints(self) \
            -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        r"""Get linear equality constraints.

        Property returns matrix ``A_eq`` and vector ``b_eq``. The number of
        rows in ``A_eq`` is equal to the number of elements in ``b_eq``, and
        it is the number of equality constraints. The number of columns in
        ``A_eq`` is equal to :py:attr:`x_count`. These constraints are defined
        as:

        .. math::
           A_{eq} \times x = b_{eq}

        Returns:
            :Tuple that contains two elements:

            - matrix ``A_eq`` or empty ``ndarray`` if linear equality
              constraints are not defined;
            - vector ``b_eq`` or empty ``ndarray`` if linear inequality
              constraints are not defined.
        """

    @property
    def implements_gradient(self) -> bool:
        """Indicates if gradient is implemented.

        If the derived class overrides :py:func:`gradient` method, and
        implements it properly, this property should return ``True``.
        Otherwise, the property must return ``False``.

        Returns:
            ``True`` if gradient is implemented, ``False`` otherwise.
        """
        return False

    @property
    def implements_hessian(self) -> bool:
        """Indicates if hessian is implemented.

        If the derived class overrides :py:func:`hessian` method, and
        implements it properly, this property should return ``True``.
        Otherwise, the property must return ``False``.

        Returns:
            ``True`` if hessian is implemented, ``False`` otherwise.
        """
        return False

    @property
    @abc.abstractmethod
    def nonlin_ineq_constraints_count(self) -> int:
        """Get nonlinear inequality constraints count.

        Returns:
            Nonlinear inequality constraints count.
        """

    @property
    @abc.abstractmethod
    def nonlin_eq_constraints_count(self) -> int:
        """Get nonlinear equality constraints count.

        Returns:
            Nonlinear equality constraints count.
        """

    # pylint: disable=invalid-name
    @abc.abstractmethod
    def nonlinear_constraints(self, x: NDArray[np.float_]) \
            -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Get nonlinear constraints at point ``x``.

        Args:
            x: Vector of variable values for which the constraints should be
                calculated. It has :py:attr:`x_count` elements.

        Returns:
            :Tuple that contains four ``ndarray`` instances:

            - Vector of nonlinear inequality constraint values at point ``x``.
              The size of the vector is equal to the number of nonlinear
              inequality constraints defined. If no such constraints are
              defined, it will be an empty ``ndarray`` instance.
            - Vector of nonlinear equality constraint values at point ``x``.
              The size of the vector is equal to the number of nonlinear
              equality constraints defined. If no such constraints are
              defined, it will be an empty ``ndarray`` instance.
        """

    @abc.abstractmethod
    def nonlinear_constraints_jacobian(self, x: NDArray[np.float_]) \
            -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Get Jacobian matrices of nonlinear constraints at point ``x``.

        *Jacobian matrix* of a vector-valued function is the matrix of all its
        first-order partial derivatives.

        Args:
            x: Vector of variable values, with :py:attr:`x_count` elements.

        Returns:
            :Tuple that contains two matrices:

            - Jacobian matrix of nonlinear inequality constraints. Number of
              rows is equal to the number of nonlinear inequality constraints
              defined, while number of columns is equal to :py:attr:`x_count`.
              If no nonlinear inequality constraints are defined, it will be an
              empty ``ndarray`` instance.
            - Jacobian matrix of nonlinear equality constraints. Number of rows
              is equal to the number of nonlinear equality constraints defined,
              while number of columns is equal to :py:attr:`x_count`. If no
              nonlinear equality constraints are defined, it will be an empty
              ``ndarray`` instance.
        """

    @abc.abstractmethod
    def objective(self, x: NDArray[np.float_]) -> float:
        """Calculate objective function value at point ``x``.

        This method is actually *objective function* - the function whose
        return value solvers should minimize.

        Args:
            x: Vector of variable values, with :py:attr:`x_count` elements.

        Returns:
            Objective function value at point ``x``.
        """

    def gradient(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Calculate gradient of the objective function at point ``x``.

        Note that this method does not have to be implemented. If it is not
        implemented, the solvers will be able to calculate some approximation
        of it. But if it is properly implemented, it can significantly speed up
        the optimization process, and sometimes even somewhat improve the
        solution.

        .. note::
           If this method is overriden and properly implemented,
           :py:attr:`implements_gradient` property should also be overriden to
           return ``True``.

        Args:
            x: Vector of variable values, with :py:attr:`x_count` elements.

        Raises:
            NotImplementedError: if the method is not implemented, in which
                case :py:attr:`implements_gradient` has to return ``False``.

        Returns:
            Gradient of the objective function. It's a vector with
            :py:attr:`x_count` elements.
        """
        raise NotImplementedError

    # flake8: noqa: E731
    # pylint: disable=line-too-long
    def hessian(self,
                x: NDArray[np.float_],
                sigma: SupportsFloat,
                lambda_ineq: NDArray[np.float_],
                lambda_eq: NDArray[np.float_]
                ) -> NDArray[np.float_]:
        """Calculate analytical Hessian at point ``x``.

        The method should return *Hessian of Lagrangian*. It is not a simple
        topic that can be explained here, so please do some research online.
        For start here are some implementations:

        - https://github.com/mechmotum/cyipopt/blob/master/examples/hs071.py
        - https://www.mathworks.com/help/optim/ug/fmincon-interior-point-algorithm-with-analytic-hessian.html
        - https://github.com/jonathancurrie/OPTI/blob/master/Examples/NonlinearProgramming.m

        Note that this method does not have to be implemented. If it is not
        implemented, the solvers will be able to calculate some approximation
        of it. But if it is properly implemented, it can significantly speed up
        the optimization process, and sometimes even somewhat improve the
        solution.

        .. note::
           If this method is overriden and properly implemented,
           :py:attr:`implements_hessian` property should also be overriden to
           return ``True``.

        .. note::
           Implementing this method without implementing :py:func:`gradient`
           doesn't make sense, so if gradient isn't implemented, this method
           will be ignored no matter :py:attr:`implements_hessian` value.

        Args:
            x: Vector of variable values, with length :py:attr:`x_count`.
            sigma: Coefficient for objective function Hessian.
            lambda_ineq: Vector of Lagrangian coefficients of nonlinear
                inequality constraints.
            lambda_eq: Vector of Lagrangian coefficients of nonlinear equality
                constraints.

        Raises:
            NotImplementedError: if the method is not implemented, in which
                case :py:attr:`implements_hessian` has to return ``False``.

        Returns:
            Hessian matrix. It is a square matrix with :py:attr:`x_count` rows.
        """
        raise NotImplementedError

    # pylint: disable=unused-argument
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def intermediate(self, **kwargs) -> bool:
        """Callback method that is called once per iteration.

        The method can be used to obtain information about the optimization
        status while solving. It also can be used to break the process: if this
        method returns ``False`` solving will be terminated.

        Args:
            **kwargs: Arguments provided by actual solvers used, and can be
                different for different solvers.

        Returns:
            ``False`` if solving process should be aborted, ``True`` otherwise.
        """
        return True
