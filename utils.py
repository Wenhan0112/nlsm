"""
utils.py:
Content:
	class Fitter
	class WeightedLinearRegressor
	class FunctionValidator
	function weighted_nonlinear_regression
	function interpolate_array

Author: Wenhan Sun
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import inspect
import scipy.optimize
from typing import Tuple, Optional, Union, Callable, List
from abc import ABC


class Fitter(ABC):
	"""
	An abstract fitter class for all function fitters and validtors.
	"""

	def __init__(self):
		self._fit = False

	def fit(self, x, x_err, y, y_err):
		raise RuntimeError(f"Fitting is not implemented for {self}. ")


class WeightedLinearRegressor(Fitter):
	"""
	A linear weighted regressor.
	@field biased (bool): True iff the model is biased.
	@field n (int): The number of training samples. It has this attribute iff
		the regressor is fitted.
	@field x (numpy.ndarray): An 1d-array of all the training data. It has this
		attribute iff the regressor is fitted.
	@field y (numpy.ndarray): An 1d-array of all the training labels in order
		with the training samples. It has this attribute iff the regressor is
		fitted.
	@field _error_specified (bool): True iff the error of the model is
		specified. It has this attribute iff the regressor is fitted.
	@field slope (float): The slope of the fitted model.
	@field slope_err (float): The errof the slope of the fitted model. It has
		this attribute iff the error of the data is specified.
	@field intercept (float): The intercept of the fitted model. It has this
		attribute iff the model is biased.
	@field intercept_err (float): The error of the intercept of the fitted
		model. It has this attribute iff the model is biased and the error is
		specified.
	@field y_est (numpy.ndarray): An 1d-array of all the estimated labels in
		order with the training samples.
	@field y_res (numpy.ndarray): An 1d-array of all the residuals in order
		with the training samples.
	@field y_est (numpy.ndarray): An 1d-array of all the relative residuals in
		order with the training samples.
	@field weights (numpy.ndarray): An 1d-array of all the weights in order
	"""

	def __init__(self, biased: bool = True):
		"""
		The constructor of the weighted linear regressor.
		"""
		super().__init__()
		self.biased = biased

	def fit(self, x: np.ndarray, x_err: Optional[np.ndarray],
			y: np.ndarray, y_err: Optional[np.ndarray]) -> None:
		"""
		Fit the weighted linear regressor based on the inputs.
		If no error is specified and with bias, the unweighted linear regressor
		with bias is fitted.
		If no error is specified and without bias, the unweighted linear
		regressor without bias is fitted.
		If error are specified and with bias, the weighted linear regressor
		with bias is fitted.
		If error are specified and without bias, the weighted linear regressor
		without bias is fitted.
		@params x (numpy.ndarray): The array of the independent variable data
			with shape (n,).
		@params x_err (Optional[numpy.ndarray]): The array of the error of the
			independent variable in sequence with x with shape (n,).
		@params y (numpy.ndarray): The array of the dependent variable data in
			sequence with x with shape (n,).
		@params y_err (Optional[numpy.ndarray]): The array of the error of the
			dependent variable in sequence with x with shape (n,).
		"""
		assert x.ndim == 1, \
			"The independent variable data array should have a dimension " + \
			f"1, but it is {x.ndim}. "
		assert y.shape == x.shape, \
			"The indepdent variable data array and the dependent variable " + \
			"data array should have the same shape, but they have shape " + \
			f"{x.shape} and {y.shape}, respectively. "
		self.n = len(x)
		self.x = x
		self.y = y

		if x_err is None and y_err is None:
			self._error_specified = False
			if self.biased:
				self._fit_biased_simple(x, y)
			else:
				self._fit_unbiased_simple(x, y)
		else:
			assert x_err is None or x_err.shape == x.shape, \
				"The independent variable data array and the independent " + \
				"variable error array should have the same shape if " + \
				f"specified, but they have shape {x.shape} and " + \
				f"{x_err.shape}, respectively. "
			assert y_err is None or y_err.shape == y.shape, \
				"The dependent variable data array and the dependent " + \
				"variable error array should have the same shape if " + \
				f"specified, but they have shape {y.shape} and " + \
				f"{y_err.shape}, respectively. "
			self._error_specified = True
			if self.biased:
				self._fit_biased(x, x_err, y, y_err)
			else:
				self._fit_unbiased(x, x_err, y, y_err)

		self._fit = True

	def _fit_biased_simple(self, x: np.ndarray, y: np.ndarray):
		"""
        The unweighted linear regression with bias for N samples.
		@params x (numpy.ndarray): The array of the independent variable data
			with shape (n,).
		@params y (numpy.ndarray): The array of the dependent variable data in
			sequence with x with shape (n,). The shape consistency is enforced
			by the caller function.
        """
		xy_avg = np.average(x * y)
		x_avg = np.average(x)
		y_avg = np.average(y)
		x2_avg = np.average(x ** 2)
		self.slope = (xy_avg - x_avg*y_avg)/(x2_avg - x_avg**2)
		self.intercept = (x2_avg*y_avg - x_avg*xy_avg)/(x2_avg - x_avg**2)
		self.y_est = x * self.slope + self.intercept
		self.y_res = y - self.y_est


	def _fit_biased(self, x: np.ndarray, x_err: Optional[np.ndarray],
			y: np.ndarray, y_err: Optional[np.ndarray]) -> None:
		"""
        The weighted linear regression with bias for N samples.
		@params x (numpy.ndarray): The array of the independent variable data
			with shape (n,).
		@params x_err (Optional[numpy.ndarray]): The array of the error of the
			independent variable in sequence with x with shape (n,). The shape
			consistency is enforced by the caller function.
		@params y (numpy.ndarray): The array of the dependent variable data in
			sequence with x with shape (n,). The shape consistency is enforced
			by the caller function.
		@params y_err (Optional[numpy.ndarray]): The array of the error of the
			dependent variable in sequence with x with shape (n,). Either x_err
			or y_err must not be None. This is enforced by the caller function.
			The shape consistency is enforced by the caller function.
		@error ValueError: Error if infinite weights are encountered.
        """

		if x_err is not None:
			xy_avg = np.average(x * y)
			x_avg = np.average(x)
			y_avg = np.average(y)
			x2_avg = np.average(x ** 2)
			slope = (xy_avg - x_avg*y_avg)/(x2_avg - x_avg**2)
			weights_inv = (slope * x_err)**2
			if y_err is not None:
				weights_inv += y_err ** 2
		else:
			weights_inv = y_err ** 2
		if np.any(weights_inv == 0):
			raise ValueError("Inifite weight encountered. ")
		weights = 1 / weights_inv
		self.weights = weights

		xy_avg = np.average(x * y, weights = weights)
		x_avg = np.average(x, weights = weights)
		y_avg = np.average(y, weights = weights)
		x2_avg = np.average(x ** 2, weights = weights)

		self.slope = (xy_avg - x_avg*y_avg)/(x2_avg - x_avg**2)
		self.intercept = (x2_avg*y_avg - x_avg*xy_avg)/(x2_avg - x_avg**2)
		self.y_est = x * self.slope + self.intercept
		self.y_res = y - self.y_est
		self.y_res_r = self.y_res * np.sqrt(weights)
		self.slope_err = np.sqrt((1/np.sum(weights)) / (x2_avg-x_avg**2))
		self.intercept_err = self.slope_err * np.sqrt(x2_avg)

	def _fit_unbiased_simple(self, x: np.ndarray, y: np.ndarray):
		"""
        The unweighted linear regression without bias for N samples.
		@params x (numpy.ndarray): The array of the independent variable data
			with shape (n,).
		@params y (numpy.ndarray): The array of the dependent variable data in
			sequence with x with shape (n,). The shape consistency is enforced
			by the caller function.
        """
		xy_avg = np.average(x * y)
		x_avg = np.average(x)
		y_avg = np.average(y)
		x2_avg = np.average(x ** 2)
		self.slope = xy_avg/x2_avg
		self.y_est = x * self.slope
		self.y_res = y - self.y_est

	def _fit_unbiased(self, x: np.ndarray, x_err: Optional[np.ndarray],
			y: np.ndarray, y_err: Optional[np.ndarray]) -> None:
		"""
		The weighted linear regression without bias for N samples.
		@params x (numpy.ndarray): The array of the independent variable data
			with shape (n,).
		@params x_err (Optional[numpy.ndarray]): The array of the error of the
			independent variable in sequence with x with shape (n,). The shape
			consistency is enforced by the caller function.
		@params y (numpy.ndarray): The array of the dependent variable data in
			sequence with x with shape (n,). The shape consistency is enforced
			by the caller function.
		@params y_err (Optional[numpy.ndarray]): The array of the error of the
			dependent variable in sequence with x with shape (n,). Either x_err
			or y_err must not be None. This is enforced by the caller function.
			The shape consistency is enforced by the caller function.
		@error ValueError: Error if infinite weight encountered.
		"""

		if x_err is not None:
			xy_avg = np.average(x * y)
			x_avg = np.average(x)
			y_avg = np.average(y)
			x2_avg = np.average(x ** 2)
			slope = xy_avg/x2_avg
			weights_inv = (slope * x_err)**2
			if y_err is not None:
				weights_inv += y_err ** 2
		else:
			weights_inv = y_err ** 2
		if np.any(weights_inv == 0):
			raise ValueError("Inifite weight encountered. ")
		weights = 1 / weights_inv
		self.weights = weights

		xy_avg = np.average(x * y, weights=weights)
		x_avg = np.average(x, weights=weights)
		y_avg = np.average(y, weights=weights)
		x2_avg = np.average(x ** 2, weights=weights)

		self.slope = xy_avg/x2_avg
		self.y_est = x * self.slope
		self.y_res = y - self.y_est
		self.y_res_r = self.y_res * np.sqrt(weights)
		self.slope_err = np.sqrt((1/np.sum(weights)) / x2_avg)

	def predict(self, x: Union[float, np.ndarray],
			x_err: Optional[Union[float, np.ndarray]] = None) \
			-> Union[Union[float, np.ndarray],
			Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]:
		"""
		Predict the indepdent variable based on the dependent variable value.
		@params x (Union[float, numpy.ndarray]): The input data.
		@params x_err (Optional[Union[float, numpy.ndarray]], default None):
			The input error. If both the error and the data are numpy.ndarray,
			then they must have the same shape. If the error is None, then the
			error of the prediction is not returned.
		@return (Union[Union[float, np.ndarray],
			Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]):
			If x_err is not None, the output data and the output error are
			returned. If x_err is None, only the output data is returned.
		@error RuntimeError: Error if the function is called before the
			regressor is fitted.
		@error RuntimeError: Error if the input data error is specified and the
			errors of the training data are not specified.
		"""
		if type(x) == np.ndarray and type(x_err) == np.ndarray:
			assert x.shape == x_err.shape, "The shape of the data and the " + \
				f"error must be the same, but they are {x.shape} and " + \
				f"{x_err.shape}, respectively. "

		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		if x_err is None:
			if self.biased:
				return x * self.slope + self.intercept
			else:
				return x * self.slope
		if self._error_specified:
			raise RuntimeError("The error cannot be derived without input " +
				"errors. ")
		if self.biased:
			y = x * self.slope + self.intercept
			y_err = np.sqrt(self.intercept_err**2 + (self.slope*x_err)**2 +
				(x*self.slope_err)**2)
		else:
			y = x * self.slope
			y_err = np.sqrt((self.slope*x_err)**2 + (x*self.slope_err)**2)
		return y, y_err

	def get_slope(self, get_err: bool = False) \
			-> Union[float, Tuple[float, float]]:
		"""
		Get the slope and its error in the fitted parameters of the given data.
		@params get_err (bool): True iff the error of the slope is returned.
		@return (Union[float, Tuple[float, float]]): The slope in the fitted
			parameters of the given data if get_err is False, or the slope and
			its error in sequence if get_err is True.
		@error RuntimeError: Error if the function is called before the
			regressor is fitted.
		@error RuntimeError: Error if the get_err is True and the errors of the
			input data are not specified.
		"""
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		if get_err is False:
			return self.slope
		if not self._error_specified:
			raise RuntimeError("The slope error cannot be derived without" +
				"input errors. ")
		return self.slope, self.slope_err

	def get_intercept(self, get_err: bool = False) \
			-> Union[float, Tuple[float, float]]:
		"""
		Get the intercept and its error in the fitted parameters of the given
		data.
		@params get_err (bool): True iff the error of the intercept is
			returned.
		@return (Union[float, Tuple[float, float]]): The intercept in the
			fitted parameters of the given data if get_err is False, or the
			intercept and its error in sequence if get_err is True.
		@error RuntimeError: Error if the function is called before the
			regressor is fitted.
		@error RuntimeError: Error if the model is unbiased.
		@error RuntimeError: Error if the get_err is True and the errors of the
			input data are not specified.
		"""
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		if not self.biased:
			raise RuntimeError("The model is not biased. ")
		if get_err is False:
			return self.intercept
		if not self._error_specified:
			raise RuntimeError("The slope error cannot be derived without" +
				"input errors. ")
		return self.intercept, self.intercept_err


	def get_chi_squared(self) -> float:
		"""
		Get the chi-squared value of the given data.
		@return (float): The chi-squared value of the given data.
		@error RuntimeError: Error if the function is called before the
			regressor is fitted.
		@error RuntimeError: Error if neither of the indepdent or dependent
			variable error is specified.
		"""
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		if not self._error_specified:
			raise RuntimeError("The chi-squared value could not be " +
				"calculated without specifying error. ")
		chi_squared = np.sum(self.y_res_r**2)
		if self.biased:
			return chi_squared / (self.n - 2)
		else:
			return chi_squared / (self.n - 1)

	def get_correlation_coef(self) -> float:
		"""
		Get the correlation coefficient of the given data.
		@return (float): The correlation coefficient of the given data.
		@error RuntimeError: Error if the function is called before the
			regressor is fitted.
		"""
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		if self._error_specified:
			weights = self.weights
		else:
			weights = None
		x_avg = np.average(self.x, weights = weights)
		y_avg = np.average(self.y, weights = weights)
		cov_xy = np.average((self.x - x_avg) * (self.y - y_avg), weights = weights)
		sd_x = np.sqrt(np.average((self.x - x_avg) ** 2, weights = weights))
		sd_y = np.sqrt(np.average((self.y - y_avg) ** 2, weights = weights))
		return cov_xy/(sd_x * sd_y)

	def get_q_value(self) -> float:
		"""
		Get the Q value, the average of the squared residual, of the given
		data.
		@return (float): The Q value of the given data.
		@error RuntimeError: Error if the function is called before the
			validator is fitted.
		"""
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		return np.mean(self.y_res ** 2)

	def get_y_est(self, interpolate: int =1) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Get the one-dimensional array of the estimated value of the given data
		in order.
		@params interpolate (int): The number of the interval between any of
			the two original data point. It must be positive.
		@return: The estimated value of the given data.
		@error RuntimeError: Error if the function is called before the
			regressor is fitted.
		"""
		assert interpolate > 0, \
			"The number of intervals interpolated must be positive, but " + \
			f"it is {interpolate}. "
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		if interpolate == 1:
			return self.x, self.y_est
		else:
			x_interpolate = \
				np.append(np.linspace(self.x[:-1], self.x[1:], num=interpolate,
					endpoint=False).T.flatten(), self.x[-1])
			if self.biased:
				return x_interpolate, \
					x_interpolate * self.slope + self.intercept
			else:
				return x_interpolate, x_interpolate * self.slope

	def get_y_res_r(self) -> np.ndarray:
		"""
		Get the one-dimensional array of the normalized residuals of the given
		data in order.
		@return (np.ndarray): The normalized residuals of the given data.
		@error RuntimeError: Error if the function is called before the
			regressor is fitted.
		@error RuntimeError: Error if neither of the indepdent or dependent
			variable error is specified.
		"""
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		if not self._error_specified:
			raise RuntimeError("The normalized residual could not be " +
				"calculated without specifying error. ")
		return self.y_res * np.sqrt(self.weights)

	def get_y_res(self) -> np.ndarray:
		"""
		Get the one-dimensional array of the residuals of the given data
		in order.
		@return (np.ndarray): The residuals of the given data.
		@error RuntimeError: Error if the function is called before the
			validator is fitted.
		"""
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		return self.y_res

def interpolate_array(arr: np.ndarray, k: int):
	"""
	Sort and interpolate the array ARR with K intervals between each two of
	the original elements.
	@params arr (numpy.ndarray): The 1-dimensional array to be interpolated.
	@params k (int): The number of intervals between each of the two original
		elements. It must be greater than 1.
	"""
	arr = np.array(arr)
	assert arr.ndim == 1, "The array is not one-dimensional. "
	assert len(arr) > 1, "Not enough elements to interpolate. "
	assert k > 1, "The number of intervals must be greater than 1. "
	arr.sort()
	result = np.linspace(arr[:-1], arr[1:], num=k, endpoint=False).T.flatten()
	return np.append(result, arr[-1])

class WeightedNonlinearRegressor(Fitter):
	"""
	Weighted nonlinear regressor.
	"""
	def __init__(self, fn, fn_derivative=None):
		super().__init__()
		self.fn = np.vectorize(fn)
		if fn_derivative:
			self.fn_derivative = np.vectorize(fn_derivative)
		else:
			self.fn_derivative = None

	def fit(self, x: np.ndarray, x_err: Optional[np.ndarray],
			y: np.ndarray, y_err: Optional[np.ndarray],
			init_params: Optional[List[float]] = None) -> None:
		"""
		Fit the weighted nonlinear regressor.
		@params x (numpy.ndarray): The array of the independent variable data
			with shape (n,).
		@params x_err (Optional[numpy.ndarray]): The array of the error of the
			independent variable in sequence with x with shape (n,).
		@params y (numpy.ndarray): The array of the dependent variable data in
			sequence with x with shape (n,).
		@params y_err (Optional[numpy.ndarray]): The array of the error of the
			dependent variable in sequence with x with shape (n,).
		@params init_params (Optional[List[float]]): The list of an initial
			guess of the parameters. It must have the same signature as self.fn
			except the first argument, which is omitted.
		"""
		assert x.ndim == 1, \
			"The independent variable data array should have a dimension " + \
			f"1, but it is {x.ndim}. "
		assert y.shape == x.shape, \
			"The indepdent variable data array and the dependent variable " + \
			"data array should have the same shape, but they have shape " + \
			f"{x.shape} and {y.shape}, respectively. "
		self.n = len(x)
		self.x = x
		self.y = y

		if x_err is None and y_err is None:
			self._error_specified = False
			self._fit_simple(x, y, init_params)
		else:
			assert x_err is None or x_err.shape == x.shape, \
				"The independent variable data array and the independent " + \
				"variable error array should have the same shape if " + \
				f"specified, but they have shape {x.shape} and " + \
				f"{x_err.shape}, respectively. "
			assert y_err is None or y_err.shape == y.shape, \
				"The dependent variable data array and the dependent " + \
				"variable error array should have the same shape if " + \
				f"specified, but they have shape {y.shape} and " + \
				f"{y_err.shape}, respectively. "
			self._error_specified = True
			self._fit_full(x, x_err, y, y_err, init_params)
		self._fit = True

	def _fit_simple(self, x: np.ndarray, y: np.ndarray,
			init_params: Optional[List[float]] = None) -> None:
		"""
		The unweighted nonlinear regression for N samples.
		@params x (numpy.ndarray): The array of the independent variable data
			with shape (n,).
		@params y (numpy.ndarray): The array of the dependent variable data in
			sequence with x with shape (n,). The shape consistency is enforced
			by the caller function.
		@params init_params (Optional[List[float]]): The list of an initial
			guess of the parameters. The signature is enforced by the caller
			function.
		"""
		params_opt, params_cov = scipy.optimize.curve_fit(self.fn, x, y,
			p0=init_params, check_finite=True)
		self.params_std = np.sqrt(np.diagonal(params_cov))
		self.params_opt = params_opt
		self.y_est = self.fn(x, *self.params_opt)

	def _fit_full(self, x: np.ndarray, x_err: Optional[np.ndarray],
			y: np.ndarray, y_err: Optional[np.ndarray],
			init_params: Optional[List[float]] = None):
		"""
		The weighted nonlinear regression for N samples.
		@params x (numpy.ndarray): The array of the independent variable data
			with shape (n,).
		@params x_err (Optional[numpy.ndarray]): The array of the error of the
			independent variable in sequence with x with shape (n,). The shape
			consistency is enforced by the caller function.
		@params y (numpy.ndarray): The array of the dependent variable data in
			sequence with x with shape (n,). The shape consistency is enforced
			by the caller function.
		@params y_err (Optional[numpy.ndarray]): The array of the error of the
			dependent variable in sequence with x with shape (n,). Either x_err
			or y_err must not be None. This is enforced by the caller function.
			The shape consistency is enforced by the caller function.
		@params init_params (Optional[List[float]]): The list of an initial
			guess of the parameters. The signature is enforced by the caller
			function.
		@error RuntimeError: Error if the fn derivative is not specified but
			x_err is specified.
		@error ValueError: Error if infinite weights are encountered.
		"""
		sigma2 = y_err ** 2
		if x_err is not None:
			if self.fn_derivative is None:
				raise RuntimeError("Model function derivative is not " +
					"specified. ")
			self._fit_simple(x, y, init_params)
			sigma2 += (self.fn_derivative(x, *self.params_opt) * x_err) ** 2
		self.sigma = np.sqrt(sigma2)

		self.params_opt, params_cov = scipy.optimize.curve_fit(self.fn, x, y,
			p0=init_params, sigma=self.sigma, check_finite=True)
		self.params_std = np.sqrt(np.diagonal(params_cov))
		self.y_est = self.fn(x, *self.params_opt)
		self.y_res_r = (self.y_est - y) / self.sigma

	def get_params(self) -> Tuple[List[float], List[float]]:
		"""
		Get the fitted paramters are their error of the given data.
		@return (Tuple[List[float]], List[float]]): the fitted paramters are
			their error of the given data.
		@error RuntimeError: Error if the function is called before the
			regressor is fitted.
		"""
		if not self._fit:
			raise RuntimeError("The regressor has not been fitted. ")
		return self.params_opt, self.params_std

	def get_y_est(self):
		return self.y_est

	def get_y_res_r(self):
		return self.y_res_r

def weighted_nonlinear_regression(fn, x_data, x_err, y_data, y_err,
        init_params, fn_derivative=None):
    """
    Nonlinear least-square regression with the hypothesis denoted by the
    function FN with N samples, where N>0, and P parameters, where P>0, and
    return the parameter with their error in order. This uses
    scipy.optimize.curve_fit.
    @params fn (function): The function of the hypothesis.
        Arguments of the function: (x, *params)
            x (float): The independent variable
            params (numpy.ndarray): Parameters of the hypothesis with shape
                (p,).
        Retun value of the function (float): The dependent variable value.
    @params x_data (numpy.ndarray): Independent variable X with shape (n,)
    @params x_err (numpy.ndarray, default None): Error of independent variable
        X in order with shape (n,). Iff it is None, then the error is 0 for all
        sample points.
    @params y_data (numpy.ndarray): Dependent variable Y with shape (n,)
    @params y_err (numpy.ndarray): Error of dependent variable Y in order
        with shape (n,). None of the element can be zero.
    @params init_params (numpy.ndarray): Initial parameters in order with FN
        with shape (p,).
    @params fn_derivative (function): The derivative function of the function
        of the hypothesis.
        Arguments of the function: (x, *params)
            x (float): The independent variable
            params (numpy.ndarray): Parameters of the hypothesis with shape
                (p,).
        Return value of the function (float): The derivative of the function
            with dependent variable X.
    @return params_opt (numpy.ndarray): The fitted parameters in order with FN
        with shape (p,).
    @return params_std (numpy.ndarray): The standard deviations of the fitted
        parameters in order with FN with shape (p,).
    @error ValueError: Iff either y_data or x_data contain NaNs, or if
        incompatible options are used.
    @error RuntimeError: If the least-squares minimization fails.
    @error OptimizeWarning: Iff covariance of the parameters cannot be
        estimated.
    """

    assert x_err is None or fn_derivative is not None, \
        "The derivative of the hypothesis function is not specified."

    x_data = np.array(x_data)
    x_err = np.array(x_err)
    y_data = np.array(y_data)
    y_err = np.array(y_err)
    init_params = np.array(init_params)
    n = len(x_data)
    x_err_is_none = False
    if x_err is None:
        x_err_is_none = True
        x_err = np.zeros(n)
    assert np.all(y_err), "None of the error of Y can be zero."
    assert x_data.shape == x_err.shape == y_data.shape == y_err.shape, \
        "All the inputs must have the same shape"
    assert x_data.shape == (n,), \
        "The input data must be 1-dimensional."
    assert len(init_params) + 1 == len(inspect.signature(fn).parameters), \
        "The length of the list of INIT_PARAMS does not equal to the " + \
        "number of parameters."

    fn = np.vectorize(fn)
    fn_derivative = np.vectorize(fn_derivative)

    if not x_err_is_none:
        params_opt, params_cov = scipy.optimize.curve_fit(fn, x_data, y_data,
            p0=init_params, check_finite=True)
        sigma = np.sqrt(y_err*2 + (x_err * \
            fn_derivative(x_data, *params_opt))**2)
    else:
        sigma = y_err

    params_opt, params_cov = scipy.optimize.curve_fit(fn, x_data, y_data,
        p0=init_params, sigma=sigma, check_finite=True)
    params_std = np.sqrt(np.diagonal(params_cov))
    y_est = fn(x_data, *params_opt)
    y_res_r = (y_est - y_data) / sigma
    return params_opt, params_std, y_est, y_res_r

class FunctionValidator(Fitter):
	"""
	The validator of a function with the given data.
	@field self.fn (Callable[[float], float]): The function of the validator.
	@field self.dfn (Optional[Callable[[float], float]]): The derivative
		function of the function of the validator.
	@field self._error_specified (bool): True iff the error of the data is
		specified. Either x error or y error are sufficient.
	@field self._fit (bool): True iff the validator is fitted.
	@field self.x (np.ndarray): The one dimensional array of the indepdent
		variable data.
	@field self.y_est (np.ndarray): The one dimensional array of the estimated
	 	value of the given data in order.
	@field self.y_res (np.ndarray): The one dimensional array of the residual
		of the given data in order.
	@field weights (np.ndarray): The one dimensional array of the weights of
		the given data in order.
	"""

	def __init__(
			self, fn: Callable[[float], float],
			dfn: Optional[Callable[[float], float]]=None
		):
		"""
		Initialze a function validator.
		@params fn (Callable[[float], float]): The function of the validator.
		@params dfn (Optional[Callable[[float], float]]): The derivative
			function of the function of the validator.
		"""
		super().__init__()
		self.fn = np.vectorize(fn)
		self.dfn = np.vectorize(dfn)

	def fit(self, x: np.ndarray, x_err: np.ndarray, y: np.ndarray,
			y_err: np.ndarray) -> None:
		"""
		Fit the function validator. Suppose there is
		@params x (np.ndarray): The one-dimensional array of the independent
			variable data.
		@params x_err (Optional[np.ndarray]): The one-dimensional array of the
			error of the independent variable data. If it is NONE, then the
			independent variable error is not specified. If it is a
			numpy.ndarray, it is in order with X and must have the same shape
			as X.
		@params y (np.ndarray): The one-dimensional array of the dependent
			variable data. It is in order with X and must have the same shape
			as X.
		@params y_err (Optional[np.ndarray]): The one-dimensional array of the
			error of the dependent variable data. If it is NONE, then the
			dependent variable error is not specified. If it is a
			numpy.ndarray, it is in order with X and must have the same shape
			as X.
		@error RuntimeError: Error if the derivative function is not specified
			if x_err is not None.
		"""
		assert x.ndim == 1, \
			f"The array should have a dimension of 1, but it is {x.ndim}. "
		assert x.shape == x_err.shape and x.shape == y.shape and \
			x.shape == y_err.shape, \
			"The array should have the same lengths, but they are " + \
			f"{len(x)}, {len(x_err)}, {len(y)}, and {len(y_err)}. "

		self._error_specified = x_err is not None or y_err is not None
		self._fit = True
		self.x = x
		self.y_est = self.fn(x)
		self.y_res = y - self.y_est

		if x_err is None:
			if y_err is None :
				self.weights = np.ones(len(x))
			else:
				self.weights = 1 / y_err ** 2
		else:
			if self.dfn is None:
				raise RuntimeError("The derivative function is not " +
					"specified. ")
			if y_err is None:
				self.weights = 1 / (self.dfn(x) * x_err) ** 2
			else:
				self.weights = 1 / (y_err ** 2 + (self.dfn(x) * x_err) ** 2)

	def get_correlation_coef(self) -> float:
		"""
		Get the correlation coefficient of the given data.
		@return (float): The correlation coefficient of the given data.
		@error RuntimeError: Error if the function is called before the
			validator is fitted.
		"""
		if not self._fit:
			raise RuntimeError("The data has not been fitted. ")
		y = self.y_est + self.y_res
		y_est_avg = np.average(self.y_est, weights = self.weights)
		y_avg = np.average(y, weights = self.weights)
		covariance = np.average((self.y_est - y_est_avg) * (y - y_avg),
			weights = self.weights)
		sd_y_est = np.sqrt(np.average((self.y_est - y_est_avg) ** 2,
			weights = self.weights))
		sd_y = np.sqrt(np.average((y - y_avg) ** 2,
			weights = self.weight))
		self.corr = covariance / (sd_y_est * sd_y)
		return self.corr

	def get_chi_squared(self) -> float:
		"""
		Get the chi-squared value of the given data.
		@return (float): The chi-squared value of the given data.
		@error RuntimeError: Error if the function is called before the
			validator is fitted.
		@error RuntimeError: Error if neither of the indepdent or dependent
			variable error is specified.
		"""
		if not self._fit:
			raise RuntimeError("The data has not been fitted. ")
		if not self._error_specified:
			raise RuntimeError("The chi-squared value could not be " +
				"calculated without specifying error. ")
		return np.mean(self.y_res**2 * self.weights)

	def get_q_value(self) -> float:
		"""
		Get the Q value, the average of the squared residual, of the given
		data.
		@return (float): The Q value of the given data.
		@error RuntimeError: Error if the function is called before the
			validator is fitted.
		"""
		if not self._fit:
			raise RuntimeError("The data has not been fitted. ")
		return np.mean(self.y_res ** 2)

	def get_y_est(self, interpolate: int =1) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Get the one-dimensional array of the estimated value of the given data
		in order.
		@params interpolate (int): The number of the interval between any of
			the two original data point. It must be positive.
		@return: The estimated value of the given data.
		@error RuntimeError: Error if the function is called before the
			validator is fitted.
		"""
		assert interpolate > 0, \
			"The number of intervals interpolated must be positive, but " + \
			f"it is {interpolate}. "
		if not self._fit:
			raise RuntimeError("The data has not been fitted. ")
		if interpolate == 1:
			return self.x, self.y_est
		else:
			x_interpolate = \
				np.append(np.linspace(self.x[:-1], self.x[1:], num=interpolate,
					endpoint=False).T.flatten(), self.x[-1])
			return x_interpolate, self.fn(x_interpolate)

	def get_y_res_r(self) -> np.ndarray:
		"""
		Get the one-dimensional array of the normalized residuals of the given
		data in order.
		@return (np.ndarray): The normalized residuals of the given data.
		@error RuntimeError: Error if the function is called before the
			validator is fitted.
		@error RuntimeError: Error if neither of the indepdent or dependent
			variable error is specified.
		"""
		if not self._fit:
			raise RuntimeError("The data has not been fitted. ")
		if not self._error_specified:
			raise RuntimeError("The normalized residual could not be " +
				"calculated without specifying error. ")
		return self.y_res * np.sqrt(self.weights)

	def get_y_res(self) -> np.ndarray:
		"""
		Get the one-dimensional array of the residuals of the given data
		in order.
		@return (np.ndarray): The residuals of the given data.
		@error RuntimeError: Error if the function is called before the
			validator is fitted.
		"""
		if not self._fit:
			raise RuntimeError("The data has not been fitted. ")
		return self.y_res
