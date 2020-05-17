"""
Implementation of LiNGAM-GC proposed in [1]

Shoubo Hu (shoubo.sub [at] gmail.com)
2020-05-17

[1] Chen, Zhitang, and Laiwan Chan. "Causality in linear nongaussian 
    acyclic models in the presence of latent gaussian confounders."
    Neural Computation 25.6 (2013): 1605-1641.
"""
from __future__ import division
import numpy as np
from scipy.stats import kurtosis
from sklearn.utils import check_array
from sklearn.linear_model import LassoLarsIC, LinearRegression

class LiNGAM_GC(object):
	def __init__(self):
		self._causal_order = None # list storing the causal order

	def fit(self, X):
		"""
		fit LiNGAM-GC on X and estimate the adjacency matrix

		Input:
			X 		 the matrix of observed data
					 (n_instance, n_var) numpy array
		"""
		self.data = check_array(X)
		n_var = self.data.shape[1]

		U = np.arange(n_var)
		K = []
		X_ = np.copy(X)
		for _ in range(0, n_var):
			cu_i = self._search_exogenous_x(X_, U)
			for i in U:
				if i != cu_i:
					X_[:, i] = self._residual( X_[:, i], X_[:, cu_i] )
			K.append(cu_i)
			U = U[U != cu_i]

		self._causal_order = K
		self._estimate_adjacency_matrix(X)

	def get_results(self):
		if self._causal_order == None:
			print('Error: model not yet fitted on data!')
			return self
		print('Estimated causal order')
		print(self._causal_order)
		print('Estimated graph structure')
		print(self.adjacency_matrix_)
		return self._causal_order, self.adjacency_matrix_

	def _estimate_adjacency_matrix(self, X):
		"""
		estimate adjacency matrix according to the causal order.

		Input:
			X 		 the matrix of observed data
					 (n_instance, n_var) numpy array

		Output:
			self 	 the object itself
		"""
		B = np.zeros([X.shape[1], X.shape[1]], dtype='float64')
		for i in range(1, len(self._causal_order)):
			coef = self._predict_adaptive_lasso(
				X, self._causal_order[:i], self._causal_order[i])
			B[self._causal_order[i], self._causal_order[:i]] = coef
		self.adjacency_matrix_ = B
		return self

	def _predict_adaptive_lasso(self, X, predictors, target, gamma=1.0):
		"""
		predict with Adaptive Lasso.

		Input:
			X 				 training instances 
							 (n_instances, n_var) numpy array
			predictors 		 indices of predictor variables
							 (n_predictors, ) list()
			target 			 index of target variable
							 int

		Output:
			coef 			 Coefficients of predictor variable 
							 (n_predictors,) numpy array
		"""
		lr = LinearRegression()
		lr.fit(X[:, predictors], X[:, target])
		weight = np.power(np.abs(lr.coef_), gamma)
		reg = LassoLarsIC(criterion='bic')
		reg.fit(X[:, predictors] * weight, X[:, target])
		return reg.coef_ * weight

	def _search_exogenous_x(self, X, U):
		"""
		find the exogenous variable in the remaining ones

		Input:
			X 		 the matrix of observed data
					 (n_instance, n_var) numpy array
			U 		 list of remaining variables to be ordered
					 list()

		Output:
			i 		 the index of the exogenous variable
					 int
		"""
		if len(U) == 1:
			return U[0]

		M_list = []
		for i in U:
			M = 0.0
			xi_hat = self._standardize( X[:,i] )
			for j in U:
				if i != j:
					xj_hat = self._standardize( X[:,j] )
					R_xi_xj = self._causal_measure(xi_hat, xj_hat)
					M += np.min( [0, R_xi_xj] )**2
			M_list.append( -1.0*M )
		return U[np.argmax(M_list)]

	def _residual(self, xi, xj):
		return xi - ( self._cross_cumulant_4th(xj, xi) / self._cross_cumulant_4th(xj, xj) ) * xj

	def _cross_cumulant_4th(self, x, y):
		return np.mean(x**3 * y) - 3*np.mean(x*y)*np.mean(x**2)

	def _causal_measure(self, x, y):
		"""
		compute the cumulant-based measure in LiNGAM-GC

		Input:
			x, y 		the vector of standardized data of x and y
						(n_instance,) numpy array

		Output:
			R 			the cumulant-based measure in LiNGAM-GC
						float
		"""
		C_xy = self._cross_cumulant_4th(x, y)
		C_yx = self._cross_cumulant_4th(y, x)
		R = C_xy**2 - C_yx**2
		return R

	def _standardize(self, x):
		"""
		standardize the data to have unit kurtosis

		Input:
			x 			the vector of data
						(n_instance, ) numpy array

		Output:
			x_hat 		the standardized data
						(n_instance, ) numpy array
		"""
		kurts = kurtosis(x)          # calculate Fisher kurtosis
		k_x = np.abs(kurts)**(1./4)  # the quantity for standardization (k_x in [1])
		x_hat = x / k_x              # the standardized data
		return x_hat