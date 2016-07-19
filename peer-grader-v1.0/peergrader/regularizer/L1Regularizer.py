from math import copysign
from Regularizer import Regularizer

class L1Regularizer(Regularizer):
	"""
	A L1Regularizer class is used to enforce L1 norm penalty on the variable.
	
	It is to be used on variables with a Laplace-prior (input parameters)
	"""
	
	@staticmethod
	def getGradient(varValue,regParams = (1,0), numEx ):
		"""
		Returns gradient
		"""
		( regConstScale,regConstMean ) = regParams
		return copysign( 1, varValue - regConstMean ) * (regConstScale/float(numEx))
	
	@staticmethod
	def getHessian(varValue,regParams = (1,0))
		"""
		Returns second order derivative
		"""
		return 0