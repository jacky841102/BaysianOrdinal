
class L2Regularizer:
	"""
	A L2Regularizer class is one with the L2-norm regularizer on the variable.
	
	This is used for variables with a Normal-prior (input mean, variance)
	"""
	
	@staticmethod
	def getGradient(varValue,regParams = (1,0), numEx = 1 ):
		"""
		Returns gradient
		"""
		( regConstScale,regConstMean ) = regParams
		return (regConstScale/float(numEx)) * (varValue - regConstMean)
	
	@staticmethod
	def getHessian(varValue,regParams = (1,0), numEx = 1):
		"""
		Returns second order derivative
		"""
		( regConstScale,regConstMean ) = regParams
		return (regConstScale/float(numEx))