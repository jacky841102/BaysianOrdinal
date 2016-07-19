
class GammaRegularizer:
	"""
	A GammaRegularizer class is used for variables with a Gamma-prior (input parameters)
	"""
	
	@staticmethod
	def getGradient(varValue,regParams = (11,1) , numEx = 1 ):
		"""
		Returns gradient
		"""
		( regConstAlpha,regConstBeta ) = regParams
		return  ( - ( (regConstAlpha - 1) / varValue ) + regConstBeta )/float(numEx)
		
	@staticmethod
	def getHessian(varValue,regParams = (11,1),  numEx = 1):
		"""
		Returns second order derivative
		"""
		( regConstAlpha,regConstBeta ) = regParams
		return (regConstAlpha - 1) / (varValue*varValue*float(numEx))