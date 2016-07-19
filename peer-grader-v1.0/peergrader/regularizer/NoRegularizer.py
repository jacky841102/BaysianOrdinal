class NoRegularizer(Regularizer):
	"""
	A NoRegularizer class is one where we do not want to regularize the variable.
	
	Consequently the gradient and hessian functions return zeros.
	"""
	
	@staticmethod
	def getGradient(varValue):
		"""
		Returns 0 to indicate no regularizer
		"""
		return 0
	
	@staticmethod
	def getHessian(varValue):
		"""
		Returns 0 to indicate no regularizer
		"""
		return 0