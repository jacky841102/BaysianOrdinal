

class Regularizer(object):
	"""
	Defines a general (abstract) class for a regularizer
	"""
	
	@staticmethod
	def getGradient():
		"""
		Subclasses of Regularizer need to implement this function which returns the gradient of this variable
		"""
		raise NotImplementedError
	
	@staticmethod
	def getHessian():
		"""
		Subclasses of Regularizer need to implement this function which returns the second order derivative of this variable
		"""
		raise NotImplementedError