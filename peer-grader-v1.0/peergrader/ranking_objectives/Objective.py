

class Objective(object):
	"""
	Defines a general (abstract) class for an objective function
	"""
	
	@staticmethod
	def getGradient():
		"""
		Subclasses of Objective need to implement this function which returns the gradient of a specific variable given the example and current variable values
		"""
		raise NotImplementedError
	
	@staticmethod
	def getHessian():
		"""
		Subclasses of Objective need to implement this function which returns the seconf order derivative of a pair of specific variables given the example and current variable values
		"""
		raise NotImplementedError