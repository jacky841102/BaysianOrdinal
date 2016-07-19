from math import sqrt

class NormalCardinalScore:
	"""
	Defines a general (abstract) class for an objective function
	"""
	
	@staticmethod
	def getGradient_Doc(obj_options, cur_scores,this_example, user_rel=1.0, user_bias = 0.0, user_scale = 1.0):
		"""
		Gets the gradient for the consecutive pairs objective
		"""
		this_gradient = {}	
		bias_grad = 0.0

		for doc in this_example.all_items:
			this_score = this_example.scored_items[doc]
			cur_score = cur_scores[doc]
			grad_ch = float(user_rel * (this_score - (user_bias+cur_score) ) )
			this_gradient[doc]= - grad_ch
			bias_grad -= grad_ch
			

		return (this_gradient, bias_grad)

	@staticmethod
	def getGradient_User(obj_options, cur_scores,this_example, user_rel=1.0, user_bias = 0.0, user_scale = 1.0):
		this_gradient = 0.0
		for doc in this_example.all_items:
			this_score = this_example.scored_items[doc] 
			cur_score = cur_scores[doc]
			diff = (this_score - (user_bias+cur_score) )
			this_gradient += float( diff*diff )/float(2)
			this_gradient -= float(1)/float(2*user_rel)
		return this_gradient