from math import exp,pow
from math import fabs

class PairsObjective:
	"""
	Defines a general (abstract) class for an objective function
	"""
	
	@staticmethod
	def getGradient_Doc(obj_options, cur_scores,this_example, user_rel=1.0):
		"""
		Gets the gradient for the consecutive pairs objective
		"""
		model_ties = obj_options.get("model_ties",False)
		score_hinge =  obj_options.get("score_hinge", False)
		
		this_gradient = {}
		this_gt = None	
		if obj_options["all_pairs"]:
			this_gt = this_example.all_greater_than
		else:
			this_gt = this_example.direct_greater_than
		
		for doc in this_example.all_items:
			this_gradient[doc] = 0.0
			
		for doc1 in this_gradient:
			cur_score1 = cur_scores[doc1]
			if score_hinge :
				cur_score1 -= this_example.scored_items[doc1]
			
			for doc2 in this_gt[doc1]:	#Observed doc1 > doc2
				delta = cur_score1 - cur_scores[doc2]
				if score_hinge :
					delta += this_example.scored_items[doc2]
					
				grad_ch = 0.0
				exp_val = exp(delta*user_rel)
				
				grad_ch= 0.0
				
				if not model_ties:
					grad_ch = float(user_rel)/float(1 + exp_val )
				else:
					sq_exp = exp_val*exp_val
					grad_ch = float(user_rel*(2+exp_val) )/float(1 + exp_val+sq_exp )
				
				this_gradient[doc1] -= grad_ch
				this_gradient[doc2] += grad_ch
				
			if model_ties:
				cur_score1 = cur_scores[doc1]
				for doc2 in this_example.tied_to[doc1]:
					delta = cur_score1 - cur_scores[doc2]
					exp_term1 = exp(user_rel*delta)
					exp_term2 = float(1)/float(exp_term1)
					
					this_gradient[doc1] += float(user_rel * (exp_term1 - exp_term2) ) / float(1 + exp_term1 + exp_term2)
					

		return this_gradient

	@staticmethod
	def getGradient_User(obj_options, cur_scores,this_example, user_rel=1.0):
		"""
		Gets the gradient for the consecutive pairs objective
		"""
		model_ties = obj_options.get("model_ties",False)
		score_hinge =  obj_options.get("score_hinge", False)
		
		this_gradient = 0.0
		this_gt = None	
		if obj_options["all_pairs"]:
			this_gt = this_example.all_greater_than
		else:
			this_gt = this_example.direct_greater_than
				
		for doc1 in this_gt:
			cur_score1 = cur_scores[doc1]
			if score_hinge :
				cur_score1 -= this_example.scored_items[doc1]
			
			for doc2 in this_gt[doc1]:	#Observed doc1 > doc2
				delta = cur_score1 - cur_scores[doc2]
				if score_hinge:
					delta += this_example.scored_items[doc2]
				
				exp_val = exp(delta*user_rel)
				if not model_ties:
					this_gradient -= float(delta)/float(1 + exp_val )
				else:
					sq_exp = exp_val*exp_val
					this_gradient -= float(delta * (2+exp_val) )/float(1 + exp_val+sq_exp )
					
			if model_ties:
				cur_score1 = cur_scores[doc1]
				for doc2 in this_example.tied_to[doc1]:
					delta = cur_score1 - cur_scores[doc2]						
					exp_term1 = exp(user_rel*delta)
					exp_term2 = float(1)/float(exp_term1)
					
					this_gradient += float(delta * (exp_term1 - exp_term2) ) / float(1 + exp_term1 + exp_term2)
				
		return this_gradient
