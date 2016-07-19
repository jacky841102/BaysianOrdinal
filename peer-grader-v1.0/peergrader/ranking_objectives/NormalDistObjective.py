from math import exp, sqrt

havescipy = False
try:
	from scipy.stats import norm
except:
	logger.warn("Missing scipy. Cannot run some methods.")
	raise Exception("Missing scipy. Cannot run some methods.")

class NormalDistObjective:
	"""
	Defines a general (abstract) class for an objective function
	"""
	
	@staticmethod
	def getGradient_Doc(obj_options, cur_scores,this_example, user_rel=1.0):
		"""
		Gets the gradient for the consecutive pairs objective
		"""
		#u_mult = float(user_rel*user_rel)/float(2)
		u_mult = sqrt(user_rel)
		threshold = obj_options.get("tie_threshold",0.0)
		
		this_gradient = {}
		this_gt = None	
		if obj_options["all_pairs"]:
			this_gt = this_example.all_greater_than
		else:
			this_gt = this_example.direct_greater_than
		this_tied = this_example.tied_to
		
		for doc in this_example.all_items:
			this_gradient[doc] = 0.0
			
		for doc1 in this_gradient:
			cur_score1 = cur_scores[doc1]
			if obj_options.get("score_hinge",False):
				cur_score1 -= this_example.scored_items[doc1]
			
			for doc2 in this_gt[doc1]:	#Observed doc1 > doc2
				delta = cur_score1 - cur_scores[doc2]
				if obj_options.get("score_hinge",False):
					delta += this_example.scored_items[doc2]
				
				delta -= threshold
				delta *= u_mult
				grad_ch = ( norm.pdf(delta)/norm.cdf(delta) ) * u_mult
					
				this_gradient[doc1] -= grad_ch
				this_gradient[doc2] += grad_ch
				
			if threshold > 0:
				cur_score1 = cur_scores[doc1]
				for doc2 in this_tied[doc1]:	#Observed doc1 > doc2
					delta = cur_score1 - cur_scores[doc2]					
					delta1 = (delta - threshold)*u_mult
					delta2 = (delta + threshold)*u_mult
					grad_ch = u_mult*float( norm.pdf(delta2) - norm.pdf(delta1) ) / float( norm.cdf(delta2) - norm.cdf(delta1) )  
					this_gradient[doc1] -= grad_ch

		return this_gradient



	@staticmethod
	def getGradient_User(obj_options, cur_scores,this_example, user_rel=1.0):
		"""
		Gets the gradient for the consecutive pairs objective
		"""
		#u_mult = float(user_rel*user_rel)/float(2)
		u_mult = sqrt(user_rel)
		threshold = obj_options.get("tie_threshold",0.0)
		
		this_gradient = 0.0
		this_gt = None	
		if obj_options["all_pairs"]:
			this_gt = this_example.all_greater_than
		else:
			this_gt = this_example.direct_greater_than
		this_tied = this_example.tied_to
			
		for doc1 in this_gt:
			cur_score1 = cur_scores[doc1]
			if obj_options.get("score_hinge",False):
				cur_score1 -= this_example.scored_items[doc1]
			
			for doc2 in this_gt[doc1]:	#Observed doc1 > doc2
				delta = cur_score1 - cur_scores[doc2]
				if obj_options.get("score_hinge",False):
					delta += this_example.scored_items[doc2]
				
				delta -= threshold			
				delta *= u_mult

				this_gradient -= ( norm.pdf(delta)/norm.cdf(delta) ) * (delta/float(2*user_rel))
				#this_gradient -= ( norm.pdf(delta2)/norm.cdf(delta2) ) * user_rel * delta
				
			if threshold > 0:
				cur_score1 = cur_scores[doc1]
				for doc2 in this_tied[doc1]:	#Observed doc1 > doc2
					delta = cur_score1 - cur_scores[doc2]					
					delta1 = (delta - threshold)*u_mult
					delta2 = (delta + threshold)*u_mult
					grad_ch = (float(1)/float(2*user_rel)) * float( (delta2*norm.pdf(delta2)) - (delta1*norm.pdf(delta1)) ) / float( norm.cdf(delta2) - norm.cdf(delta1) )  
					this_gradient -= grad_ch
					
		return this_gradient