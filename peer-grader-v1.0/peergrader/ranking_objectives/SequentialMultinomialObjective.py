# RELATED To Repeated-Insertion Model from Lu and Boutellier? Also BTL model?

from math import exp
from math import fabs


class SequentialMultinomialObjective:
	"""
	Sequential multinomial objective function
	"""
	
	@staticmethod
	def getGradient_Doc(obj_options, cur_scores,this_example,user_rel=1.0):
		"""
		Gets the gradient for the sequential multinomial pairs objective
		"""
		score_hinge = obj_options.get("score_hinge",False)
		model_ties = obj_options.get("model_ties",False)
		
		this_gradient = {}
		this_gt = this_example.all_greater_than
		this_tied = this_example.tied_to
		this_expscore = {}
		
		for doc in this_example.all_items:
			this_gradient[doc] = 0.0
			cur_score = cur_scores[doc]
			if score_hinge:
				cur_score -= this_example.scored_items[doc]
			this_expscore[doc] = exp(cur_score*user_rel)
			
		for doc1 in this_gradient:
			if len(this_gt[doc1]) == 0 and ( (not model_ties) or (model_ties and len(this_tied[doc1]) == 0) ):	#No term 
				continue
			
			exp_score1 = this_expscore[doc1]
			denom = exp_score1
			for doc2 in this_gt[doc1]:	#Observed doc1 > doc2
				denom += this_expscore[doc2]
				
			if model_ties:
				for doc2 in this_tied[doc1]:
					denom += this_expscore[doc2]
			
			this_gradient[doc1] +=  ( float(user_rel*exp_score1)/float(denom) ) - user_rel	
			for doc2 in this_gt[doc1]:	#Observed doc1 > doc2				
				this_gradient[doc2] += ( float(user_rel*this_expscore[doc2])/float(denom) )
				
			if model_ties:
				for doc2 in this_tied[doc1]:
					this_gradient[doc2] += ( float(user_rel*this_expscore[doc2])/float(denom) )

		return this_gradient

	@staticmethod
	def getGradient_User(obj_options, cur_scores,this_example,user_rel=1.0):
		"""
		Gets the gradient for the sequential multinomial pairs objective
		"""
		score_hinge = obj_options.get("score_hinge",False)
		model_ties = obj_options.get("model_ties",False)
		
		this_gradient = 0.0
		this_gt = this_example.all_greater_than
		this_tied = this_example.tied_to
		this_expscore = {}
		this_expscore2 = {}
		
		for doc in this_example.all_items:
			cur_score = cur_scores[doc]
			if score_hinge:
				cur_score -= this_example.scored_items[doc]
			this_expscore[doc] = exp(cur_score*user_rel)
			this_expscore2[doc] = cur_score*exp(cur_score*user_rel)
			
		for doc1 in this_expscore:
			if len(this_gt[doc1]) == 0 and ( (not model_ties) or (model_ties and len(this_tied[doc1]) == 0) ):	#No term if nothing is less than doc1
				continue
			
			denom = this_expscore[doc1]
			numer = this_expscore2[doc1]
			for doc2 in this_gt[doc1]:	#Observed doc1 > doc2
				denom += this_expscore[doc2]
				numer += this_expscore2[doc2]
				
			if model_ties:
				for doc2 in this_tied[doc1]:
					denom += this_expscore[doc2]
					numer += this_expscore2[doc2]
			
			
			ratio_term = 0.0
			if denom > 0:
				ratio_term = ( float(numer)/float(denom) )
			this_gradient +=  ratio_term - cur_scores[doc1]

		return this_gradient