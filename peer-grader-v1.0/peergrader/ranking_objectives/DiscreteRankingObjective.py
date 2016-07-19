# EXACT objective

from math import exp, pow
from math import fabs
import itertools as it


class DiscreteRankingObjective:
	"""
	Discrete ranking objective function
	"""
	
	@staticmethod
	def getGradient_Doc(obj_options, cur_scores,this_example,user_rel=1.0):
		"""
		Gets the gradient for the Exact discrete ranking objective
		"""
		this_gradient = {}
		neg_user = -user_rel
		
		this_ele_scores = [cur_scores[d] for d in this_example.all_items]

		num_ele = len(this_ele_scores)
		num_ele_range = range(num_ele)
		
		first_denom = 0.0
		first_numer = [0.0]*num_ele
		
		second_denom = 0.0
		second_numer = [0.0]*num_ele
		
		ref_score = sum( (i+1)*this_ele_scores[i] for i in num_ele_range )	#reference score
		this_val_rankings = this_example.valid_rankings
		
		for ranking in it.permutations( range(1,num_ele+1) ):		
			inner_prod = sum( ranking[i]*this_ele_scores[i] for i in num_ele_range ) - ref_score
			exp_score = exp(neg_user * inner_prod)
			second_denom += exp_score		
			for j in num_ele_range:	second_numer[j] += (exp_score*user_rel*ranking[j])
				
			if ranking in this_val_rankings:
				first_denom += exp_score
				for j in num_ele_range:	first_numer[j] += (exp_score*user_rel*ranking[j])
		
		
		for i in num_ele_range:
			doc = this_example.all_items[i]
			first_term = float(first_numer[i])/float(first_denom)
			second_term = float(second_numer[i])/float(second_denom)
			this_gradient[doc] = first_term - second_term
			
		return this_gradient

	@staticmethod
	def getGradient_User(obj_options, cur_scores,this_example,user_rel=1.0):
		"""
		Gets the gradient for the Exact discrete ranking objective
		"""
		this_gradient = 0.0
		neg_user = -user_rel
		
		this_ele_scores = [cur_scores[d] for d in this_example.all_items]
		num_ele = len(this_ele_scores)
		num_ele_range = range(num_ele)
		
		first_denom = 0.0
		first_numer = 0.0
		
		second_denom = 0.0
		second_numer = 0.0
		
		ref_score = sum( (i+1)*this_ele_scores[i] for i in num_ele_range )	#reference score
		this_val_rankings = this_example.valid_rankings
		
		for ranking in it.permutations( range(1,num_ele+1) ):		
			inner_prod = sum( ranking[i]*this_ele_scores[i] for i in num_ele_range ) 
			exp_score = exp( neg_user * (inner_prod - ref_score) )
			second_denom += exp_score		
			second_numer += (exp_score*inner_prod)
				
			if ranking in this_val_rankings:
				first_denom += exp_score
				first_numer += (exp_score*inner_prod)
			
		first_term = float(first_numer)/float(first_denom)
		second_term = float(second_numer)/float(second_denom)
		this_gradient = first_term - second_term
		
		return this_gradient