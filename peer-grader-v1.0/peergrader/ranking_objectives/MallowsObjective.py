# EXACT objective

from math import exp
from math import fabs
import itertools as it


class MallowsObjective:
	"""
	Mallows objective function
	"""

	@staticmethod
	def rankByMallows(q_examples, q_scores, malOptions, userrel_scores, doc_weights = None):
		mallowScores = {}
		mallowKScores = {}
		borda = malOptions.get("borda",False)
		usegt = malOptions.get("usegt",True)
		kemen = malOptions.get("kemen",False)
		margin = malOptions.get("margin",False)
		
		hasWeights = (doc_weights != None )
		
		
		for qid in q_examples:
			ex_list = q_examples[qid]
			
			doc_allgt = {}
			doc_allgt2 = {}
			doc_alllt_count = {}
			for doc in q_scores[qid]:
				doc_allgt[doc] = {}
				doc_allgt2[doc] = {}
				doc_alllt_count[doc] = 0.0
			
			for ex in ex_list:	#GET ALL-GTs
				uid = ex.uid
				u_rel = userrel_scores[uid]
				for doc1 in ex.all_greater_than:
					this_gt = ex.all_greater_than[doc1]
					for doc2 in this_gt:
						increm = u_rel
						if margin: increm *= (ex.scored_items[doc1] - ex.scored_items[doc2]) 
						
						doc_allgt[doc1][doc2] = doc_allgt[doc1].get(doc2,0) + increm
						temp1 = doc_allgt2[doc1].get(doc2,0) + increm
						doc_allgt2[doc1][doc2] = temp1
						doc_allgt2[doc2][doc1] = - temp1
						if usegt:
							doc_allgt[doc2][doc1] = - temp1

			if not borda:	
				for doc1 in doc_allgt:	#GET ALL-LTs and LT-COUNTS
					for doc2 in doc_allgt[doc1]:
						temp = doc_allgt[doc1][doc2]
						if hasWeights: temp *= abs( doc_weights[qid][doc1] - doc_weights[qid][doc2] )
						doc_alllt_count[doc2] += temp	
			elif borda:
				temp_dict1 = {}
				temp_dict2 = {}
				for doc1 in doc_alllt_count:
					temp_dict1[doc1] = 0.0
					temp_dict2[doc1] = 0.0
				
				for ex in ex_list:	#GET ALL-GTs
					uid = ex.uid
					u_rel = userrel_scores[uid]
					for doc1 in ex.all_greater_than:
						temp_dict1[doc1] += u_rel*ex.ranked_items[doc1]
						temp_dict2[doc1] += u_rel

				for doc1 in doc_alllt_count:
					doc_alllt_count[doc1] = float( temp_dict1[doc1] ) / float( temp_dict2[doc1] )
					if hasWeights: doc_alllt_count[doc1] *= -doc_weights[qid][doc1]
			
			mallowScores[qid] = {}	
			mallowKRanking  = []
			curDocsLeft = len(doc_alllt_count)
			while curDocsLeft > 0:			
				curBestSet = []
				curBestScore = 1000000
				for doc in doc_alllt_count:	#Set best score
					curBestScore =  doc_alllt_count[doc]
					break
				
				for doc in doc_alllt_count:	#GET BEST DOCS
					val = doc_alllt_count[doc]
					if val < curBestScore:
						curBestScore = val
						curBestSet = [doc]
					elif val == curBestScore:
						curBestSet.append(doc)
				
				for doc1 in curBestSet:
					mallowKRanking.append(doc1)
					mallowScores[qid][doc1] = curDocsLeft
					doc_alllt_count.pop(doc1)

				if not borda:
					for doc1 in curBestSet:				
						for doc2 in doc_allgt[doc1]:
							if doc2 in doc_alllt_count:
								v = doc_allgt[doc1][doc2]
								if hasWeights: v *= abs( doc_weights[qid][doc1] - doc_weights[qid][doc2] )
								doc_alllt_count[doc2] -= v
						doc_allgt.pop(doc1)
				
				curDocsLeft -= len(curBestSet)	
			
			#PERFORM KEMENIZATION
			if kemen:
				swapped  = True
				numDocs = len(mallowKRanking)
				numIter = 0
				while swapped:
					numIter += 1
					swapped  = False
					tempRanking = []
					for i in range(numDocs - 1):
						d1 = mallowKRanking[i]
						d2 = mallowKRanking[i+1]
						
						if  doc_allgt2[d1].get(d2,0) < 0:
							mallowKRanking[i+1] = d1
							mallowKRanking[i] = d2
							swapped = True
				print "Swapped for ", numIter
				
				mallowKScores[qid] = {}
				for i in range(numDocs):
					mallowKScores[qid][ mallowKRanking[i] ] = numDocs - i
		
		if kemen:
			return mallowKScores
		return mallowScores
	

	@staticmethod
	def getGradient_User(obj_options, cur_scores,this_example,user_rel=1.0):
		"""
		Gets the gradient for the Mallows Objective
		"""		
		
		noallVal = False
		if obj_options.get("noallVal",False): noallVal = True
		
		#STEP1: FIRST GET the KENDALL-TAU difference
		kendall_tau = 0.0
		this_gt = this_example.all_greater_than
		for doc1 in this_gt:
			sc1 = cur_scores[doc1]			
			for doc2 in this_gt[doc1]:
				sc2 = cur_scores[doc2]
				if sc2 > sc1: kendall_tau += 1
				#elif (sc2 == sc1 and obj_options.get("use_ties",False)): kendall_tau += 0.5
		
		this_gradient =  kendall_tau
		
		
		exp_factor = exp(-user_rel)
		if not noallVal:
			allDocs = []
			for doc in this_example.tied_to:
				if doc in allDocs: continue
				for doc2 in this_example.tied_to[doc]: allDocs.append(doc2)
				
				tie_count = len(this_example.tied_to) + 1
				
				for j in range(2,tie_count+1):
					exp_term1 = pow(exp_factor,j)
					term1 = float( exp_term1 * j ) / float( 1 - exp_term1 )
					this_gradient -= term1
					term2 = float( exp_factor  ) / float( 1 - exp_factor )
					this_gradient += term2
			
		
		
		#STEP2: GET THE TERM FROM THE NORM CONST
		
		for j in range(2, len(this_gt)):
			exp_term1 = pow(exp_factor,j)
			term1 = float( exp_term1 * j ) / float( 1 - exp_term1 )
			this_gradient += term1
			term2 = float( exp_factor  ) / float( 1 - exp_factor )
			this_gradient -= term2
		
		return this_gradient