import random
import time
import logging

verbose = False

def runSGD1(q_item_count, q_examples, user_item_count, user_rel, sgd_options, randomInit = False, prevInp = []):
	"""
	Runs SGD to convergence for scores followed by user followed by scores (for a certain number of steps)
	"""
	logger = logging.getLogger('peergrader')
	
	userRelFlag = False
	start = time.time()
	if len(prevInp) == 0:
		userrel_scores = {}
		user_biases = {}
		user_scales = {}
		for uid in user_rel:
			userrel_scores[uid] = 1.0
			user_biases[uid] = 0.0
			user_scales[uid] = 1.0
		
		querydoc_scores = {}
		for qid in q_item_count:	#Initialize the scores
			cur_scores = {}
			if randomInit:	
				for docid in q_item_count[qid]:
					cur_scores[docid] = random.random()
				else:
					cur_scores[docid] = 0.0
			querydoc_scores[qid] = cur_scores
	else:
		(querydoc_scores, userrel_scores, user_biases, user_scales) = prevInp
		userRelFlag = True
		
	numOverallSteps = sgd_options["numSteps"]
	
	for numS in range(1, numOverallSteps + 1):
		logger.info( "---------\n\t\tSTARTING OVERALL SGD (USER+SCORES) STEP #" + str(numS) + "\n\t\t---------\n" )
		
		if userRelFlag or numS > 1:	#RUN USER_REL SGD
			logger.info( "-----\n\t\tSTARTING SGD (USER) STEP #"+ str(numS)+ "\n\t\t-----")
			(userrel_scores,user_scales) = runSGD_User( querydoc_scores, userrel_scores, user_biases, user_scales, user_item_count, q_examples, sgd_options )
	
		logger.info( "-----STARTING SGD (SCORES) STEP #"+ str(numS)+ "\n\t\t-----")
		(querydoc_scores,user_biases) = runSGD_Score( querydoc_scores, userrel_scores, user_biases, user_scales, q_item_count, q_examples, sgd_options )
	
	end = time.time()
	logger.info("[TIME] SGD took (%d) seconds" % (end-start))
	return (querydoc_scores, userrel_scores, user_biases, user_scales)
	
def runSGD_Score( querydoc_scores, userrel_scores, user_biases, user_scales, q_item_count, q_examples, sgd_options):
	"""
	Runs SGD for a given objective function
	q_item_count: Count of how many examples a document occurs in per query
	q_examples: List of all examples for a particular query
	"""
	
	logger = logging.getLogger('peergrader')
	
	if sgd_options.get("usemallows",False):
		if sgd_options.get("usemallows_as_subrout",None) != None:
			mallowScores = sgd_options["usemallows_as_subrout"].rankByMallows(q_examples, q_item_count, sgd_options["obj_options"],userrel_scores)
			sgd_options["obj_options"]["ranking"] = mallowScores
		elif sgd_options.get("usemallows_as_init",None) != None:
			mallowScores = sgd_options["usemallows_as_init"].rankByMallows(q_examples, q_item_count, sgd_options["obj_options"],userrel_scores)
			for qid in mallowScores:
				numDocs = len(mallowScores[qid])
				for docid in mallowScores[qid]:
					querydoc_scores[qid][docid] = float(2*mallowScores[qid][docid] - numDocs)*float(3)/float(numDocs)
			sgd_options["usemallows"] = False
			sgd_options["usemallows_as_init"] = None
		else:
			mallowScores = sgd_options["objective"].rankByMallows(q_examples, q_item_count, sgd_options["obj_options"],userrel_scores)
			return (mallowScores, user_biases)
	
	for qid in querydoc_scores:	#Initialize the scores
		cur_scores = querydoc_scores[qid]
		item_count = q_item_count[qid]
		example_list = q_examples[qid]
		numEx = len(example_list)
		numDocs = len(item_count)
		
		stepSize = sgd_options["start_stepSize_doc"]
		batchSize = sgd_options["batchSize_doc"]
		adaptSteps = sgd_options["adaptSteps"]
		prevNormDiffs = [1000000]*adaptSteps
		
		maxSteps = sgd_options["maxSteps_doc"]
		
		outputFlag = False
		normDiff = 1000000
		iterNum = 0
		while normDiff > (sgd_options["eps_doc"]*numDocs)  and iterNum < maxSteps:	#Stopping condition
			score_change = {}
			for docid in cur_scores:
				score_change[docid] = 0.0
				
			#Shuffle training set
			random.shuffle( example_list )
			curEx = 0
			
			while curEx < numEx:
				new_batch = example_list[curEx:curEx+batchSize]
				curEx += batchSize
			
				new_gradient = {}	#Accumulated gradient
				for this_ex in new_batch:
					ui = this_ex.uid
					user_rel = userrel_scores[ui]
					
					if sgd_options.get("hasbias", False):
						ub = user_biases[ui]
						us = user_scales[ui]
						(ex_grad, bias_grad) = sgd_options["objective"].getGradient_Doc(sgd_options["obj_options"],cur_scores,this_ex,user_rel, ub, us)
						user_biases[ui] -= stepSize*bias_grad
					else:
						ex_grad = sgd_options["objective"].getGradient_Doc(sgd_options["obj_options"],cur_scores,this_ex,user_rel)
					
					for doc in this_ex.all_items:
						regGrad = sgd_options["score_reg"].getGradient(cur_scores[doc],sgd_options["score_reg_params"], item_count[doc] )
						new_gradient[doc] = new_gradient.get(doc,0.0)  + regGrad  + ex_grad[doc]
						
				#Make gradient update
				for doc in new_gradient:
					change = stepSize * new_gradient[doc]
					cur_scores[doc] -= change
					score_change[doc] += change

				#END OF One GRADIENT STEP
		
			normDiff = 0.0
			for doc in score_change: #Get norm of all changes
				ch = score_change[doc]
				normDiff += ch*ch
			iterNum += 1
			
			if (not outputFlag) and (iterNum < 6 or iterNum % 20 == 0):
				outputFlag = True
			
			if verbose or outputFlag:
				logger.info( "[DOCSGD] QID: "+ qid+ ": ITERATION #"+ str(iterNum)+ " NORM-DIFF="+ str(normDiff)+ " STEPSIZE="+ str(stepSize) )
			
			logger.debug( "[DOCSGD] QID: "+ qid+ ": ITERATION #"+ str(iterNum)+ " NORM-DIFF="+ str(normDiff)+ " STEPSIZE="+ str(stepSize) )
			
			outputFlag = False
			curInd = iterNum%adaptSteps
			if normDiff > sgd_options["adaptThres"]*prevNormDiffs[curInd]:
				stepSize /= float(2)
				outputFlag = True
			prevNormDiffs[curInd] = normDiff
			#END OF One  OVERALL ITERATION
			
		querydoc_scores[qid] = cur_scores
		#END OF One QUERY
	
	if sgd_options.get("usemallows_as_subrout",None) != None:
		mallowScores = sgd_options["usemallows_as_subrout"].rankByMallows(q_examples, q_item_count, sgd_options["obj_options"],userrel_scores, querydoc_scores)
		return (querydoc_scores, mallowScores)
	
	return (querydoc_scores, user_biases)

def runSGD_User( querydoc_scores, userrel_scores, user_biases, user_scales, user_item_count, q_examples, sgd_options):
	"""
	Runs SGD for a given objective function
	user_item_count: Count of how many examples a user provides
	q_examples: List of all examples for a particular query
	"""
	logger = logging.getLogger('peergrader')
	
	query_ex_pairs = []
	for qid in querydoc_scores:	#Initialize the scores
		for exNum in range(len(q_examples[qid])):
			query_ex_pairs.append( (qid,exNum) )
	numEx = len(query_ex_pairs)

	stepSize = sgd_options["start_stepSize_user"]
	batchSize = sgd_options["batchSize_user"]
	adaptSteps = sgd_options["adaptSteps"]
	numU = len(user_item_count)
		
	maxSteps = sgd_options["maxSteps_user"]

	outputFlag = False
	prevNormDiffs = [1000000]*adaptSteps		
	normDiff = 1000000
	iterNum = 0
	while normDiff > (sgd_options["eps_user"]*numU) and iterNum < maxSteps:	#Stopping condition
		score_change = {}
		for uid in user_item_count:
			score_change[uid] = 0.0
				
		#Shuffle training set
		random.shuffle( query_ex_pairs )
		curEx = 0	
		while curEx < numEx:
			new_batch = query_ex_pairs[curEx:curEx+batchSize]
			curEx += batchSize
			
			new_gradient = {}	#Accumulated gradient
			for (qid, exNum) in new_batch:
				this_ex = q_examples[qid][exNum]
				uid = this_ex.uid
				user_rel = userrel_scores[uid]
				user_scale = 1.0
				
				doc_scores = querydoc_scores[qid]
				if sgd_options.get("hasbias", False):
					ub = user_biases[uid]
					user_scale = user_scales[uid]
					exGrad = sgd_options["objective"].getGradient_User(sgd_options["obj_options"],querydoc_scores[qid],this_ex,user_rel, ub, user_scale)
					regGrad = sgd_options["user_reg"].getGradient(user_rel,sgd_options["user_reg_params"], user_item_count[uid] )
				else:
					exGrad = sgd_options["objective"].getGradient_User(sgd_options["obj_options"],querydoc_scores[qid],this_ex,user_rel)
					regGrad = sgd_options["user_reg"].getGradient(user_rel,sgd_options["user_reg_params"], user_item_count[uid] )
				
				new_gradient[uid] = new_gradient.get(uid,0.0)  + regGrad  + exGrad
				
			#Make gradient update
			for uid2 in new_gradient:
				change = stepSize * new_gradient[uid2]
				if False: #sgd_options.get("hasbias", False):
					user_scales[uid2] -= change
					if user_scales[uid2] <= 0:
						user_scales[uid2] = 0.001
				else:
					userrel_scores[uid2] -= change
					if userrel_scores[uid2] <= 0:
						userrel_scores[uid2] = 0.001
					
				score_change[uid2] -= change	
			#END OF ONE GRADIENT STEP
		
		normDiff = 0.0
		for uid in score_change: #Get norm of all changes
			ch = score_change[uid]
			normDiff += ch*ch
		iterNum += 1
			
		if (not outputFlag) and (iterNum < 6 or iterNum % 20 == 0):
			outputFlag = True
		if verbose or outputFlag:
			logger.info("[USERSGD] ITERATION #"+ str(iterNum)+ " NORM-DIFF="+ str(normDiff)+ " STEPSIZE="+str(stepSize) )
		
		logger.debug("[USERSGD] ITERATION #"+ str(iterNum)+ " NORM-DIFF="+ str(normDiff)+ " STEPSIZE="+str(stepSize) )
		
		outputFlag = False
		curInd = iterNum%adaptSteps
		if normDiff > sgd_options["adaptThres"]*prevNormDiffs[curInd]:
			stepSize /= float(2)
			outputFlag = True
		prevNormDiffs[curInd] = normDiff
		#END OF One  OVERALL ITERATION


	return (userrel_scores, user_scales)
		

