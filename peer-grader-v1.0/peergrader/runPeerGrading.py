import sys
import time
import traceback

import operator

import logging

import Example
from math import sqrt


import gd_solver.stochasticGradientDescent

from ranking_methods.scoreAverager import *
from ranking_objectives.NormalCardinalScore import NormalCardinalScore

from ranking_objectives.MallowsObjective import MallowsObjective
from ranking_objectives.DiscreteRankingObjective import DiscreteRankingObjective
from ranking_objectives.PairsObjective import PairsObjective
from ranking_objectives.SequentialMultinomialObjective import SequentialMultinomialObjective

havescipy = True
try:
	from ranking_objectives.NormalDistObjective import NormalDistObjective
except Exception as exc:
	havescipy = False

from regularizer.L2Regularizer import L2Regularizer
from regularizer.GammaRegularizer import GammaRegularizer



def readDataAsTrain(inF, class_options):
	"""
	Reads the ranking examples (in pgf format) from a file
	"""
	logger = logging.getLogger('peergrader')	# create logger
	query_item_count = {}
	query_examples = {}
	user_reliability = {}
	user_item_count = {}
	
	numEx = 0
	for line in file(inF):
		if line.startswith("OVERALL"):
			continue
		
		if line.startswith("USER"):
			uid = line[len("USER:"):line.find("(")]
			parts = line[line.find("(")+1:line.find(")")]
			user_reliability[uid] = [float(s) for s in parts.split(",")]
			continue
		
		this_ex = Example.Example(line,class_options)
		qid = this_ex.qid
		uid = this_ex.uid
		
		if uid not in user_reliability:
			user_reliability[uid] = [1.0]
		user_item_count[uid] = user_item_count.get(uid, 0) + 1
			
		if qid not in query_item_count:
			query_item_count[qid] = {}
			query_examples[qid] = [ this_ex ]
		else:
			query_examples[qid].append(this_ex)
		
		for docid in this_ex.all_items:
			query_item_count[qid][docid] = query_item_count[qid].get(docid,0) + 1
		
		numEx += 1
	
	logger.info( "Found " + str(len(query_item_count)) + " queries and " + str(numEx) + " examples.")
	return (query_item_count, query_examples, user_reliability,user_item_count)

def writeScoresAndRelsToFile(docScores, userRels, outFPrefix):
	"""Writes the scores and user reliabilities to files"""
	
	#WRITE DOC SCORES
	with open(outFPrefix + "_docscores.txt","w") as outF:
		for qid in docScores:
			outF.write("TASK ID: "+qid+"\n")
			sortedDocScores = sorted(docScores[qid].iteritems(), key=operator.itemgetter(1), reverse=True)
			for (docid,score) in sortedDocScores:
				outF.write( docid + ' %0.4f\n' % score )
				
	#WRITE USERREL SCORES
	with open(outFPrefix + "_userrels.txt","w") as outF:
		sortedUserRels = sorted(userRels.iteritems(), key=operator.itemgetter(1), reverse=True)
		for (uid,score) in sortedUserRels:
			outF.write( uid + ' %0.4f\n' % score )
	
	return

def runPeerGradingMethods(trainF, outFPrefix, whichMethod, runOpts):
	"""MAIN Method which runs the different algorithms by invoking the other functions"""
	methods = whichMethod
	
	
	logger = logging.getLogger('peergrader')	# create logger
	logger.info("\n\n\t\tSTARTING TO PROCESS")
	
	data_options = {"handle_ties":True, "store_scores":True,"store_direct":True, "store_all":True, "store_ranks":True}
	logger.info("Reading Training Data:")
	try:
		(train_item_count, train_examples, u_rel, train_user_count) = readDataAsTrain(trainF, data_options)
	except Exception as excep:
		tb = traceback.format_exc()
		print "Incorrect data format.\n TRACEBACK: " + str(tb)
		sys.exit("Incorrect data format")
	logger.info("Successfully Finished Reading Training Data:\n\n")

	start_time = time.time()
	if "MAL" == methods:
		try:	
			obj_options = runOpts
			sgd_options = {"objective":MallowsObjective,"obj_options":obj_options, "user_reg": GammaRegularizer, "user_reg_params": (10,0.1), "eps_user": 1e-6, "start_stepSize_user": 5e-3, "batchSize_user":1, "maxSteps_user": 1000, "score_reg": L2Regularizer, "score_reg_params": (float(1)/float(9),0), "eps_doc": 1e-6, "start_stepSize_doc": 0.1, "batchSize_doc":1, "maxSteps_doc": 3000, "hasbias": False, "adaptThres":0.95, "adaptSteps":20, "numSteps": 1, "usemallows":True }
				
			logger.info("Running MAL (ITER:1)")
			(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [])	

			sgd_options["start_stepSize_doc"] /= 5
			for it in range(2,runOpts['iter']+1):
				if it==3: sgd_options["start_stepSize_user"] /= 2
				logger.info("Running MAL+G (ITER:"+ str(it)+")")
				(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [docScores, userRels, u_b, u_s] )	
			
			logger.info("Finished SGD. Writing to Files")
			writeScoresAndRelsToFile(docScores, userRels, outFPrefix)
		except Exception as excep:
			tb = traceback.format_exc()
			errStr = "EXCEPTION [MALLOWS] : " + str(type(excep)) + str(excep) +  "TB: " + str(tb)
			logger.error(errStr)					

	
	if "MALS" == methods:
		try:	
			loadAllRankTime = 0.0
			#print "GETTING ALL RANKINGS"
			start_load_time = time.time()
			for qid in train_examples:
				for ex in train_examples[qid]:
					ex.storeValidRankings()
			end_load_time = time.time()
			loadAllRankTime = end_load_time - start_load_time
			logger.info("FINISHED GETTING ALL VALID RANKINGS. LOADING TIME = " + str(loadAllRankTime))
				
			obj_options = runOpts
			sgd_options = {"objective":DiscreteRankingObjective,"obj_options":obj_options, "user_reg": GammaRegularizer, "user_reg_params": (10,0.1), "eps_user": 5e-6, "start_stepSize_user": 1e-3, "batchSize_user":1, "maxSteps_user": 300, "score_reg": L2Regularizer, "score_reg_params": (float(1)/float(9),0), "eps_doc": 1e-6, "start_stepSize_doc": 0.01, "batchSize_doc":1, "maxSteps_doc": 3000, "hasbias": False, "adaptThres":0.95, "adaptSteps":20, "numSteps": 1, "usemallows":True, "usemallows_as_init": MallowsObjective }
				
			logger.info("Running SCORE-WEIGHTED-MALLOWS (ITER:1)")
			(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [])	

			sgd_options["start_stepSize_doc"] /= 10
			for it in range(2,runOpts['iter']+1):
				if it==3: sgd_options["start_stepSize_user"] /= 2
				logger.info("Running SCORE-WEIGHTED-MALLOWS (ITER:"+ str(it)+")")
				(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [docScores, userRels, u_b, u_s] )	
			
			logger.info("Finished SGD. Writing to Files")
			writeScoresAndRelsToFile(docScores, userRels, outFPrefix)
		except Exception as excep:
			tb = traceback.format_exc()
			errStr = "EXCEPTION [SCORE-WEIGHTED-MALLOWS] : " + str(type(excep)) + str(excep) +  "TB: " + str(tb)
			logger.error(errStr)					
	
	
	if "BT" == methods:	
		try:
			obj_options = runOpts
			sgd_options = {"objective":PairsObjective,"obj_options":obj_options, "user_reg": GammaRegularizer, "user_reg_params": (10,0.1), "eps_user": 1e-6, "start_stepSize_user": 1e-3, "batchSize_user":1, "maxSteps_user": 1000, "score_reg": L2Regularizer, "score_reg_params": (float(1)/float(9),0), "eps_doc": 5e-7, "start_stepSize_doc": 0.1, "batchSize_doc":1, "maxSteps_doc": 3000, "hasbias": False, "adaptThres":0.95, "adaptSteps":20, "numSteps": 1 }
			
			logger.info("Running BRADLEY-TERRY MODEL (ITER:1)")
			(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [])	
			
			sgd_options["start_stepSize_doc"] /= 5
			for it in range(2,runOpts['iter']+1):
				if it==3: sgd_options["start_stepSize_user"] /= 2
				logger.info("Running BRADLEY-TERRY MODEL (ITER:" + str(it) +")")
				(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [docScores, userRels, u_b, u_s] )
			
			logger.info("Finished SGD. Writing to Files")
			writeScoresAndRelsToFile(docScores, userRels, outFPrefix)
			
		except Exception as excep:
			tb = traceback.format_exc()
			errStr = "EXCEPTION [BT] : " + str(type(excep)) + str(excep) +  "TB: " + str(tb)
			logger.error(errStr)	

					
	if "THUR" == methods:	
		try:
			if not havescipy: raise Exception("Missing SCIPY")
			
			obj_options = runOpts
			sgd_options = {"objective":NormalDistObjective,"obj_options":obj_options, "user_reg": GammaRegularizer, "user_reg_params": (10,0.1), "eps_user": 1e-5, "start_stepSize_user": 5e-3, "batchSize_user":1, "maxSteps_user": 300, "score_reg": L2Regularizer, "score_reg_params": (float(1)/float(9),0), "eps_doc": 5e-7, "start_stepSize_doc": 0.1, "batchSize_doc":1, "maxSteps_doc": 3000, "hasbias": False, "adaptThres":0.95, "adaptSteps":20, "numSteps": 1 }
			
			logger.info("Running THURSTONE MODEL (ITER:1)")
			(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [])	
			
			sgd_options["start_stepSize_doc"] /= 5
			for it in range(2,runOpts['iter']+1):
				if it==3: sgd_options["start_stepSize_user"] /= 2
				logger.info("Running THURSTONE MODEL (ITER:" + str(it) +")")
				(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [docScores, userRels, u_b, u_s] )
			
			logger.info("Finished SGD. Writing to Files")
			writeScoresAndRelsToFile(docScores, userRels, outFPrefix)
			
		except Exception as excep:
			tb = traceback.format_exc()
			errStr = "EXCEPTION [THURSTONE] : " + str(type(excep)) + str(excep) +  "TB: " + str(tb)
			logger.error(errStr)	
			
		
	if "PL" == methods:	
		try:
			obj_options = runOpts
			sgd_options = {"objective":SequentialMultinomialObjective,"obj_options":obj_options, "user_reg": GammaRegularizer, "user_reg_params": (10,0.1), "eps_user": 1e-6, "start_stepSize_user": 1e-3, "batchSize_user":1, "maxSteps_user": 1000, "score_reg": L2Regularizer, "score_reg_params": (float(1)/float(9),0), "eps_doc": 5e-7, "start_stepSize_doc": 0.1, "batchSize_doc":1, "maxSteps_doc": 3000, "hasbias": False, "adaptThres":0.95, "adaptSteps":20, "numSteps": 1 }
			
			logger.info("Running PLACKETT-LUCE MODEL (ITER:1)")
			(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [])	
			
			sgd_options["start_stepSize_doc"] /= 10
			for it in range(2,runOpts['iter']+1):
				if it==3: sgd_options["start_stepSize_user"] /= 2
				logger.info("Running PLACKETT-LUCE MODEL (ITER:" + str(it) +")")
				(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [docScores, userRels, u_b, u_s] )
			
			logger.info("Finished SGD. Writing to Files")
			writeScoresAndRelsToFile(docScores, userRels, outFPrefix)
			
		except Exception as excep:
			tb = traceback.format_exc()
			errStr = "EXCEPTION [PLACKETT-LUCE] : " + str(type(excep)) + str(excep) +  "TB: " + str(tb)
			logger.error(errStr)				

	
	if "SCAVG" == methods:
		try:
			logger.info("Running SCORE-AVERAGING")
			start_time = time.time()
			docScores = rankByScoreAveraging(train_examples, train_item_count)
			writeScoresAndRelsToFile(docScores, {}, outFPrefix)
			logger.info("FINISHED SCORE-AVERAGING")
		except Exception as excep:
			tb = traceback.format_exc()
			errStr = "EXCEPTION [SCORE-AVERAGING] : " + str(type(excep)) + str(excep) +  "TB: " + str(tb)
			logger.error(errStr)

	
	if "NCS" == methods:	
		try:
			obj_options = runOpts
			sgd_options = {"objective":NormalCardinalScore,"obj_options":obj_options, "user_reg": GammaRegularizer, "user_reg_params": (10,0.1), "eps_user": 5e-7, "start_stepSize_user": 1e-2, "batchSize_user":1, "maxSteps_user": 1000, "score_reg": L2Regularizer, "score_reg_params": (float(1)/float(9),0), "eps_doc": 1e-6, "start_stepSize_doc": 0.1, "batchSize_doc":1, "maxSteps_doc": 3000, "hasbias": True, "adaptThres":0.95, "adaptSteps":20, "numSteps": 1 }
			
			logger.info("Running NORMAL-CARDINAL-SCORE MODEL (ITER:1)")
			(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [])	
			
			sgd_options["start_stepSize_doc"] /= 5
			for it in range(2,runOpts['iter']+1):
				if it==3: sgd_options["start_stepSize_user"] /= 5
				logger.info("Running NORMAL-CARDINAL-SCORE MODEL (ITER:" + str(it) +")")
				(docScores, userRels, u_b, u_s) = gd_solver.stochasticGradientDescent.runSGD1(train_item_count, train_examples,  train_user_count, u_rel, sgd_options, True, [docScores, userRels, u_b, u_s] )
			
			logger.info("Finished SGD. Writing to Files")
			writeScoresAndRelsToFile(docScores, userRels, outFPrefix)
			
		except Exception as excep:
			tb = traceback.format_exc()
			errStr = "EXCEPTION [NCS] : " + str(type(excep)) + str(excep) +  "TB: " + str(tb)
			logger.error(errStr)	

	return
