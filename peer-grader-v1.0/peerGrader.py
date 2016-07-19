
import argparse
import os
import time
import traceback


import logging

import sys

import peergrader.runPeerGrading
import peergrader.utilities.convertCMTXLSToPGFFormat
import peergrader.utilities.convertTSVToPGFFormat




def getLogger(fName, level):
	logger = logging.getLogger('peergrader')	# create logger
	if level == "DEBUG": logger.setLevel(logging.DEBUG)
	elif level == "INFO": logger.setLevel(logging.INFO)
	elif level == "WARNING": logger.setLevel(logging.WARNING)
	elif level == "ERROR": logger.setLevel(logging.ERROR)
	elif level == "CRITICAL": logger.setLevel(logging.CRITICAL)
	
	# add a file handler
	fh = logging.FileHandler(fName)
	
	# create a formatter and set the formatter for the handler.
	frmt = logging.Formatter('%(asctime)s   %(levelname)s   %(message)s')
	fh.setFormatter(frmt)
	
	# add the Handler to the logger
	logger.addHandler(fh)
	
	return logger

"""Main  program to parse the arguments and call the required function"""
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='This program will run the peer grading toolkit.')
	
	#Input data options
	parser.add_argument('-i', type=str, help='Input data file (PGF/TSV/CMTXLS format).', required=True, metavar='INPUTFILE')
	parser.add_argument('-f', type=str, default="PGF", help='Input data format. Options: PGF,TSV,CMTXLS', choices=['PGF','TSV','CMTXLS'], metavar='FORMAT')

	#Input data options -> Columns
	parser.add_argument('-doccol', type=int, default=1, help='Document ID column which contains the ID of the document (index starts from 1). Applicable only for TSV and CMTXLS format files.', metavar='DOCID-COLUMN')
	parser.add_argument('-grcol', type=int, default=2, help='Grader/Reviewer ID column (index starts from 1). Applicable only for TSV and CMTXLS format files.', metavar='GRADERID-COLUMN')
	parser.add_argument('-vcol', type=int, default=3, help='Data value column which contains the grade given (index starts from 1). Applicable only for TSV and CMTXLS format files.', metavar='VALUE/GRADE-COLUMN')

	#RUN Options
	parser.add_argument('-m', type=str, default="MAL", help='Choice of methods to run include MAL (Mallows Model), MALS (Mallows with Scores), BT (Bradley-Terry), THUR (Thurstone Model), PL (Plackett-Luce Model). Also included is the cardinal method: Score-Averaging (SCAVG).', choices=['MAL','MALS','BT','THUR','PL','SCAVG','NCS'], metavar='METHOD')
	parser.add_argument('-iter', type=int, default=1, help='Number of iterations for estimating reliabilities', metavar='NUM-ITERATIONS')

	#METHOD Options -> Mallows
	parser.add_argument('--borda', action='store_true', default=False, help='Use Borda Count for Mallows Model')
	parser.add_argument('--kemen', action='store_true', default=False, help='Use Kemenization for Mallows Model')

	#METHOD Options -> PAIR Models
	parser.add_argument('--all_pairs', action='store_false', default=True, help='Use All Pairs')
	parser.add_argument('--model_ties', action='store_true', default=False, help='Use variant that models ties')

	#OUTPUT Options
	parser.add_argument('-o', type=str, help='Output file prefix (two files will be generated: A scores file and a reliabilities file).', required=True, metavar='OUTPUT-PREFIX')
	parser.add_argument('-log', type=str, help='Log file path.', required=True, metavar='LOG-FILE')

	#Run Options
	parser.add_argument('-v', type=str, default="INFO", help='Level of verbosity. Options (in decreasing order): DEBUG/INFO/WARNING/ERROR/CRITICAL', metavar='VERBOSITY', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
	
	
	#Parse the arguments
	args = parser.parse_args()
	argVars= vars(args)
	
	
	#START the logger
	logger = getLogger( argVars['log'], argVars['v'] )
	
	#Log the variables read
	for k in argVars:
		logger.debug("FIELD: " + k + " | VALUE: " + str(argVars[k]))
		
	inputFile = argVars['i']

	#CONVERT DATA if needed
	if argVars['f'] == "TSV" or argVars['f']=="CMTXLS":
		docCol = argVars['doccol']
		graderCol = argVars['grcol']
		valueCol = argVars['vcol']
		
		tempOutFile = "convertedInput.pgf"

		try:
			if argVars['f']=="CMTXLS":
				peergrader.utilities.convertCMTXLSToPGFFormat.convertXLStoPGF(inputFile, tempOutFile, valueCol, graderCol, docCol)
			else:
				peergrader.utilities.convertTSVToPGFFormat.convertTSVtoPGF(inputFile, tempOutFile, valueCol, graderCol, docCol)
		except Exception as excep:
			tb = traceback.format_exc()
			print "ERROR in conversion.\n TRACEBACK:\n"+str(tb)
			sys.exit(" Error in conversion" )
			
		inputFile = tempOutFile

	#RUN THE CODE
	peergrader.runPeerGrading.runPeerGradingMethods(inputFile, argVars['o'], argVars['m'] ,argVars) 
