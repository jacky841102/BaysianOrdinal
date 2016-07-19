from os import sep
import sys
import operator
import xml.etree.cElementTree as etree

import logging


def processTag(s):
	return s.replace("{urn:schemas-microsoft-com:office:spreadsheet}","")

def convertXLStoPGF(xml_file, outF, dataColumn, graderColumn, docidColumn, numRowsLeaveOut = 3):
	"""
	This function takes in a CMT XLS file (which really is XML Spreadsheet 2003) and converts it into a PGF format file to run the peer-grading functions.
	"""
	
	logger = None
	try:
		logger = logging.getLogger('peergrader')	# create logger
		logger.info(" Starting conversion ")
	except Exception as excep:
		raise excep
		#Do nothing
		blah=False
	
	allGrades = {}
	
	allDataColumns  = [dataColumn-1]
	
	tree = etree.iterparse(xml_file)
	curColInd = 1
	curRow = []
	thisData = ""
	rowCount = 0
	
	for event, elem in tree:
		thisTag = processTag(elem.tag)
		if thisTag == "Worksheet": break	#PROCESS ONLY FIRST SHEET
		
		if thisTag == "Row":
			rowCount += 1
			
			if rowCount >= numRowsLeaveOut: 
				#logger.info( str(rowCount) + ":" + str(curRow) )
				
				grader = "rvwrid_"+curRow[graderColumn-1].replace(" ","")
				thisID = "assgnid_"+curRow[docidColumn-1].replace(" ","")
				#logger.info( grader + " " + str(graderColumn) + " " + thisID + " " + str(docidColumn) + " " + str(allDataColumns[0]) + " " +curRow[allDataColumns[0]])
    
				if grader not in allGrades:
					allGrades[grader] = {}
    
				for q in allDataColumns:
					if q not in allGrades[grader]:
						allGrades[grader][q] = {}
					allGrades[grader][q][thisID] = float(curRow[q])
				
			curColInd = 1
			curRow = []
			thisData = ""
			
		if thisTag == "Cell":
			curRow.append( thisData )
			curColInd += 1
			thisData = ""
		
		if thisTag == "Data":
			thisData = elem.text
	
	
	if logger != None: logger.info( "Finished Reading XML File. Now starting conversion" )
	with open(outF,"w") as oF:
		for grader in allGrades:
			for q in allGrades[grader]:
				#query = "q_"+str(q)
				
				thisScores = allGrades[grader][q]
				sortedScores = sorted(thisScores.iteritems(), key=operator.itemgetter(1), reverse=True)
	
				(docid,score) = sortedScores[0]
				orderStr = docid + " ("+str(score)+")"
				prevScore = score
				for (docid,score) in sortedScores[1:]:
					if prevScore == score:
						orderStr += " ? "
					else:
						orderStr += " > "
					prevScore = score
					orderStr += docid + " ("+str(score)+")"
	  
				oF.write("task1 " + grader + " " + orderStr + "\n")
  
	if logger != None : logger.info( "Done with conversion" )

if __name__ == "__main__":
	logger = logging.getLogger('peergrader')	# create logger
	logger.setLevel(logging.INFO)
	
	# add a file handler
	fh = logging.FileHandler(configuration['logfile'])
	
	# create a formatter and set the formatter for the handler.
	frmt = logging.Formatter('%(asctime)s   %(levelname)s   %(message)s')
	fh.setFormatter(frmt)
	
	# add the Handler to the logger
	logger.addHandler(fh)


	convertXLStoPGF( sys.argv[1] , sys.argv[2] , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) )