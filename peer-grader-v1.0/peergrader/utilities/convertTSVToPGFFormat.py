import sys
import operator


def convertTSVtoPGF(inF, outF, dataColumn, graderColumn, docidColumn, sep = '\t'):
	"""
	This function takes in a CSV/TSV file and converts it into a PGF format file to run the peer-grading functions.
	"""
	allDataColumns  = [dataColumn-1]
	
	allGrades = {}    
	first = True
	for line in file(inF):
		if first:
			first = False
			continue
    
		line = line.strip()
		parts = line.split(sep)

		grader = "grdr_"+parts[graderColumn-1]
		thisID = "item_"+parts[docidColumn-1]
    
		if grader not in allGrades:
			allGrades[grader] = {}
    
		for q in allDataColumns:
			if q not in allGrades[grader]:
				allGrades[grader][q] = {}
			allGrades[grader][q][thisID] = float(parts[q])
      
	print "Finished Reading All. Now starting conversion"
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
	  
				oF.write("task1" + " " + grader + " " + orderStr + "\n")
  
	return