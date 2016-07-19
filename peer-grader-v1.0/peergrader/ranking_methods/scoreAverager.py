

def rankByScoreAveraging(q_examples, q_scores):
	avgScores = {}
	for qid in q_examples:
		ex_list = q_examples[qid]
		ex_scores = q_scores[qid]
		
		totScores = {}
		totCounts = {}
		for doc in ex_scores:
			totScores[doc] = 0.0
			totCounts[doc] = 0.0
			
		for ex in ex_list:
			for doc in ex.scored_items:
				totCounts[doc] += 1
				totScores[doc] += ex.scored_items[doc]
			
		avgScores[qid] = {}
		for doc in totScores:
			if totCounts[doc] > 0:
				avgScores[qid][doc] = float(totScores[doc])/float(totCounts[doc])
			else:
				avgScores[qid][doc] = 0.0			
	return avgScores