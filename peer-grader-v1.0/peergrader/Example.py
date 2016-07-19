import re
import itertools
import sys
import copy

class Example(object):
	"""
	This class implements a ranking example.
	It implements different ways of storing a ranking example.
	"""
	
	qid = ""
	uid = ""
	
	all_items = []	#Stores a list of all the items
	
	this_relations = []
	
	#Ties relations:
	tied_to = {}
	
	# Direct relations
	direct_greater_than = {}	# Stores the set of direct relations of greater than for all set of items
	direct_less_than = {}
	
	# Ranked relations
	ranked_items = {}	# The rank of all items 
	max_rank = 0
	
	#If scores are provided as input as well
	scored_items = {}	#Scores for all the items
	
	# All-item relations
	all_greater_than = {}	# Stores all elements this dictionary element is better than 
	all_less_than = {}
	
	#Valid rankings
	valid_rankings = set()
	
	#All rankings (accounting for ties)
	all_rankings = {}
	all_ranking_pairs = []
	all_rankings_unscored = {}
	
	def __init__(self, line_string, class_option):
		"""
		Constructor for this object.
		Format of line : QID UID RANKING
		Format of RANKING: d1 [>?=] d2 ...
		"""
		line_string = line_string.strip()
		if line_string.startswith("["):
			line_string = line_string.split(' ',1)[1]
		
		lineParts = re.split("\s+",line_string,2)
		self.qid = lineParts[0]
		
		uStr = lineParts[1]
		#self.uid = re.sub("\((\d+\.\d*|\.\d+|\d+)\)","",uStr)
		ind1 = uStr.find("(")
		if ind1 > 0:
			self.uid = uStr[:ind1]
		else:
			self.uid = uStr
		
		#tmp = re.findall("\((\d+\.\d*|\.\d+|\d+)\)",uStr)
		tmp = re.findall("([-+]*\d+\.\d*|\.\d+|\d+)[,\)]",uStr)
		#if "user_reliability" in class_option and len(tmp) > 0:
			#self.u_reliability = float(tmp[class_option["user_reliability"]])
		#else:
			#self.u_reliability = 1
		
		rankingStr = lineParts[2]
		rankingStr_wo_scores = re.sub("\s+","", re.sub("\(\s*([-+]*\d+\.\d*|\.\d+|\d+)\s*\)","",rankingStr) )
		scores = [float(s) for s in re.findall("\(\s*([-+]*\d+\.\d*|\.\d+|\d+)\s*\)",rankingStr)]
		
		self.all_items = []
		self.all_items = re.split(">|=|\?",rankingStr_wo_scores)
		doc_relations = re.findall(">|=|\?",rankingStr_wo_scores)
		
		self.tied_to = {}
		if class_option["handle_ties"]:
			self.storeTies(doc_relations)

		self.ranked_items = {}
		self.max_rank = 0
		if class_option["store_ranks"]:
			self.storeRankedItems(doc_relations)

		self.scored_items = {}
		if class_option["store_scores"]:
			self.storeScoredItems(scores)

		self.direct_greater_than = {}
		self.direct_less_than = {}			
		if class_option["store_direct"]:
			self.storeDirectRelations(doc_relations)
			
		self.all_greater_than = {}
		self.all_less_than = {}
		if class_option["store_all"]:
			self.storeAllRelations(doc_relations)
		
		self.valid_rankings = set()
		if class_option.get("store_vald_rankings",False):
			self.storeValidRankings(doc_relations)
			
		self.this_relations = doc_relations
		
		self.all_rankings = {}
		self.all_ranking_pairs = []
		self.all_rankings_unscored = {}
		
		return
	
	def printExample(self):
		"""
		Prints the attributes of the example
		"""
		print "QID: ",self.qid
		print "UID: ",self.uid
		print "All Items ", self.all_items
		print "Tied to ", self.tied_to
		print "Direct > and < ", self.direct_greater_than, self.direct_less_than
		print "Ranked Items and Max-Rank ", self.ranked_items, self.max_rank
		print "Scored Items ", self.scored_items
		print "All > and < ", self.all_greater_than, self.all_less_than
		return
	
	def strExample(self):
		"""
		Prints the attributes of the example
		"""
		s = self.qid + " " + self.uid
		
		n = len(self.this_relations)
		for i in range( n ):
			rln = self.this_relations[i]
			docid = self.all_items[i]
			score = self.scored_items.get(docid,0.0)
			
			s += " " + docid + " (" + str(score) + ") " + rln
		dn = self.all_items[n]
		scn = self.scored_items.get(dn,0.0)
		s += " " + dn + " (" + str(scn) + ")"
		
		return s
				
	def storeScoredItems(self, scores):
		"""
		Stores the scores of the documents as a dictionary
		"""
		flag = True
		if len(self.all_items) != len(scores):
			flag = False
			#sys.stderr.write("Documents and scores do not match for QID:"+self.qid+" UID:"+self.uid+"\n")
			#sys.stderr.write( str(scores)+" " + str(len(scores)) +'\n')
			#sys.stderr.write( str(self.all_items) + " " + str(len(self.all_items)) +'\n')
			#sys.exit()
		
		self.scored_items = {}
		for it, sc in itertools.izip(self.all_items,scores):
			if flag: self.scored_items[it] = sc
			else: self.scored_items[it] = - self.ranked_items.get(it,0.0)
		return
		
	def storeRankedItems(self, relations):
		"""
		This takes in the document order and creates a ranking out of this.
		"""
		cur_rank = 1
		self.ranked_items = {}
		cur_iter = iter(self.all_items)
		temp = next(cur_iter)
		self.ranked_items[ temp ] = cur_rank
		buf = 1
		for rln in relations:
			if rln == ">":
				cur_rank += buf
				buf = 1
			else:
				buf += 1
			temp = next(cur_iter)
			self.ranked_items[ temp ] = cur_rank
		self.max_rank = cur_rank
		return
	
	def helper_addToTies(self,cur_l):
		for it1 in cur_l:
			self.tied_to[it1] = list(cur_l)
			self.tied_to[it1].remove(it1)
	
	def storeTies(self, relations):
		"""
		This stores the ties in the relations
		"""
		cur_iter = iter(self.all_items)
		cur_list = [ next(cur_iter) ]
		for rln in relations:
			if rln != ">":
				cur_list.append( next(cur_iter) )
			else:
				self.helper_addToTies( cur_list )
				cur_list = [ next(cur_iter) ]
		self.helper_addToTies( cur_list )
		return
		
	def helper_addToDirect(self, cur_list, greater_than):
		for it1 in cur_list: self.direct_less_than[it1] = list(greater_than)
		for it1 in greater_than: self.direct_greater_than[it1] = list(cur_list)
	
	def storeDirectRelations(self, relations):
		"""
		This creates the direct less than and greater than relations
		"""
		cur_iter = iter(self.all_items)
		cur_list = [ next(cur_iter) ]
		greater_than = []
		for rln in relations:
			if rln != ">":
				cur_list.append( next(cur_iter) )
			else:
				self.helper_addToDirect(cur_list, greater_than)
				greater_than = list( cur_list )
				cur_list = [ next(cur_iter) ]
		self.helper_addToDirect(cur_list, greater_than)
		self.helper_addToDirect( [], cur_list)
		return
			
	def helper_addToAll(self, cur_list, greater_than):
		for it1 in cur_list: 
			self.all_less_than[it1] = list(greater_than)
			self.all_greater_than[it1] = []
		for it1 in greater_than: self.all_greater_than[it1].extend(cur_list)
	
	def storeAllRelations(self, relations):
		"""
		This creates the all less than and greater than relations
		"""
		cur_iter = iter(self.all_items)
		cur_list = [ next(cur_iter) ]
		greater_than = []
		for rln in relations:
			if rln != ">":
				cur_list.append( next(cur_iter) )
			else:
				self.helper_addToAll(cur_list, greater_than)
				greater_than.extend( cur_list )
				cur_list = [ next(cur_iter) ]
		self.helper_addToAll(cur_list, greater_than)
		return
	
	def storeValidRankings(self):		
		"""
		This takes in the document order and creates a ranking out of this.
		"""
		if len(self.valid_rankings) > 0:
			return
		
		cur_rankings = [ [] ]
		cur_tier_elements = []
		
		cur_iter = iter(self.all_items)

		cur_ele_num = 1
		cur_tier_elements.append(cur_ele_num)
		
		buf = 1
		for rln in self.this_relations:
			if rln == ">":	#Add to the rankings
				new_rankings = []
				for this_perm in itertools.permutations(cur_tier_elements):
					perm_list = list(this_perm)
					for cur_rnkng in cur_rankings:
						temp_rnkng = cur_rnkng + perm_list
						new_rankings.append(temp_rnkng)
						
				cur_rankings = new_rankings
				cur_tier_elements = []

			cur_ele_num += 1
			cur_tier_elements.append(cur_ele_num)
		
		self.valid_rankings = set()
		for this_perm in itertools.permutations(cur_tier_elements):
			perm_list = list(this_perm)
			for cur_rnkng in cur_rankings:
				temp_rnkng = cur_rnkng + perm_list
				self.valid_rankings.add(tuple(temp_rnkng))
		
		return
	
	def getAllRankings(self):
		"""This function will get the exponents used in all the mallows models"""
		
		if len(self.all_rankings) > 0:
			return
		
		#STEP 1: Get element tiers and put in dictionary
		element_order = {}

		cur_ele_num = 0
		cur_tier = 0
		
		cur_tier_elements = []
		cur_tier_elements.append(cur_ele_num)
		for rln in self.this_relations:
			if rln == ">":	#Add to the rankings
				for ele in cur_tier_elements:
					element_order[ele] = cur_tier
				cur_tier_elements = []
				cur_tier += 1

			cur_ele_num += 1
			cur_tier_elements.append(cur_ele_num)
		
		for ele in cur_tier_elements:
			element_order[ele] = cur_tier
		
		#print "A) ", element_order, self.this_relations
		
		
		numEle = cur_ele_num + 1
		numTier = cur_tier + 1
		
		emptyArr = []
		for i in range(numTier):
			emptyArr.append( [] )
		
		#STEP 2: Go through each perm and add to the list and the unscored dict
		allEles = range(numEle)
		all_rankings = {}
		all_ranking_pairs = {}
		all_rankings_unscored = {}
		for this_perm in itertools.permutations(allEles):
			#STEP 3: For each keep running count of violations
			numErrs = 0
			thisOrder = [0]*numEle
			thisPairs = []
			thisCurOrder = copy.deepcopy(emptyArr)
			
			#print  "B) ", this_perm
			
			for d1 in this_perm:
				tier1 = element_order[d1]
				thisCurOrder[tier1].append(d1)
				
				for tier2 in range(tier1+1,numTier):
					eles2 = thisCurOrder[tier2]
					temp1 = len(eles2)
					numErrs += temp1
					thisOrder[d1] -= temp1
					for d2 in eles2:
						thisOrder[d2] += 1
						thisPairs.append( (d1,d2) )
						
					
			temp1 = tuple(thisOrder)
			all_rankings[temp1] = all_rankings.get(temp1,0) + 1
			temp2 = tuple(thisPairs)
			all_ranking_pairs[temp2] = all_ranking_pairs.get(temp2,0) + 1
			all_rankings_unscored[numErrs] = all_rankings_unscored.get(numErrs,0) + 1
			
			#print  "C) ", thisOrder, "   ||   ", numErrs
			
		self.all_rankings = all_rankings
		self.all_ranking_pairs = all_ranking_pairs
		self.all_rankings_unscored = all_rankings_unscored
		
		return