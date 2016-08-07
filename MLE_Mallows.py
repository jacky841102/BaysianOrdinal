def computeX_d(C, Sigma, g, d):
	tot = 0
	for dPrime in range(len(Sigma)):
	# for dPrime in C:
		try:
			if Sigma[g][dPrime] < Sigma[g][d]:
				tot += 1
			if Sigma[g][dPrime] > Sigma[g][d]:
				tot -= 1
		except:
			pass
	return tot

#D: assignment
#G: grader
#Sigma[g][d] the ranking of assignment, d, given by grader, g
#return: MLE ranking of mallows model,
def MLE_Mallows(D, Sigma, G):
	C = set(d for d in D)
	print(Sigma)
	rank = {}
	MLE_sigma = {}
	for i in range(len(D)):
		x = {}
		for d in C:
			x[d] = sum([computeX_d(C, Sigma, g, d) for g in G])
		# print(x)
		d_star = min(x, key=x.get)
		rank[d_star] = i          #d_star的rank是i
		# MLE_sigma[d_star] = i
		C.remove(d_star)

	for i in D:
		MLE_sigma[rank[i]] = i
	return MLE_sigma
