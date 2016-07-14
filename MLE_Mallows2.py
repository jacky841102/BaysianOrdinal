def computeX_d(C, Sigma, g, d):
	tot = 0
	for dPrime in range(len(Sigma)):
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
def MLE_Mallows(D, Sigma, G):
	C = set(d for d in D)
	MLE_sigma = {}
	for i in range(len(D)):
		x = {}
		for d in C:
			x[d] = sum([computeX_d(C, Sigma, g, d) for g in G])
		# print(x)
		d_star = min(x, key=x.get)
		MLE_sigma[d_star] = i
		C.remove(d_star)
	return MLE_sigma
