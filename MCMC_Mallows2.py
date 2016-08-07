import numpy as np
import mallows as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from MLE_Mallows import MLE_Mallows
import random
import csv
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as hcluster
from sklearn.preprocessing import normalize

SAMPLE_SIZE = 50

def preprocessing(G, Sigma, i, j, reliability):
	tot = 0
	for g in G:
		# arr = sorted([Sigma[g][k] for k in Sigma[g]])
		try:
			if Sigma[g][i] < Sigma[g][j]:
				tot += reliability[g]
			if Sigma[g][i] > Sigma[g][j]:
				tot -= reliability[g]
		except:
			pass
	return tot

def ratio(x, sigma, sigmaPrime, D):
	tot = 0
	for i in D:
		for j in D:
			tmp = 0
			try:
				if sigmaPrime.index(i) < sigmaPrime.index(j):
					tmp += 1
				if sigma.index(i) < sigma.index(j):
					tmp -= 1
				tot += x[i][j] * tmp
			except:
				pass
	return np.exp(tot)

def Metropolis_Hastings(D, T, Sigma, G):
	#x = np.zeros((D,D))
	# x = [[0 for j in D] for i in D]
	x = [[i for i in D] for j in D]
	reliability = {i : 1 for i in range(SAMPLE_SIZE)} #10.0/((i+1) * 10) for i in range(30)}

	for i in D:
		for j in D:
			# print(i, j)
			x[i][j] = preprocessing(G, Sigma, i, j, reliability)
	# for i in range(len(D)):
	# 	print()
	# 	print(x[i])
	#
	# x = (normalize(x) * 5).tolist()

	sigma0 = MLE_Mallows(D, Sigma, G)
	print(sigma0)
	sigma = []
	for i in range(len(sigma0)):
		sigma.append(sigma0[i])  #MLE ordering
	keyList = []
	for i in D:
		keyList.append(i)
	print(sigma)
	burnin = 500
	#samples = np.zeros((D, D))
	samples = {i : [] for i in D}
	for t in range(T):
		sigmaPrime = ma.mallows.createMallows(0.3678, sigma).draw()
		r = ratio(x, sigma, sigmaPrime, D)
		# print(sigma, sigmaPrime)
		# print(r)
		# print("-----------------")
		p = np.random.binomial(1, min(r, 1))
		if p is 1:
			sigma = sigmaPrime
		if t % 1000 == 0:
			print(t)
		# print(sigma)
		if t >= burnin:
			for i in D:
				samples[sigma[i]].append(i)

	return samples

def main(assignment):

	num = SAMPLE_SIZE
	D = set(i for i in range(num))
	G = set(i for i in range(num))
	# D = set()
	# G = set()
	model = ma.mallows.createMallows(0.3678, [i for i in range(num)])
	Sigma = {}
	# lst = [i for i in range(num)]
	# tuples = [(i,j) for i in lst for j in lst[i+1::]]
	# assignment = {i:[] for i in range(num)}
	# while len(tuples) > 50:
	# 	i = random.randint(0,num-1)
	# 	if len(assignment[i]) > 5	: continue
	# 	assign = random.randint(0, num-1)
	# 	if i == assign or assign in assignment[i]: continue
	# 	assignment[i].append(assign)
	# 	for e in assignment[i]:
	# 		if (e,assign) in tuples or (assign,e) in tuples:
	# 			if (e,assign) in tuples:
	# 				tuples.remove((e,assign))
	# 			if (assign,e) in tuples:
	# 				tuples.remove((assign,e))


	print("-------------------------------------")
	# print(assignment)
	# for a in D:
	# 	assignment[a] = [i for i in range(max(0, a-2), min(a+3, SAMPLE_SIZE))]

	for i in range(num):
	# 	# draw = model.draw()
	# 	# Sigma[i] = {j : draw[j] for j in assignment[i]}
		try:
			Sigma[i] = {j : (j) % SAMPLE_SIZE for j in assignment[i]}
		except:
			pass


	# with open("./peer_dataset/grades_studens_all.tsv") as students_file:
	# 	students_file.readline()
	# 	reader = csv.reader(students_file, delimiter = "\t")
	# 	for line in reader:
	# 		G.add(int(line[0]))
	# 		D.add(int(line[1]))
	# 		try:
	# 			Sigma[int(line[0])][int(line[1])] = 5-int(line[2])
	# 		except:
	# 			Sigma[int(line[0])] = {}
	# 			Sigma[int(line[0])][int(line[1])] = 5-int(line[2])

	# print(D)
	# print(G)
	# print(Sigma)
	return Metropolis_Hastings(D, 3000, Sigma, G)

def expectRank(distribution):
	sum = 0
	for i in range(SAMPLE_SIZE):
		sum += i * distribution[i]
	return sum

def rank_simple(vector):
	return sorted(range(len(vector)), key=vector.__getitem__)

def kendallTauDist(vector):
	sum = 0
	for i in range(SAMPLE_SIZE):
		for j in range(i+1, SAMPLE_SIZE):
			if vector[i] > vector[j]: sum -= 1
	return sum

def normalizedKendall(vector):
	sum = 0
	for i in range(SAMPLE_SIZE):
		for j in range(i+1, SAMPLE_SIZE):
			if vector[i] > vector[j]: sum -= 1
			else: sum += 1
	return 2.0 * sum / (SAMPLE_SIZE) / (SAMPLE_SIZE-1)

def postProcessing(samples):
	hists = [0 for i in range(SAMPLE_SIZE)]

	for i in range(SAMPLE_SIZE):
		hists[i] = plt.hist(samples[i], np.arange(0,(SAMPLE_SIZE+1),1), normed=True, align='left')

	expect_ranks = [expectRank(hists[i][0]) for i in range(SAMPLE_SIZE)]
	# rank = rank_simple(expect_ranks)
	array = np.array(expect_ranks)
	temp = array.argsort()
	ranks = np.empty(len(array), int)
	ranks[temp] = np.arange(len(array))

	H_dist = [[0 for i in range(SAMPLE_SIZE)] for j in range(SAMPLE_SIZE)]
	for i in range(SAMPLE_SIZE):
	    for j in range(SAMPLE_SIZE):
	        H_dist[i][j] = sum( [np.power((np.sqrt(hists[i][0][k]) - np.sqrt(hists[j][0][k])), 2) for k in range(SAMPLE_SIZE)])

	return ranks, hists, H_dist

def closetCluster(H_dist):
	minS = 2 * SAMPLE_SIZE
	toGrade = []
	for row in range(len(H_dist)):
		arr = np.array(H_dist[row])
		arr = arr.argsort()[:5]
		tmp = sum([H_dist[row][i] for i in arr])
		if tmp < minS:
			minS = tmp
			# arr = np.append(arr, row)
			toGrade = arr
	return toGrade

def randomAssignment(assignment):
	for row in assignment:
		while True:
			if len(assignment[row]) >= 5: break
			assign = random.randint(0, SAMPLE_SIZE-1)
			if  assign in assignment[row]: continue
			assignment[row].append(assign)

def printVariance():
	for i in range(SAMPLE_SIZE):
		print(np.var(samples[i]))

n_cl = 2
# plt.figure(2)
kmenas = KMeans(n_clusters=n_cl, tol = 1e-6)

comp_count = {i : 0 for i in range(SAMPLE_SIZE)}
assignment = {i:[] for i in range(SAMPLE_SIZE)}
tmp_arr = [i for i in range(SAMPLE_SIZE)]
np.random.shuffle(tmp_arr)
for i in range(SAMPLE_SIZE // 5):
	for t in range(5 * i, 5 * i + 5):
		assignment[i].append(tmp_arr[t])
pca = PCA(n_components=2)
ranks = None
hists = None
H_dist = None
t = SAMPLE_SIZE // 5
while t < SAMPLE_SIZE:
	print("---------------- %d ----------------" % t)
	print(assignment)
	samples = main(assignment)
	# plt.figure(1)
	ranks, hists, H_dist = postProcessing(samples)
	print(ranks)
	vectors = []
	# plt.figure(2)
	for row in hists:
		vectors.append(row[0])
	# print(vectors)
	# y_pred = kmenas.fit_predict(vectors)
	# for j in range(SAMPLE_SIZE):
	# 	assignment[n_cl * i + y_pred[j]].append(j)
		# if y_pred[j] == 0:
			# assignment[2 * i].append(j)
		# else:
			# assignment[2 * i + 1].append(j)

	# clusters = np.array(hcluster.fclusterdata(vectors, 0.05, criterion="distance"))
	# print(clusters)
	print("\n".join(["".join([" {:.2f}".format(item) for item in row]) for row in H_dist]))

	eps = 2
	for r in range(SAMPLE_SIZE):
		for c in range(r+1, SAMPLE_SIZE):
			if H_dist[r][c] < eps:
				eps = H_dist[r][c]

	dbscan = DBSCAN(eps=min(np.percentile(H_dist, 5), 1.5), metric="precomputed", min_samples=2)
	y_pred = np.array(dbscan.fit_predict(H_dist))
	print(y_pred)


	for a in range(0, max(y_pred)+1):
		indexs = np.where(y_pred == a)[0]
		tmp = {k : comp_count[k] for k in indexs}
		for idx in indexs:
			c = min(tmp, key=tmp.get)
			if t >= SAMPLE_SIZE: break
			assignment[t].append(c)
			tmp[c] += 1
			comp_count[c] += 1
			del tmp[c]
			if len(assignment[t]) >= 5: t += 1
		if len(assignment[t]) >= 5:
			t += 1
	if t >= SAMPLE_SIZE: break

	if len(assignment[t]) > 0:
		tmp = comp_count.copy()
		while len(assignment[t]) < 5:
			c = min(tmp, key=tmp.get)
			arr = []
			for k in tmp:
				if tmp[k] == tmp[c]:
					arr.append(k)
			elem = np.random.choice(arr, 1, False)[0]
			if elem in assignment[t]:
				del tmp[elem]
				continue
			else:
				assignment[t].append(elem)
				comp_count[elem] += 1
				tmp[elem] += 1
			# a = random.randint(0, SAMPLE_SIZE-1)
			# if a in assignment[t] or (min(y_pred) == -1 and y_pred[a] != -1): continue
			# else: assignment[t].append(a)
		t += 1
	if t >= SAMPLE_SIZE: break

	if max(y_pred) == -1:
		tmp = comp_count.copy()
		while len(assignment[t]) < 5:
			c = min(tmp, key=tmp.get)
			arr = []
			for k in tmp:
				if tmp[k] == tmp[c]:
					arr.append(k)
			elem = np.random.choice(arr, 1, False)[0]
			if elem in assignment[t]:
				del tmp[elem]
				continue
			else:
				assignment[t].append(elem)
				comp_count[elem] += 1
				tmp[elem] += 1
		t += 1

	# if t < SAMPLE_SIZE and len(assignment[t]) > 0:
	# 	while len(assignment[t]) < 5:
	# 		a = random.randint(0, SAMPLE_SIZE)
	# 		if a in assignment[t]: continue
	# 		else: assignment[t].append(a)
	# 	t += 1


	# for a in range(max(y_pred)+1):
	# 	indexs = np.where(y_pred == a)[0]
	# 	np.random.shuffle(indexs)
	# 	if t >= SAMPLE_SIZE: break
	# 	for idx in indexs:
	# 		assignment[t].append(idx)
	# 		if len(assignment[t]) >= 5: break
	# 	if len(assignment[t]) >= 5:
	# 		t += 1
	# if t >= SAMPLE_SIZE: break
	# if len(assignment[t]) > 0:
	# 	while len(assignment[t]) < 5:
	# 		a = random.randint(0, SAMPLE_SIZE-1)
	# 		if a in assignment[t]: continue
	# 		else: assignment[t].append(a)
	# 	t += 1
	# if t >= SAMPLE_SIZE: break

	# plt.subplot(SAMPLE_SIZE/n_cl, 1, i + 1)
	# vectors = pca.fit_transform(vectors)
	# vectors = np.matrix(vectors)
	# plt.scatter(vectors[:,0], vectors[:,1], c=y_pred)

samples = main(assignment)
	# plt.figure(1)
ranks, hists, H_dist = postProcessing(samples)
# plt.show()



#for i in range(5):
#	plt.figure(i)
#	plt.hist(samples[i], np.arange(0,6.,1), normed=True, align='left')
#plt.show()
