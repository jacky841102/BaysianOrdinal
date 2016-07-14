import numpy as np
import mallows as ma
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from MLE_Mallows import MLE_Mallows
import random
import csv
import math

SAMPLE_SIZE = 30

def preprocessing(G, Sigma, i, j):
	tot = 0
	for g in G:
		try:
			if Sigma[g][i] < Sigma[g][j]:
				tot += 1
			if Sigma[g][i] > Sigma[g][j]:
				tot -= 1
		except:
			pass
	return tot

def ratio(x, sigma, sigmaPrime, D):
	tot = 0
	for i in D:
		for j in D:
			tmp = 0
			try:
				if sigmaPrime[i] < sigmaPrime[j]:
					tmp += 1
				if sigma[i] < sigma[j]:
					tmp -= 1
				tot += x[i][j] * tmp
			except:
				pass
	return np.exp(tot)

def Metropolis_Hastings(D, T, Sigma, G):
	#x = np.zeros((D,D))
	x = [[0 for j in D] for i in D]
	for i in D:
		for j in D:
			x[i][j] = preprocessing(G, Sigma, i, j)
	for i in range(len(D)):
		print()
		print(x[i])

	sigma0 = MLE_Mallows(D, Sigma, G)
	print(sigma0)
	sigma = [0 for i in D]
	for i in D:
		sigma[i] = sigma0[i]
	keyList = []
	for i in D:
		keyList.append(i)
	print(sigma)
	burnin = 500
	#samples = np.zeros((D, D))
	samples = {i : [] for i in D}
	for t in range(T):
		sigmaPrime = ma.mallows.createMallows(0.01, sigma).draw()
		r = ratio(x, sigma, sigmaPrime, D)
		# print(sigma, sigmaPrime)
		# print(r)
		# print("-----------------")
		p = np.random.binomial(1, min(r, 1))
		if p is 1:
			sigma = sigmaPrime
		if t >= burnin:
			print(sigma)
			for i in D:
				samples[sigma[i]].append(i)

	return samples

def main():

	num = SAMPLE_SIZE
	D = set(i for i in range(num))
	G = set(i for i in range(num))
	model = ma.mallows.createMallows(0.1, [i for i in range(num)])
	Sigma = {}
	lst = [i for i in range(num)]
	tuples = [(i,j) for i in lst for j in lst[i+1::]]
	assignment = {i:[] for i in range(num)}
	while len(tuples) > 0:
		i = random.randint(0,num-1)
		if len(assignment[i]) > 15	: continue
		assign = random.randint(0, num-1)
		if i == assign or assign in assignment[i]: continue
		assignment[i].append(assign)
		for e in assignment[i]:
			if (e,assign) in tuples or (assign,e) in tuples:
				if (e,assign) in tuples:
					tuples.remove((e,assign))
				if (assign,e) in tuples:
					tuples.remove((assign,e))


	print("-------------------------------------")
	# print(assignment)
	# for a in D:
	# 	assignment[a] = [i for i in range(max(0, a-2), min(a+3, SAMPLE_SIZE))]

	for i in range(num):
		draw = model.draw()
		Sigma[i] = {j : draw[j] for j in assignment[i]}
		# Sigma[i] = {j : j for j in assignment[i]}

	'''
	with open("./peer_dataset/grades_studens_all.tsv") as students_file:
		students_file.readline()
		reader = csv.reader(students_file, delimiter = "\t")
		for line in reader:
			G.add(int(line[0]))
			D.add(int(line[1]))
			try:
				Sigma[int(line[0])][int(line[1])] = 5-int(line[2])
			except:
				Sigma[int(line[0])] = {}
				Sigma[int(line[0])][int(line[1])] = 5-int(line[2])
	'''
	#print(D)
	#print(G)
	#print(Sigma)
	return assignment, Sigma, Metropolis_Hastings(D, 2000, Sigma, G)
assignment, Sigma, samples = main()

hists = [0 for i in range(SAMPLE_SIZE)]
for i in range(SAMPLE_SIZE):
	hists[i] = plt.hist(samples[i], np.arange(0,(SAMPLE_SIZE+1),1), normed=True, align='left')

H_dist = [[0 for i in range(SAMPLE_SIZE)] for j in range(SAMPLE_SIZE)]
for i in range(SAMPLE_SIZE):
    for j in range(SAMPLE_SIZE):
        H_dist[i][j] = sum( [(math.sqrt(hists[i][0][k]) - math.sqrt(hists[j][0][k])) ** 2 for k in range(SAMPLE_SIZE)])


#for i in range(5):
#	plt.figure(i)
#	plt.hist(samples[i], np.arange(0,6.,1), normed=True, align='left')
#plt.show()
