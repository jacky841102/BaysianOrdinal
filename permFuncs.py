#! /usr/bin/env python

# permFuncs.py
#
# Description: basic functions for manipulating permutations
#
# Author: Jonathan Huang

import pickle
from math import log;
import numpy as np;
import re
import operator
import random

def load(fname):
	fid = open(fname,'r')
	s = fid.readlines()
	fid.close();
	D = {};
	for str in s:
		tmp = str.split(',');
		d = []
		for x in tmp[0:-2]:
			if int(x) >= 0:
				d.append(int(x));
		sigma = tuple(d);
		if sigma not in D:
			D[sigma] = int(tmp[-2])
		else:
			D[sigma] += int(tmp[-2])
	return D;

#def get_n(D):
#	return maxx([maxx(x)+1 for x in D]);

def printData(D):
	for sigma in D:
		print(str(sigma)+": "+str(D[sigma]))

def write(D,n,fname):
	fid = open(fname,'w');
	for x in D:
		for y in x:
			fid.write(str(y)+',');
		for i in range(len(x),n):
			fid.write(str(-1)+',');
		fid.write(str(D[x])+',\n');
	fid.close();

def firstOrder(D, n, option='full', ploton = False):
	M = np.zeros((n,n));
	for sigma in D:
		M += D[sigma]*permMat(n,sigma,option);
#	if ploton == True:
#		imgplot = plt.imshow(M/M.max(), cmap = plt.cm.cool, interpolation='nearest')
#		plt.show()
	return M

def permMat(n,sigma,option = 'full'):
	M = np.zeros((n,n));
	for i in range(n):
		if option == 'full':
			M[sigma[i]][i] = 1
		elif option == 'partial':
			if i in sigma:
				M[sigma.index(i)][i] = 1;
			else:
				for j in range(len(sigma),n):
					M[j][i] = 1.0/(n-len(sigma)+1)
	return M;

def blockDiag(A,B):
	Z1 = np.mat(np.zeros((A.shape[0],B.shape[1])));
	Z2 = np.mat(np.zeros((B.shape[0],A.shape[1])));
	return np.bmat('A Z1; Z2 B');

def relRanks(D,A,option = 'full'):
	'''D is a dataset of permutations of 1...n and A is some subset and this function
		marginalizes the dataset to relative rankings of item subset A'''
	D_A = {};
	if option == 'full':
		for sigma in D:
			rr_A = tuple(relrank(sigma,A));
			if rr_A not in D_A:
				D_A[rr_A] = D[sigma];
			else:
				D_A[rr_A] += D[sigma];
	if option == 'partial':
		for siginv in D:
			rrinv_A = filter(lambda x: x in A,siginv);
			if rrinv_A not in D_A:
				D_A[rrinv_A] = D[siginv];
			else:
				D_A[rrinv_A] += D[siginv];
	return D_A;

def learnDict(D,sigmas):
	f = {};
	Z = 0.0;
	for x in sigmas:
		f[tuple(x)] = 0;
	for x in D:
		f[x] += D[x]
		Z += D[x];
	for x in f:
		f[x] = f[x]/Z;
	return f;

################################################################################## MISC

def maxx(x):
	try:
		return max(x)
	except:
		return 0;

def randperm(n,m,f = 0):
	'''m is the number of random permutations to draw'''
	'''f is assumed to either be 0 or a normalized distribution'''
	sigmas = [];
	if f == 0:
		for i in range(m):
			x = range(n)
			random.shuffle(x)
			sigmas.append(x);
	else:
		allperms = enumerateperms(range(n));
		mdraws = np.random.mtrand.multinomial(m,f);
		for ind,reps in zip(range(len(f)),mdraws):
			for i in range(reps):
				sigmas.append(allperms[ind])
	return sigmas

def randdict(m,D):
	draws = [];
	mdraws = np.random.mtrand.multinomial(m,[D[x] for x in D]);
	for x,reps in zip(D, mdraws):
		for i in range(reps):
			draws.append(x)
	random.shuffle(draws);
	return draws

# setting alpha=1 means that you always want left hand cards to drop first
# setting alpha=0 means that you always want right hand cards to drop first
# tau is a list returned sorted in ascending order
def drawBiasedRiffleSubset(n,k,alpha):
	leftcnt = k;
	rightcnt = n-k;
	tau = [];
	for i in range(n-1,-1,-1):
		if leftcnt != 0 and rightcnt != 0:
			p_left = alpha*leftcnt/(alpha*leftcnt+(1-alpha)*rightcnt);
			dropleft = int(np.random.random()<p_left);
		if rightcnt == 0 or (leftcnt>0 and dropleft):		
			tau.append(i)
			leftcnt -= 1
		else:
			rightcnt -= 1
	tau.reverse()
	return tau


def normalizeDict(D):
	Z = float(sum(D.values()));
	for x in D:
		D[x] = D[x]/Z;

def enumerateperms(X):
	sigmas = [];
	if len(X) == 1:
		sigmas.append([ X[0] ]);
		return sigmas;
	X.sort(reverse = True);
	for i in X:
		Xi = X[:];
		Xi.remove(i);
		coset = enumerateperms(Xi);
		map(lambda x: x.append(i),coset)
		sigmas.extend(coset);
	return sigmas;

def enumerateInterleavings(X,k):
	n = len(X)
	prevrow = [[]]*(k+1);
	for i in range(n+1):
		currrow = [[]]*(k+1);
		for j in range(max(0,k-n+i),min(i,k)+1):
			if j==0:
				currrow[j] = np.mat(())
			elif j == i:
				currrow[j] = np.mat(X[0:i]); 
			else:
				if prevrow[j-1].shape[0] >= 1:
					tmp = np.hstack((prevrow[j-1],np.mat(X[i-1]*np.ones((prevrow[j-1].shape[0],1)))));
				else:
					tmp = np.mat(X[i-1]);
				currrow[j] = np.vstack((tmp,prevrow[j]));
		prevrow = currrow[:]
	taus = [];
	omega = set(X);
	for m in currrow[k]:
		taup = m.tolist();
		taups = [int(x) for x in taup[0]] 
		taups.extend(sorted(list(omega.difference(set(taups)))))
		taus.append(taups)
	return taus;

def randsubset(n,k):
	x = range(n)
	random.shuffle(x)
	return x[0:k]

def allksubsets(n,k):
	omega = [];	
	return omega;

def allsubsets(n):
	omega = [];
	for i in range(pow(2,n)):
		s = int2bin(i,n);
		a = [];
		for x in re.finditer('1',s):
			a.append(x.start());
		omega.append(set(a));	
	return omega;

def int2bin(n,cnt):
	return "".join([str((n >> i) & 1) for i in range(cnt-1, -1, -1)]);

def compose(pi1,pi2):
	return [pi1[x] for x in pi2];

def riffle(tau,sigp,sigq):
	return compose(tau,sigp + [x+len(sigp) for x in sigq])

def perminv(sigma):
	result = sigma[:];
	for i in range(len(sigma)):
		result[sigma[i]] = i;
	return result;

def relrank(sigma,A):
	sigA = [(sigma[a],i) for a,i in zip(list(A),range(len(A)))];
	sigA_sort = sorted(sigA, key = operator.itemgetter(0));
	phiA_inv = [x[1] for x in sigA_sort];
	return perminv(phiA_inv)

def sorted_ind(X):
	Xtup = sorted(enumerate(X), key=operator.itemgetter(1));
	return map(lambda t: t[0],Xtup);

def zeroDict(n):
	sigmas = enumerateperms(range(n))
	D = {}
	for x in sigmas:
		D[tuple(x)] = 0;
	return D;

def dictAdd(D, D2, m = 1):
	'''add m*D2 to D, assumes they have the same key set'''
	for x in D:
		D[x] += m*D2[x];

def dictUnion(D, D2):
	'''union two dictionaries assuming disjoint key sets'''
	D3 = {};
	for x in D:
		D3[x] = D[x];
	for x in D2:
		D3[x] = D2[x];
	return D3;

def dict_inc(D,x,val):
   if x not in D:
      D[x] = val;
   else:
      D[x] += val;

def makeDict(sigmas):
	D = {};
	for x in sigmas:
		dict_inc(D,tuple(x),1);
	return D;

def fill_in(siginv,n):
	X = set(range(n));
	remainder = sorted(X.difference(set(siginv)));
	np.random.shuffle(remainder);
	return tuple(siginv) + tuple(remainder);

# pi is assumed to be written in inverse notation
def adjswap(pi,i):
	pic = pi[:]
	(pic[i],pic[i+1])=(pic[i+1],pic[i])
	return pic;

# partial ranking functions

# this will return the index of the part of the partial ranking containing item i
def findpart(prankpi,item):
	for idx,part in zip(range(len(prankpi)),prankpi):
		if item in part:
			return idx
	return -1
	
# return the ranks occupied by the idx^th part in prankpi
def rankset(prankpi,idx):
	a = sum([len(x) for x in prankpi[:idx]])
	b = a + len(prankpi[idx])
	return range(a,b)

def randPrankCensor(sigma,alpha):
	n = len(sigma)
	k = random.randint(1,n-1)
	tau = drawBiasedRiffleSubset(n,k,alpha)
	tau.insert(0,-1)
	if tau[-1] != n-1:
		tau.append(n-1)
	parts = [range(x+1,y+1) for x,y in zip(tau[0:-1],tau[1:])]
	prankpi = [set([sigma[i] for i in p]) for p in parts]
	return prankpi








