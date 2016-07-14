#! /usr/bin/env python

# nestedpartiton.py
#
# Description: Basic functionality for working with hierarchical riffle 
# independent models (see Huang+Guestrin, ICML 2010).  We use this framework
# to work with generalized Mallows models (which are a special case of 
# riffle independent hierarchies).
# 
# Author: Jonathan Huang

from numpy.random.mtrand import dirichlet
from numpy.random import randint,shuffle,random_integers
import permFuncs as pf;
from operator import itemgetter
import math;
import numpy as np;
import time;
import os, sys

class hierRiffle:
	def __init__(self, X):
		self.items = X; # X is a set of items
		self.isleaf = True;

	@classmethod
	def createChain(h, n, k):
		X = set(range(n))
		self = h(X);
		if k < n: 
			self.isleaf = False;
			self.A = hierRiffle.createChain(n-k,k);
			self.B = hierRiffle(set(range(n-k,n)));
		return self;

	@classmethod
	def createReverseOneChain(h, n0, n1):	
		X = set(range(n0,n1))
		self = h(X)
		if 1 < n1-n0:
			self.isleaf = False
			self.A = hierRiffle(set([n0]))
			self.B = hierRiffle.createReverseOneChain(n0+1,n1)
		return self		

	@classmethod
	def createChainFromStructure(h,sigma0):
		X = set(sigma0)
		self = h(X)
		if len(sigma0)>1:
			self.isleaf = False
			self.A = hierRiffle(set([sigma0[0]]))
			sigma0.pop(0)
			self.B = hierRiffle.createChainFromStructure(sigma0)
		return self
	
	@classmethod 
	def createBalanced(h, X, k):
		if type(X) is int:	
			X = set(range(X));
		self = h(X);
		if k < len(X):
			self.isleaf = False;
			p = len(X)/2;
			Xl = sorted(X)
			self.A = hierRiffle.createBalanced(set(Xl[0:p]),k)
			self.B = hierRiffle.createBalanced(set(Xl[p:]),k)
		return self;

	@classmethod
	def createRandom(h,X,k):
		'''note that this algorithm doesn't draw uniformly from hierarchies'''
		'''here k represents a stopping criterion for splitting'''
		if type(X) is int:	
			X = set(range(X));
		self = h(X)
		if k < len(X):
			self.isleaf = False;
			p = randint(1,len(X));
			omega = list(X);
			shuffle(omega);
			AA = set(omega[0:p])
			self.A = hierRiffle.createRandom(AA,k);
			self.B = hierRiffle.createRandom(X.difference(AA),k);
		return self;
	
	@classmethod
	def createRandomChain(h,X,k):
		if type(X) is int:
			X = set(range(X));
		self = h(X)
		if k < len(X):
			self.isleaf = False;	
			omega = list(X);
			shuffle(omega);
			AA = set(omega[0:k])	
			self.A = hierRiffle.createRandomChain(AA,k);
			self.B = hierRiffle.createRandomChain(X.difference(AA),k);
		return self

	def __repr__(self):
		return self.tostr_recursive(0);

	def __str__(self):
		return self.tostr_recursive(0);

	def tostr_recursive(self,numtabs):
		longversion = True;
		s = '\t'*numtabs + str(self.items) + '\n';
		if self.isleaf == False:
			if hasattr(self,'m'):
				#s += '\t'*numtabs + str(len(self.m)) + '\n';
				for x in self.m:
					if longversion == True:
						if self.m[x] > 0:
							s += '\t'*numtabs + ' ' + str(x) + ': ' + str(self.m[x]) + '\n'
			s += self.A.tostr_recursive(numtabs+1)
			s += self.B.tostr_recursive(numtabs+1)
		else:
			if hasattr(self,'f'):
				for x in self.f:
					if longversion == True:
						if self.f[x] > 0:
							s += '\t'*numtabs + ' ' + str(x) + ': ' + str(self.f[x]) + '\n'
		return s;

	def split(self,A,B):
		#if A.union(B) == self.items and A.isdisjoint(B) == True and self.isleaf:
		if A.union(B) == self.items and self.isleaf:
			self.A = hierRiffle(A);
			self.B = hierRiffle(B);
			self.isleaf = False;

	def __eq__(self,other):
		if self.isleaf	== True and other.isleaf == True:
			if self.items == other.items:
				return True
		elif not (self.isleaf or other.isleaf):
			if self.items == other.items:
				return (self.A == other.A and self.B == other.B) \
					or (self.A == other.B and self.B == other.A)
		return False;

	@classmethod
	def cmptop(h,X1,X2):
		if X1.isleaf == True and X2.isleaf == True:
			return X1 == X2;
		else:
			return (X1.A.items == X2.A.items and X1.B.items == X2.B.items) \
				or (X1.A.items == X2.B.items and X1.B.items == X2.A.items)

	@classmethod
	def cmpleaf(h,X1,X2):
		if X1.allLeaves() == X2.allLeaves():
			return True;
		return False;

	def deepcopy(self):
		if self.isleaf == True:
			copy = hierRiffle(self.items.copy())
			if hasattr(self,'f'):
				copy.f = self.f.copy();
			return copy;
		copy = hierRiffle(self.items.copy())
		copy.isleaf = False;
		copy.A = self.A.deepcopy();
		copy.B = self.B.deepcopy();
		if hasattr(self,'m'):
			copy.m = self.m.copy();
		return copy;
		
	def copystruct(self):	
		if self.isleaf == True:
			copy = hierRiffle(self.items.copy())
			return copy;
		copy = hierRiffle(self.items.copy())
		copy.isleaf = False;
		copy.A = self.A.deepcopy();
		copy.B = self.B.deepcopy();
		return copy;
		
	def numparams(self):
		if self.isleaf == True:
			if hasattr(self,'f'):
				return len(self.f)
		if hasattr(self,'m'):
			return len(self.m) + self.A.numparams() + self.B.numparams();
		return 0;
	
	def maxparams(self):
		if self.isleaf == True:
			if hasattr(self,'f'):
				return len(self.f);
		if hasattr(self,'m'):
			return max(len(self.m), self.A.maxparams(), self.B.maxparams());
		return 0;	

	def firstOrder(self):
		n = len(self.items);
		if self.isleaf == True:
			return np.matrix(pf.firstOrder(self.f,n));
		else:
			M = np.matrix(pf.firstOrder(self.m,n));
			F = self.A.firstOrder();
			G = self.B.firstOrder();
			XX = list(self.items); AA = list(self.A.items); BB = list(self.B.items)
			sigma = [XX.index(i) for i in AA] + [XX.index(i) for i in BB];
			P = np.matrix(pf.permMat(n,sigma));
			return M*pf.blockDiag(F,G)*np.transpose(P);
	
	def allLeaves(self):
		L = [];
		if self.isleaf == True:
			L.append(self.items);
			return L;
		return self.A.allLeaves() + self.B.allLeaves();

	def uniformParameters(self):
		n = len(self.items);
		if self.isleaf == True:
			sigmas = pf.enumerateperms(range(n));
			self.f = {};
			for x in sigmas:
				self.f[tuple(x)] = 1.0/len(sigmas);
		else:
			k = len(self.A.items);
			taus = pf.enumerateInterleavings(range(n),k);
			self.m = {};
			for x in taus:
				self.m[tuple(x)] = 1.0/len(taus);
			self.A.uniformParameters();
			self.B.uniformParameters();

	def prob(self,sigma):
		if self.isleaf == True:
			return self.f[tuple(sigma)]; 
		XX = list(self.items); AA = list(self.A.items); BB = list(self.B.items)
		tau, phiA, phiB = decomposeSigma(sigma, [XX.index(i) for i in AA],[XX.index(i) for i in BB])
		return self.A.prob(phiA)*self.B.prob(phiB)*self.m[tuple(tau)]

	def loglike(self,sigma):
		if self.isleaf == True:
			return math.log(self.f[tuple(sigma)]); 
		XX = list(self.items); AA = list(self.A.items); BB = list(self.B.items)
		tau, phiA, phiB = decomposeSigma(sigma, [XX.index(i) for i in AA],[XX.index(i) for i in BB])
		if self.m[tuple(tau)] == 0:
			print(self.A.items);
			print(self.B.items);
			print(tau)
		return self.A.loglike(phiA) + self.B.loglike(phiB) + math.log(self.m[tuple(tau)])

	def loglike_partial(self, siginv):
		Hcond = self.deepcopy();
		Hcond.topkcondition(siginv, option = 'unnormalized');
		try:
			return math.log(Hcond.normalization());
		except ValueError:
			print(siginv);

	def normalization(self):
		if self.isleaf == True:
			return sum([self.f[x] for x in self.f])
		Z_AB = self.A.normalization()*self.B.normalization();
		return Z_AB*sum([self.m[x] for x in self.m]);

	def logZ(self):
		if self.isleaf == True:
			return math.log(sum([self.f[x] for x in self.f]));
		Z_AB = self.A.logZ() + self.B.logZ();
		return Z_AB + math.log(sum([self.m[x] for x in self.m]));
	
	def ll(self, D):
		l = 0;
		for x in D:
			l += D[x]*self.loglike(x);
		return l;
		
	def ll_partial(self, D):
		l = 0;
		for x in D:
			l += D[x]*self.loglike_partial(x);
		return l
	
	def collapse(self):
		n = len(self.items);
		sigmas = pf.enumerateperms(range(n));
		H = hierRiffle(self.items)
		H.f = {};
		for x in sigmas:
			H.f[tuple(x)] = self.prob(x)
		return H;

	# note: this function returns draws in one-line notation rather than inverse notation
	# the output is always a list of permutations
	def random(self, numdraws):
		if self.isleaf == True:
			return pf.randdict(numdraws,self.f)
		piAs = self.A.random(numdraws)
		piBs = self.B.random(numdraws)
		taus = pf.randdict(numdraws,self.m)
		X = sorted(self.items); A = sorted(self.A.items); B = sorted(self.B.items);
		return [self.riffle(pia,pib,tau) for pia,pib,tau in zip(piAs,piBs,taus)]

	def random_partial(self, numdraws):
		sigmas = self.random(numdraws);
		n = len(self.items);
		ks = random_integers(1, n, numdraws);
		return [pf.perminv(x)[:k] for x,k in zip(sigmas, ks)];

	def riffle(self,piA, piB, tau):
		#place piA and piB in A and B indices of sigma.  then compose with tau
		# we assume that piA and piB map items to rankings.
		sigma = pf.compose(tau,list(piA) + [x+len(piA) for x in piB]);
		Alist = sorted(self.A.items);
		Blist = sorted(self.B.items);
		Xlist = sorted(self.items);
		tt = pf.perminv([Xlist.index(i) for i in Alist] + [Xlist.index(i) for i in Blist])
		return pf.compose(sigma,tt);

	def findConsistent(self, fixeditem, fixedrank):
		if self.isleaf:
			if fixeditem not in self.items:
				return self.f.keys();
			itemidx = list(self.items).index(fixeditem);
			return filter(lambda x: x[itemidx] == fixedrank, self.f);
		else:	
			if fixeditem not in self.items:
				return self.m.keys();
			p = len(self.A.items)
			if fixeditem in self.A.items:
				return filter(lambda x: fixedrank in x[:p], self.m)
			elif fixeditem in self.B.items:
				return filter(lambda x: fixedrank in x[p:], self.m)
		
	def sumConsistent(self, fixeditem, fixedrank):
		if fixeditem not in self.items:		
			return 1.0;
		keys = self.findConsistent(fixeditem, fixedrank);
		if self.isleaf:
			return sum([self.f[x] for x in keys])
		else:	
			return sum([self.m[x] for x in keys])

def decomposeSigma(sigma,A,B):
	sigA = [(sigma[a],i) for a,i in zip(list(A),range(len(A)))];
	sigB = [(sigma[b],j) for b,j in zip(list(B),range(len(B)))];
	sigA_sort = sorted(sigA, key = itemgetter(0));
	sigB_sort = sorted(sigB, key = itemgetter(0));
	tauA = [x[0] for x in sigA_sort];
	tauB = [x[0] for x in sigB_sort];
	phiA_inv = [x[1] for x in sigA_sort];
	phiB_inv = [x[1] for x in sigB_sort];
	return (tauA+tauB,pf.perminv(phiA_inv),pf.perminv(phiB_inv))

