#! /usr/bin/env python

# mallows.py
# 
# Description: Data structure for a generalized mallows model, including routines
# for evaluating probabilities, drawing from the model, and fitting parameters
# from data using maximum likelihood.
#
# Author: Jonathan Huang

import nestedpartition as np
import permFuncs as pf
from math import pow, exp, log, fabs
import numpy 
import copy

class mallows:
	def __init__(self, phi, sigma0):
		self.phi = phi[:]
		self.sigma0 = sigma0[:]

	# note! sigma0 here is entered in `inverse notation',
	# so for example, [3,2,0,1] means that item 3 is in first place, 
	# item 2 is in second place, item 0 is in third place, and item 1 last
	# The parameter phi is equal to exp(-theta), the parameter that is normally
	# used in the mallows model
	@classmethod
	def createMallows(h,phi,sigma0):
		if type(phi) == float:
			phi = (len(sigma0)-1)*[float(phi)]
		self = h(phi,sigma0)
		self.H = np.hierRiffle.createChainFromStructure(sigma0[:]);
		self.setInterleavingParams(self.H,phi)
		return self;

	def setInterleavingParams(self,H,phi):
		if H.isleaf == True:
			sigmas = pf.enumerateperms(range(len(H.items)))
			H.f = {};
			for x in sigmas:
				H.f[tuple(x)] = 1.0/len(sigmas);
			return;
		n = len(H.items)
		taus = pf.enumerateInterleavings(range(n),1)
		H.m = {};
		for tau in taus:
			H.m[tuple(tau)] = pow(phi[0],tau[0])
		invZ = 1.0/sum([H.m[tau] for tau in H.m])
		for tau in H.m:
			H.m[tau] = H.m[tau]*invZ;
		self.setInterleavingParams(H.A,phi[0])
		self.setInterleavingParams(H.B,phi[1:])

	def __str__(self):
		s = 'phi: ' + str(self.phi) + '\n'
		s += 'sigma0: ' + str(self.sigma0) + '\n'
		n = len(self.H.items)
		if n <= 5:
			sigmas = pf.enumerateperms(range(n))
			for sig in sigmas:
				s += str(pf.perminv(sig)) + ': ' + str(self.prob(sig)) + '\n'
		return s

	def deepcopy(self):	
		copy = mallows(self.phi,self.sigma0)
		copy.H = self.H.deepcopy()
		return copy

	def draw(self,numdraws = 1):
		sigmas = self.H.random(numdraws)
		for i in range(len(sigmas)):
			sigmas[i] = pf.perminv(sigmas[i])
		if numdraws == 1:
			return sigmas[0]
		return sigmas

	def prob(self,sigma):
		return self.H.prob(pf.perminv(sigma))

	def loglike_partial(self, prankpi):
		Hcond = self.H.deepcopy()
		mallows.prankcondition(Hcond, prankpi, option = 'unnormalized')
		try:
			return log(Hcond.normalization())
		except ValueError:
			print(prankpi)

	def ll_partial(self,Dprank):
		l = 0
		for x in Dprank:
			l += self.loglike_partial(x)
		return l

	@classmethod 
	def fitMallows(h,n,D):
		pi0 = mallows.consensus(n,D)
		phi = mallows.fitPhi(pi0,D)
		return (pi0,phi)

	# for this next method, instead of being written in dictionary form, D is a list of
	# partial rankings with possible repeats
	@classmethod
	def fitMallows_prank(h, n, D, Pinit, numiter, verbosity = True):
		m = 5
		convtol = .1
		ll = 0
		for iter in range(numiter):	
			Dfull = {}
			if verbosity == True:
				print('Iteration ' + str(iter))
				print('  +E-step...')
			for x in D:
				Pcond = Pinit.deepcopy()
				mallows.prankcondition(Pcond.H, x, option = 'normalized')	
				samps = Pcond.draw(m)
				for s in samps:	
					pf.dict_inc(Dfull,tuple(s),1)
			if verbosity == True:
				print('  +M-step...')
			(sigfit,phifit) = mallows.fitMallows(n,Dfull)
			Pinit = mallows.createMallows(phifit,sigfit)
			llprev = ll; 
			ll = Pinit.ll_partial(D) 
			if verbosity == True:
				print('   +loglikelihood: ' + str(ll))
			if fabs(ll-llprev) < convtol: 
				break
		return Pinit		

	# here D is assumed to be in inverse form (vertical bar form)
	# and represented as a dictionary indexed by tuples
	@classmethod
	def consensus(h,n,D):
		sigma = range(n)
		Q = mallows.computeQ(n,D)
		sig0 = list(numpy.argsort(numpy.sum(Q, axis = 0)))
		bestscore = mallows.totalconsensus(sig0,D)
		while True:
			scores = (n-1)*[0]
			for i in range(n-1):
				sig = pf.adjswap(sig0,i)
				scores[i] = mallows.totalconsensus(sig,D)
			bestidx = scores.index(min(scores))
			bestsig = pf.adjswap(sig0,bestidx)
			if bestscore <= scores[bestidx]:
				break
			sig0 = bestsig[:]
			bestscore = scores[bestidx]
		return sig0	
			
	@classmethod
	def computeQ(h,n,D):
		Q = numpy.zeros((n,n))
		N = float(sum(D[x] for x in D))
		for x in D:
			for j in range(n):	
				for l in range(j+1,n):
					Q[x[j],x[l]] += float(D[x])
		return Q/N

	@classmethod
	def totalconsensus(h,pi0,D):	
		score = 0.0
		for s in D:
			score += D[s]*mallows.__kendall(s,pi0)
		return score

	@classmethod
	def fitPhi(h,pi0,D):
		n = len(pi0)
		theta = numpy.array((n-1)*[0.0])
		Vbar = numpy.array((n-1)*[0.0])
		for x in D:
			Vbar += D[x]*numpy.array(mallows.__V(x,pi0))
		m = sum([D[x] for x in D])
		Vbar /= float(m)
		for j in range(n-1):
			theta[j] = mallows.__solveFVeqn(Vbar[j],n,j+1) 
			#print(FVeqn(Vbar[j],n,j+1,phi[j]))
		return list(numpy.exp(-theta))

	# pi0 and pi1 are assumed to be written in inverse notation
	@classmethod
	def __kendall(h,pi,pi0):
		return sum(mallows.__V(pi,pi0))

	@classmethod
	def __genkendall(h,pi,pi0,theta):
		v = mallows.__V(pi,pi0)
		return sum([x*y for (x,y) in zip(v,theta)])

	# pis here is assumed to be written in inverse notation
	# this is the insertion code (the interleaving at each stage of the mallows model)
	@classmethod
	def __V(h,pi,pi0):
		V = (len(pi)-1)*[0]
		piidx = pf.perminv(list(pi))
		for j in range(len(pi)-1):
			V[j] = [piidx[item]<piidx[pi0[j]] for item in pi0[(j+1):]].count(True)
		return V

	@classmethod
	def __solveFVeqn(h,Vbar,n,j):
		(lb,ub) = (-1.0,1.0)
		fub = mallows.__FVeqn(Vbar,n,j,ub)
		flb = mallows.__FVeqn(Vbar,n,j,lb)
		while fub > 0.0:
			ub *= 2
			fub = mallows.__FVeqn(Vbar,n,j,ub)
		while flb < 0.0:
			lb *= 2
			flb = mallows.__FVeqn(Vbar,n,j,lb)
		MAXITER = 20; TOL = 1e-4; 
		itnum = 0
		while itnum < MAXITER:
			midpt = (ub+lb)/2.0
			fmid = mallows.__FVeqn(Vbar,n,j,midpt)
			if fmid == 0 or abs(ub-lb)/2.0<TOL:
				return midpt
			itnum += 1
			if fmid*fub > 0:
				ub = midpt; fub = fmid
			else:	
				lb = midpt
			#print('['+str(lb)+','+str(ub)+']: ' + str(fmid))
		return midpt
		
	@classmethod
	def __FVeqn(h,Vbar,n,j,theta):
		if theta == 0:
			return .5*(n-j)-Vbar
		return 1.0/(exp(theta)-1)-(n-j+1)/(exp((n-j+1)*theta)-1)-Vbar

	# just a debugging function, ignore
	@classmethod
	def expectedV(h,phi,j,n):
		prob = range(n-j+1)
		for i in range(n-j+1):
			prob[i] = pow(phi,i)
		invZ = 1.0/sum(prob)
		return sum([r*x*invZ for (x,r) in zip(prob,range(n-j+1))])
	
	# condition on a partial ranking which is represented as an ordered
	# list of disjoint sets.  For example, prankpi = [set((2,3)),set((0,1,3,4))]
	@classmethod
	def prankcondition(h, H, prankpi, option = 'normalized'):
		k = sum([len(x) for x in prankpi])
		if k == 0:
			return
		if H.isleaf:
			return
		curritem = list(H.A.items)[0]
		partidx = pf.findpart(prankpi,curritem)
		rankset = pf.rankset(prankpi,partidx)
		for x in H.m:
			if x[0] not in rankset:
				H.m[x] = 0
		prankpiB = copy.deepcopy(prankpi)
		prankpiB[partidx].remove(curritem)		
		if option == 'normalized':
			pf.normalizeDict(H.m)
		mallows.prankcondition(H.B, prankpiB, option)


