#Mallows 1.1

'''
This module contains a series of functions for the analysis of the Mallows Model. Please see the Readme for a
detailed introduction.

'''

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from scipy.optimize import minimize
from math import log
from operator import itemgetter
from collections import defaultdict


#----------------------------------------------------------------------------------------------------------

def mallows(r,sigma,phi):
    '''This function calculates the probabilty of ranking r occuring given
    sigma (the reference ranking) and phi (the dispersion parameter)
    This is the mallows model. See Lu and Boutillier paper for more info.

    Parameters:
    -------------
    r (a preferences ranking) : list of preferences or partial preferences

    sigma (the reference ranking) : list of preferences e.g. sigma=[1,'a','b',4,5].

    phi (the dispersion parameter) : real between 0 and 1 inclusive


    Example: mallows([(1,2),(2,'a'),('a',4),(1,5),(2,5)],[5,4,'a',2,1],0.2)'''

    m=len(sigma);
    Z=Zfun(phi,m);
    P=(Z ** -1)* (phi ** disagreement(r,sigma));
    return P


#-----------------------------------------------------------------------------------------------------------

def phigraphmallows(r,sigma):
    '''This function plots the probability of ranking r occuring given a reference ranking sigma (y-axis)
    against phi (x axis)

    Parameters:
    -------------
    r (a preferences ranking) : list of preferences or partial preferences

    sigma (the reference ranking) : list of preferences e.g. sigma=[1,'a','b',4,5].

    Example: phigraphmallows([(4,3),(3,'a'),('a',1)],[1,'a',3,4]) '''
    y=np.arange(0.0, 5.0, 0.02)
    x=[mallows(r,sigma,i) for i in y]
    return plt.plot(y, x, 'ro')


#-----------------------------------------------------------------------------------------------------------


def mallowsample(sigma,p):
    '''Samples from the mallows distribution producing a preference ranking

    Parameters:
    -------------

    sigma (the reference ranking) : list of preferences e.g. sigma=[1,'a','b',4,5].

    p : a real with 0<=p<=0.5 which is a probability with 0<=p<=0.5
        the relationship between phi and p is phi=p/(1-p)
    '''

    def mallowsamplesetup(sigma,p,v):

        sigma=partialprofile(sigma)
        m=len(sigma)
        for i in range(0,m):
            if random.random()<(1-p):
                v=v|set([(sigma[i][0],sigma[i][1])])
            else:
                v=v|set([(sigma[i][1],sigma[i][0])])

        return v
    #sigma is the reference ranking in the form [1,2,3,4,5,etc] and p is a probability with 0<=p<=0.5
    v=mallowsamplesetup(sigma,p,set([]));
    while v!=tc(v):
        v=mallowsamplesetup(sigma,p,set([]))

    return constructrank(list(v))

#------------------------------------------------------------------------------------------------------------

def MLEmallows(rankings,sigma,guess=0.2):
    '''
    This Function computes the Maximum likelihood estimate of
    phi given a particular set of rankings and a reference rank

     Parameters:
    -------------

    rankings : list of rankings e.g. [[1,2,3,4],[3,2,5,4],[1,2,3,4,5]]
                which means (person a) prefers 1 to 2 to 3 to 4 and has no preference about 5
                            (person b) prefers 3 to 2 to 5 to 4 and has no preference about 1
                            (person c) prefers 1 to 2 to 3 to 4 to 5

    sigma (the reference ranking) : list of preferences e.g. sigma=[1,'a','b',4,5].

    guess : estimate of MLE (should not matter what you choose but will be slow if far away
            if not sure leave as default)

    Example: MLEmallows([[1,2,3,4],[3,2,5,4],[1,2,3,4,5]],[2,3,4,5,1],0.2)


    '''
    def completeloglike(ranking,sigma,phi):
        m=len(sigma)
        Z=Zfun(phi,m)
        sumsetup=[disagreement(partialprofile(r),sigma) for r in ranking]
        return sum([(i*log(phi))-log(Z) for i in sumsetup])
    def f(x):
        if x > 0:
            out=(-1)*(completeloglike(rankings,sigma,x))
        else:
            out=None
        return out

    return minimize(f, guess, method='nelder-mead',options={'xtol': 1e-5, 'disp': False}).x

#------------------------------------------------------------------------------------------------------------


def RIMmallows(sigma,dispersion):
    '''This function samples from the Mallows Distribution using Repeated Insertion.
    For example, RIMmallows([1,2,3,4],0.3)

     Parameters:
    -------------

    sigma (the reference ranking) : list of preferences e.g. sigma=[1,'a','b',4,5].

    phi (the dispersion parameter) : real between 0 and 1 inclusive

    '''
    r=list([])
    r.insert(0,sigma[0])
    phi=dispersion
    p=lambda i,j: ((phi)**(i-j))*((Zfun1(phi,i)) ** -1)
    p2=lambda i,j: p(i,j) * ((1 - sum([p(i,j) for j in range(i,j,-1)])) ** -1)
    m=len(sigma)
    for i in range(2,m+1):
        for j in range(i,0,-1): #j goes from j=i to j=1
            if random.random()<p2(i,j):
                r.insert(j-1,sigma[i-1])
                break
    return r



#------------------------------------------------------------------------------------------------------------

def freqcount(h):
    '''
    This function counts the number of times each ranking occurs and puts it in ascending order

     Parameters:
    -------------

    h : list of rankings e.g. [[1,2,3,4],[1,2,3,4],[1,2,3,4],[4,3,2,1]]
        which means person 1, person 2 and person 3 think 1>2>3>4
        and person 4 things 4>3>2>1

    Example: freqcount([[1,2,3,4],[1,2,3,4],[1,2,3,4],[4,3,2,1]])
            =[[[1, 2, 3, 4], 3], [[4, 3, 2, 1], 1]]



    '''
    def counting(x):
        x=list(x)
        m=len(x)
        out=0
        l=list([])
        for i in range(0,m):
            if list(x[i]) not in l:
                out=out+1
                l.append(list(x[i]))
        return l
    h=[list(i) for i in h]
    l=counting(h)
    output=[h.count(j) for j in l]
    data=[[l[i],output[i]] for i in range(len(l))]
    return sorted(data, key=itemgetter(1),reverse=True)


#------------------------------------------------------------------------------------------------------------

def MLEsigma(r,t='default'):
    '''
    Finds the minimum disagreement permutation of t
    given rankings r by looking for the sigma element of permutations(t) with the least kemeny distance

    Parameters
    -------------

    r : list of rankings e.g. [[1,2,3,4],[3,2,5,4],[1,2,3,4,5]]
        which means (person a) prefers 1 to 2 to 3 to 4 and has no preference about 5
                    (person b) prefers 3 to 2 to 5 to 4 and has no preference about 1
                    (person c) prefers 1 to 2 to 3 to 4 to 5

    t : set of elements for which to include in the minimum rank
        (note this does not need to include all the elements which occur in the ranks)

        default: use the first ranking

    Example:
        r=[[1,2,3,4],[2,3,4,7,1],[5,4,3,2],[1,4,3,6],[1,4,3,5]]
        t=[2,3,4]

        i.e. we are only interested in the minimum disagreement rank of options 2,3,4

    MLEsigma(r,t)=[2, 4, 3]


    '''
    if t=='default':
        t=r[0]
    dist=[sum([disagreement(partialprofile(p),j) for j in r]) for p in itertools.permutations(t)]
    u=list([])
    l=dist.count(min(dist))
    perm=list(itertools.permutations(t))
    for g in range(l):
        a=min(dist)
        i=dist.index(a)
        u.append(perm[i])
        dist.pop(i)
        perm.pop(i)
    if len(u)==1:
        return list(u[0])
    else:
        return u


#------------------------------------------------------------------------------------------------------------

#Useful Functions:

phip = lambda phi: (phi)*((1+phi) ** -1) #computes phi/(1+phi)

#This function computes 1(1+phi)(1+phi+phi^2)...(1+phi+...+phi^(m-1))
Zfun = lambda phi,m: np.product([sum([(phi ** j) for j in range(0,i)]) for i in range(1,m+1)])

Zfun1=  lambda phi,m: sum([(phi ** j) for j in range(0,m)])

#------------------------------------------------------------------------------------------------------------

def tc(elements):
    '''
    This function computes the transitive closure of an array of relations. For example; the transitive closure
    of a~b, b~c, c~d can be found using tc([('a','b'),('b','c'),('c','d')]).
    The output is a set: {('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')}

    I did not write this function but copied it from:
    https://stackoverflow.com/questions/8673482/transitive-closure-python-tuples
    '''

    edges = defaultdict(set)
    # map from first element of input tuples to "reachable" second elements
    for x, y in elements: edges[x].add(y)

    for _ in range(len(elements) - 1):
        edges = defaultdict(set, (
            (k, v.union(*(edges[i] for i in v)))
            for (k, v) in edges.items()
        ))

    return set((k, i) for (k, v) in edges.items() for i in v)

#------------------------------------------------------------------------------------------------------------
def disagreement(v,sigma):
    '''
    This function calculates the disagreement between two arrays of preferences.

    Parameters:
    -------------
    v : array of sets of preferences or a list of preferences

        The input v is a collection of ordered pairs representing the (partial or not) preferences.
        For example; v=[(1,'a'),(1,3),(3,4)] means that our 'voter' prefers 'party' 1 to 'a' and prefers 'party' 1 to 3 and
        prefers 'party' 3,4.
        Note we cannot have cyclic v e.g. v=[(1,2),(2,1)] because that means our 'voter' prefers 1 to 2 and 2 to 1
        which is contradictory

    sigma : list of preferences

        sigma is the actual ranking i.e. is complete; meaning the full ranking is defined e.g 1<3<5<2<6.
        No draws are allowed.
        sigma is represented by a single array e.g. sigma=[1,'a','b',4,5] means 1 is prefered to 'a' which is prefered to 'b',etc.


    Example: v=[(1,'a'),(2,3),(3,4),(1,5),(2,5)], sigma=[5,4,3,2,1,'a'] then disagreement(v,sigma)=5
    '''
    #Input Checking:
    try: tc(v)
    except: v=partialprofile(v)
    try: tc(v)
    except:
        raise TypeError('Input should be an array of preferences')


    s=set([i for j in [list(x) for x in v] for i in j])
    t=tc(v)
    for k in s:
        assert (k,k) not in t, 'We have contradictory preferences'

    m=len(sigma);
    x=0;

    for j in range(1,m):
        for i in range(0,j):
            if (sigma[j],sigma[i]) in t:
                x=x+1;
    return x

#------------------------------------------------------------------------------------------------------------
def partialprofile(sigma):
    '''
    We take a ranking as input; output a set of all the partial preferences as a list.

    Parameters:
    ------------

    sigma : object
        sigma can be an array, list or set where for example {'a','b','c','d'}
        means a is prefered to b is prefered c, etc.

    Example: partialprofile([1,2,'a',4])
    '''
    sigma=list(sigma)
    m=len(sigma)
    v=list([])
    for i in range(0,m):
        for j in range(i+1,m):
            v.append((sigma[i],sigma[j]))
    return v


#------------------------------------------------------------------------------------------------------------
def constructrank(partial):
    '''
    Given a complete set of partial rankings this will compute the actual orderings.

    Parameters:
    ------------

    partial : list of sets

    Example:
        if you set partial=[(1, 2), (1, 3), (2, 3)]
        these is the set of partial preferences:
            1 is prefered to 2; 1 is prefered to 3, etc
            constructrank([(1, 2), (1, 3), (2, 3)])=(1, 2, 3)


    Example: constructrank([(1, 'a'), (1, 3), ('a', 3)]) = (1,'a',3). i.e. 1 is prefered to 'a', etc.

    '''

    v=setmax(partial)
    partial=list(partial)
    for p in itertools.permutations(v):
        if set(partialprofile(p))==set(partial):
            out = p
            break

    return out

#------------------------------------------------------------------------------------------------------------
def setmax(j):
    '''
    Find largest possible set using elements of all the sets in array j
    '''
    m=len(j)
    v=list([])
    for i in range(0,m):
        if j[i][0] not in v:
            v.append(j[i][0])
        if j[i][1] not in v:
            v.append(j[i][1])
    return v
print MLEmallows([[1,2,3,4],[3,2,5,4],[1,2,3,4,5]],[2,3,4,5,1],0.2)
