#!/bin/python3.12

# Compare Sieve of Kra refinements to implementations of Sieves of:
#		Erastosthenes, 
#		Atkins, 
#		Euler, 
#		Pritchard, and
#		Sundaram,
#   after they have been tuned up for better performance
# report their performance finding the primes up to n=100059960 # = 196  * 17# = 196 * 510510

#results for one run:
#MIPS=	82.916	PrimeSieveKra@7	5764697 primes upto 100059960 [2, 3, 5, 7, 11] , [100059853, 100059857, 100059863, 100059893, 100059937] 
#MIPS=	72.315	PrimeSieveKra@11	5764697 primes upto 100059960 [2, 3, 5, 7, 11] , [100059853, 100059857, 100059863, 100059893, 100059937] 
#MIPS=	83.078	SieveKra12@7	5764697 primes upto 100059960 [2, 3, 5, 7, 11] , [100059853, 100059857, 100059863, 100059893, 100059937] 
#MIPS=	72.274	SieveKra12@11	5764697 primes upto 100059960 [2, 3, 5, 7, 11] , [100059853, 100059857, 100059863, 100059893, 100059937] 
#MIPS=	49.683	SieveEratosthenes7	5764697	 primes upto 	100059960	 [2, 3, 5, 7, 11]	[100059853, 100059857, 100059863, 100059893, 100059937]
#MIPS=	45.368	SieveSundaramj	5764697	 primes upto 	100059960	 [2, 3, 5, 7, 11]	[100059853, 100059857, 100059863, 100059893, 100059937]
#MIPS=	10.322	SieveEuler	5764697	 primes upto 	100059960	 [2, 3, 5, 7, 11]	[100059853, 100059857, 100059863, 100059893, 100059937]
#MIPS=	 5.231	SieveAtkinSOj	5764697	 primes upto 	100059960	 [2, 3, 5, 7, 11]	[100059853, 100059857, 100059863, 100059893, 100059937]
#MIPS=	 0.511	SievePritchard	5764697	 primes upto 	100059960	 [2, 3, 5, 7, 11]	[100059853, 100059857, 100059863, 100059893, 100059937]

# Uses code from:
# Pritchard: https://rosettacode.org/wiki/Sieve_of_Pritchard#Python
# Euler:     https://programmingpraxis.com/2011/02/25/sieve-of-euler/ 
# Atkin:     https://gist.github.com/mineta/7840849 
#            https://stackoverflow.com/questions/21783160/sieve-of-atkin-implementation-in-python
#            https://www.geeksforgeeks.org/sieve-of-atkin/
# Sundaram:  https://www.geeksforgeeks.org/sieve-sundaram-print-primes-smaller-n/ and https://www.geeksforgeeks.org/sieve-sundaram-print-primes-smaller-n/
# ModInverse: https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/


import gc
from timeit import timeit
import math
from numba import int32, i4, int64, i8, b1
import numpy as np
from itertools import islice

import sys
#print("Example 2", file=sys.stderr)
#sys.stderr.write("Example 3") 
import time
NOW = lambda: 25557 + time.time() / 86400 # equivalent to spreadsheet NOW
from datetime import datetime
def printnow():
	deltasec=(now:=time.time())-printnow.oldsec
	printnow.oldsec=now
	dtnow=datetime.now()
	print (f"\n {dtnow}, duration={deltasec:10.3f} \n")
	sys.stderr.write(f"\n {dtnow}, duration={deltasec:10.3f} \n")
printnow.oldsec=time.time()

def printboth(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    print(*args, **kwargs)
    return(args)

from numpy import ndarray
from math import isqrt

"""
def SieveAtkinG4Gj(limit):
def SieveAtkinG4G(limit):
def SieveAtkinGHj(nmax:int)->list[int]:
def SieveAtkinGH(nmax):
def SieveAtkinSOj(limit):
def SieveAtkinSO(limit):
def SieveEuler(n):
def SievePritchard0(limit):
def SievePritchard1(limit:int):
def SieveSundaramj(n):(n):
def SieveSundaramwj(n:int)->list[int]:

functions= [


"""

"""
Sieves of Eratosthenes
0 Mark all multiples of all values from 2 to n//2
1 Only mark multiples of already known primes.
2 Start marking from the square of each known prime, +#1
3 Mark only the multiples of the primes which are less than the square root of the largest candidate. +#2
4 Separately mark evens. Harvesting only from odds, prefix 2 to the result list. +#3
5 Separately mark the odd multiples of 3. While Sieving, sieve the 1 mod 6 and 5 mod 6 skipping the 3 mod 6.  In the harvesting phase,  prefix 2 and 3 to the result list. +#4
6 Ignore evens. Don't mark them,  +#5
7 Odds only, don't allocate memory for evens.+#6
"""


def PrimeSieveKra(upToNumm:i4=100000000, pnprimorial:i4=7)->list[i4]: 
    """ Prime number sieve algorithm by David A. Kra 	
    Parameters: 
       upToNumm     The top of the range within which to find all the prime numbers
       pnprimorial  The prime number in the range 3 to 13, whose primorial to use to define the number of columns in the sieve array.
                        7 works best. Primorial n is the product of all primes <= n
    Python imports from: numba, math

    Returns:
       A list of all the prime numbers <= upToNumm
    
    Improvements over the most trivial Sieve of Eratosthenes:
      Old, well known improvements:
		#1 Only mark multiples of already known primes.
		#2 Start marking from the square of each known prime.
		#3 Mark only the multiples of sieving primes, meaning those primes which are less than the square root of the largest candidate. + #2
		#4 Separately mark evens. Harvest only from odds, prefix 2 to the result list. +#3
		#5 Separately mark the odd multiples of 3. While Sieving, sieve the 1 mod 6 and 5 mod 6 skipping the 3 mod 6
		#6 Ignore evens. Don't mark them, +#5
		#7 Odds only, don't allocate memory for evens.+#6
      New and additional improvements: 
       * Conceptually, arrange the data in a 2D array with the primorial number of columns. e.g. 7 Primorial, 7#, = 2*3*5*7 = 210 columns.
         * Actually, store the data as a 1D list
       * Initialize the result list with the primes making up the primorial.
       * Note that primes, after the first few, only appear in "do" columns.
         * "Do" columns are the columns headed by values which are relatively prime with respect to the primes making up the primorial.
       * Identify the "do" columns. 
       * Only allocate memory for and initialize the "do" columns. 
       * Only look in the "do" columns for primes whose multiples must be marked.
         * As these are discovered, add them to the result list.
       * Only mark the multiples of these primes in the "do" columns.
       * Mark multiples by iterating down each "do' column. This coded so that a future version of python could do this in parallel.
       * Find the starting point in the next "do" column by using the Modulo Multiplicative Inverse of its column heading.
       * Once marking is complete, harvest the primes which come after the square root of the sieve's highest number. 
       
    Impact:
       This sieve is faster than the sieves of Atkins, Eratosthenes, Euler, Pritchard, and Sundaram  for large upper limits.
       At 7 primorial, this sieve, compared these other sieves, uses only 23% of the memory to hold candidate numbers.
    """ 
    # was SieveKra12qh
    # qh == Quick harvest. Start harvesting while sieving.
    # b== boolean entries in possible primes. m = multiplicative inverses
    # primorial with dense list, 
    # with "dense" meaning that the entries for uninteresting do-not-do columns are absent.
    # When finished, values in the possibleprimes list are True if prime, otherwise False
    #C0,C1,C2,C3,C4,C5,C6=0,1,2,3,4,5,6
    C0:i4=0
    C1:i4=1
    C2:i4=2
    C3:i4=3
    C4:i4=4
    C5:i4=5
    C6:i4=6
    C7:i4=7
    T:b1=True
    F:b1=False
    m1G:i4=-int(10**10) # Out of Range
    oor:i4=-m1G  # Out of Range # Must be negative
    #if pnprimorial<C3: pnprimorial=C3 # could test with 2
    nprimorial:i4=C3 if pnprimorial<3 else pnprimorial
    nprimorial:i4=C7 if pnprimorial>13 else pnprimorial
    baseprimes:list[i4]= [i for i in [2, 3,  5,   7,   11,  13,  17,      19,       23] if i <=nprimorial]
    #primorials:list[i4]=           (2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870) # [math.prod(baseprimes[0:i+1]) for i in range(len(baseprimes))]
    
    primorial:i4=math.prod(baseprimes)
    
    if hasattr(upToNumm,"__iter__"): upToNumm=upToNumm[0]
    # print("upToNumm= ", upToNumm)
    upToNum:i4=max(upToNumm,100)
    upToNum:i4= primorial* ( (upToNum+primorial-1)//primorial )  # round to a whole row
    
    rows:i4=(upToNum+1)//primorial
    
    # modpriorialdo list creation
    # SLOWER:   modprimorialdo:list[i4]=[i for i in range(primorial) if kprod([i%bp for bp in baseprimes] ) ]
    modprimorialdo:list[i4]=list(range(C1,primorial,C2))
    for i in baseprimes[C1:]:
        modprimorialdo:list[i4]=[j for j in modprimorialdo if j%i]
    lmodprimorialdo:i4=len(modprimorialdo)

    # figure out a sparse subscript lookup for which columns of the 2D sparse array hold the DO elements of the full list. #
    #   e.g for [1,5] -> [-1000000000, 0,  -1000000000, -1000000000, -1000000000, 1] where 1000000000 indicates Out Of Range
    # purpose: docolinsparse[d] tells you which column handles numbers which are (d mod primorial)
    # Uninteresting entries have an out of range value that will fail any attempt to access an item in the array by using that value as a subscript
    j:i4=C0
    docolinsparse:list[i4]=[oor]*primorial
    for j in range(lmodprimorialdo):
        docolinsparse[ modprimorialdo[j] ]=j   
    # end calculate docolinsparse
    
    # calculate the modulo inverse for each do column as a long sparse list # but not as a dense list
    doinverses:list[i4]=[C0]*primorial
    doinverses[C1]=C1 #c the first is always 1.
    doinverses[primorial-C1]=primorial-C1 # The last is always itself.
    for c in modprimorialdo:
        if not doinverses[c]:
            doinverses[c]=ci=modInverse(c, primorial) # the inverse of one is the inverse of the other.
            doinverses[ci]=c
       
    # Dense list processing. Items will be the True if prime, otherwise be changed to False
    #create the list
    possibleprimes:list[b1]=[T]* (rows*lmodprimorialdo)
    lpossibleprimes:i4=len(possibleprimes)
    possibleprimes[C0]=F # 1 is not Prime 

    lenpossibleprimes:i4=len(possibleprimes) # length of the dense list
    beginlastrow:i4=primorial*(lenpossibleprimes//primorial)+1 ## in the dense
    beginlastrow:i4=primorial*(lenpossibleprimes//primorial) ## in the sparse 
    
        
    ln :i4=lenpossibleprimes # length of the dense list
    artv :i4=C0 #STart; st op; st ep #start value
    artp:i4= C0  # STart position
    op :i4=ln   # STop
    ep : int = C2 #STep
    lslice :i4= (op-artp+ep-C1 ) // ep 
    smax :i4= int(upToNum**0.5) +C1  # All factors greater than a number's square root have a matching factor less than the square root. There is no need to cast out these larger values.

    # work to the end of the row containing smax.
    rowsmax:i4=(smax+primorial-C1)//primorial
    smax:i4=rowsmax*primorial # lmodprimorialdo-1
    possmax:i4=rowsmax*lmodprimorialdo # position in dense array of smax
    
    # just to get them typed before the loop
    ddocol:i4=C0
    ddocol=C0
    artp:i4=C0
    artp=C0
    fdorowcol:list[i4]=[C0,C0]  
    isq:i4=C0
    ix2:i4=C0
    artv:i4=C0
    i:i4=C0
    j:i4=C0
    ep:i4=C0
    #startvalues:list[int]=[C0]*lmodprimorialdo
    startpositions:list[int]=[C0]*lmodprimorialdo
    
    # harvest dense list with boolean values
    # initialize
    #r=baseprimes+[0]*lenpossibleprimes
    r=baseprimes+[0]*int(1.25*upToNum/math.log(upToNum))
    rpos=len(baseprimes)-1 # position in the results

    dpos=-1
    for rowvall in range(0,smax,primorial):
        rowval=rowvall
        for dcol in range(lmodprimorialdo):
            dpos+=1
            if not possibleprimes[dpos]: continue # not a prime
            
            i=rowval+modprimorialdo[dcol]
            # harvest this prime now
            rpos+=1
            r[rpos]=i
            #
            isq=i*i
            ep=i*lmodprimorialdo #aka vertical stepdown
            isqmodprimorial:int= isq%primorial  # is (p^2 modulo n#)
            mmi:int=doinverses[i%primorial] # yes, the Modulo Multiplicative Inverse of the prime
            
            for k in range(lmodprimorialdo):
                dc=modprimorialdo[k]
                m= (  mmi * (( dc - isqmodprimorial ) %primorial)) %primorial # is the interior %primorial necessary????
                dv=isq+m*i #startvalues[k]=(dv:=isq+m*i)
                #row,fullcol=divmod(dv,primorial)
                row=(dv//primorial)
                startpositions[k]= row*lmodprimorialdo+k # should be the same as docolinsparse[fullcol]
                    
            # The following could run in parallel on CPU or GPU
            for artp in startpositions:
                possibleprimes[artp:op:ep]= [F]* ((op-artp+ep-C1 ) // ep )   
        
    # harvest dense list with boolean values
    # previously initiated:
    #     r=baseprimes+[0]*lenpossibleprimes
    #     rpos=len(baseprimes)-1
    # dpos=-1 # continue from where left off
    for rowval in range(rowval+primorial,upToNum,primorial):
        for dcol in range(lmodprimorialdo):
            dpos+=1
            if not possibleprimes[dpos]: continue
            rpos+=1
            r[rpos]=rowval+modprimorialdo[dcol]
    
    del possibleprimes
    # truncate excess elements
    r=r[0:rpos+1] 

    if r[-1] <= upToNumm: return r      

    # truncate at upToNumm
    lenr:i4=len(r) 
    for i in range(lenr-1,1,-1):
        if r[i]<=upToNumm: break 
    r=r[0:i+1]
    return r



def SieveEratosthenes0int(n: int): # all multiples, elements are the values
	C0,C1,C2=0,1,2
	ln :i4=n
	alist=list(range(ln))
	op :i4=ln
	alist[C0:C2]=[C0,C0]
	art :i4=C0
	lslice :i4=0
	for ep in range(C2,C1+ln//C2): #largest sieve at 1+n/2
	   art =ep+ep
	   lslice=(op-ep-C1)//ep
	   alist[art:op:ep]=[C0]*lslice
	alist=[i for i in range(ln) if alist[i] ]
	return alist    

def SieveEratosthenes0(n: int): # all multiples; Boolean elements
	CT,CF,C0,C1,C2=True,False,0,1,2
	ln :i4=n
	alist=[CT]*ln #list(range(ln))
	op :i4=ln
	alist[C0:C2]=[CF,CF]
	art :i4=C0
	lslice :i4=0
	for ep in range(C2,C1+ln//C2): #largest sieve at 1+n/2
	   art =ep+ep
	   lslice=(op-ep-C1)//ep
	   alist[art:op:ep]=[CF]*lslice
	alist=[i for i in range(ln) if alist[i] ]
	return alist 

def SieveEratosthenes1(n: int): # prime multiples; Boolean elements
	CT,CF,C0,C1,C2=True,False,0,1,2
	ln :i4=n
	alist=[CT]*ln #list(range(ln))
	op :i4=ln
	alist[C0:C2]=[CF,CF]
	art :i4=C0
	lslice :i4=0
	for ep in range(C2,C1+ln//C2): #largest sieve at 1+n/2
	   if not alist[ep]: continue
	   art =ep+ep
	   lslice=(op-ep-C1)//ep
	   alist[art:op:ep]=[CF]*lslice
	alist=[i for i in range(ln) if alist[i] ]
	return alist 

def SieveEratosthenes2(n: int): # start at square; prime multiples; Boolean elements
	CT,CF,C0,C1,C2=True,False,0,1,2
	ln :i4=n
	alist=[CT]*ln #list(range(ln))
	op :i4=ln
	alist[C0:C2]=[CF,CF]
	art :i4=C0
	lslice :i4=0
	for ep in range(C2,C1+ln//C2): #largest sieve at 1+n/2
	   if not alist[ep]: continue
	   art =ep*ep
	   lslice=(op-art+ep-C1)//ep # (limit - start + step - 1 ) // step 
	   alist[art:op:ep]=[CF]*lslice
	alist=[i for i in range(ln) if alist[i] ]
	return alist 

def SieveEratosthenes3(n: int): # maxsieve=sqrt(n); start at square; prime multiples; Boolean elements
	CT,CF,C0,C1,C2=True,False,0,1,2
	ln :i4=n
	alist=[CT]*ln #list(range(ln))
	op :i4=ln
	alist[C0:C2]=[CF,CF]
	art :i4=C0
	lslice :i4=0
	maxsieve:i4=int(float(ln)**0.5) +1 
	for ep in range(C2,maxsieve): #largest sieve at sqrt(n)
	   if not alist[ep]: continue
	   art =ep*ep
	   lslice=(op-art+ep-C1)//ep # (limit - start + step - 1 ) // step 
	   alist[art:op:ep]=[CF]*lslice
	alist=[i for i in range(ln) if alist[i] ]
	return alist 

def SieveEratosthenes4s(n: int): # handle evens once, but slow harvest; maxsieve=sqrt(n); start at square; prime multiples; Boolean elements
	CT,CF,C0,C1,C2,C3,C4=True,False,0,1,2,3,4
	ln :i4=n
	alist=[CT]*ln #list(range(ln))
	op :i4=ln
	alist[C0:C2]=[CF,CF]
	art :i4=C0
	lslice :i4=0
	maxsieve:i4=int(float(ln)**0.5) +1 
	# handle evens once
	ep=C2
	art=C4
	lslice= (op-art+ep-C1 )// ep # (limit - start + step - 1 ) // step 
	alist[art:op:ep]=[CF]*lslice    
	for i in range(C3,maxsieve,C2): #largest sieve at sqrt(n)
	   if not alist[i]: continue
	   ep=i+i
	   art =i*i
	   lslice=(op-art+ep-C1)//ep # (limit - start + step - 1 ) // step 
	   alist[art:op:ep]=[CF]*lslice
	alist=[i for i in range(ln) if alist[i] ]
	return alist 

def SieveEratosthenes4(n: int): # fast harvest; handle evens once; maxsieve=sqrt(n); start at square; prime multiples; Boolean elements
	CT,CF,C0,C1,C2,C3,C4=True,False,0,1,2,3,4
	ln :i4=n
	alist=[CT]*ln #list(range(ln))
	op :i4=ln
	alist[C0:C2]=[CF,CF]
	art :i4=C0
	lslice :i4=0
	maxsieve:i4=int(float(ln)**0.5) +1 
	# handle evens once
	ep=C2
	art=C4
	lslice= (op-art+ep-C1 )// ep # (limit - start + step - 1 ) // step 
	alist[art:op:ep]=[CF]*lslice    
	for i in range(C3,maxsieve,C2): #largest sieve at sqrt(n)
	   if not alist[i]: continue
	   ep=i+i
	   art =i*i
	   lslice=(op-art+ep-C1)//ep # (limit - start + step - 1 ) // step 
	   alist[art:op:ep]=[CF]*lslice
	rlist=[C2]
	rlist.extend([i for i in range(C3,ln,C2) if alist[i] ])
	return rlist 

def SieveEratosthenes5(n: int): # fast harvest; handle evens and 3's once; maxsieve=sqrt(n); start at square; prime multiples; Boolean elements
	CT,CF,C0,C1,C2,C3,C4,C5,C6=True,False,0,1,2,3,4,5,6
	ln :i4=n
	alist=[CT]*ln #list(range(ln))
	op :i4=ln
	alist[C0:C2]=[CF,CF]
	art :i4=C0
	lslice :i4=0
	maxsieve:i4=int(float(ln)**0.5) +1 
	# handle evens & 3's once
	for i in [C2]:
		ep=i
		art=i+i
		lslice= (op-art+ep-C1 )// ep # (limit - start + step - 1 ) // step 
		alist[art:op:ep]=[CF]*lslice  

	for i in [C3]:
		ep=i+i    # odd multiples only
		art=i+i+i # start at 9
		lslice= (op-art+ep-C1 )// ep # (limit - start + step - 1 ) // step 
		alist[art:op:ep]=[CF]*lslice  

	for i in range(C5,maxsieve,C2): # handle the 1 mod 6 and then the 5 mod 6 
		if not alist[i]: continue
		art =i*i  # always 1 mod 6
		ep=i*C6
		lslice=(op-art+ep-C1)//ep
		alist[art:op:ep]=[CF]*lslice
		# now start at the next multiple which is 5 mod 6
		while art%C6 != C5:
			 art+=i+i # skip evens
		lslice=(op-art+ep-C1)//ep
		alist[art:op:ep]=[CF]*lslice  
			
	rlist=[C2,C3]
	rlist.extend([i for i in range(C5,ln,C2) if alist[i] ])
	return rlist 


def SieveEratosthenes6(n: int): # ignore evens;fast harvest; handle evens and 3's once; maxsieve=sqrt(n); start at square; prime multiples; Boolean elements
	CT,CF,C0,C1,C2,C3,C4,C5,C6=True,False,0,1,2,3,4,5,6
	ln :i4=n
	alist=[CT]*ln #list(range(ln))
	op :i4=ln
	alist[C0:C3]=[CF]*C3
	art :i4=C0
	lslice :i4=0
	maxsieve:i4=int(float(ln)**0.5) +1 
	# handle evens & 3's once
	while False:
		for i in [C2]:
			ep=i
			art=i+i
			lslice= (op-art+ep-C1 )// ep # (limit - start + step - 1 ) // step 
			alist[art:op:ep]=[CF]*lslice  

	for i in [C3]:
		ep=i+i    # odd multiples only
		art=i+i+i # start at 9
		lslice= (op-art+ep-C1 )// ep # (limit - start + step - 1 ) // step 
		alist[art:op:ep]=[CF]*lslice  

	for i in range(C5,maxsieve,C2): # handle the 1 mod 6 and then the 5 mod 6 
		if not alist[i]: continue
		art =i*i  # always 1 mod 6
		ep=i*C6
		lslice=(op-art+ep-C1)//ep
		alist[art:op:ep]=[CF]*lslice
		# now start at the next multiple which is 5 mod 6
		while art%C6 != C5:
			 art+=i+i # skip evens
		lslice=(op-art+ep-C1)//ep
		alist[art:op:ep]=[CF]*lslice  
			
	rlist=[C2,C3]+[i for i in range(C5,ln,C2) if alist[i] ]
	#rlist.extend([i for i in range(C5,ln,C2) if alist[i] ])
	return rlist 


def SieveEratosthenes7(n: int): # no evens;fast harvest; 3's once; maxsieve=sqrt(n); start at square; prime multiples; Boolean elements
	CT,CF,C0,C1,C2,C3,C4,C5,C6=True,False,0,1,2,3,4,5,6
	ln :i4=n
	alist=[CT]*int(1+ln/2)  # odds only, starting with 1 at index 0
	op :i4=len(alist)
	alist[C0]=CF
	maxsieve:i4=int(float(ln)**0.5) +1 # value, not index. value = 1 + 2*index
	maxindex=(2+maxsieve)//C2

	for i in [C3]:
		ep=i   # odd multiples only
		art=C4 # start at 9
		lslice= (op-art+ep-C1 )// ep # (limit - start + step - 1 ) // step 
		alist[art:op:ep]=[CF]*lslice  

	v=3
	for i in range(C2,maxindex): # handle the 1 mod 6 and then the 5 mod 6 
		# i is the index, v is the value 
		v+=2
		if not alist[i]: continue
		vstart =v*v  # always 1 mod 6
		art= vstart//2 #(vstart-1)/2
		ep=v*C3
		lslice=(op-art+ep-C1)//ep
		alist[art:op:ep]=[CF]*lslice
		# now start at the next multiple which is 5 mod 6
		while vstart%C6 != C5:
			 vstart+=v+v # skip evens
		art= vstart//2
		lslice=(op-art+ep-C1)//ep
		alist[art:op:ep]=[CF]*lslice  
			
	rlist=[C2,C3]
	rlist.extend([i+i+1 for i in range(C2,len(alist)) if alist[i] ]) 
	return rlist 


def SieveEratosthenes7int(n: int): # no evens;fast harvest; 3's once; maxsieve=sqrt(n); start at square; prime multiples; Boolean elements
#   The array holds the values, rather than a Boolean.
	CT,CF,C0,C1,C2,C3,C4,C5,C6=True,False,0,1,2,3,4,5,6
	ln :i4=n
	#alist=[CT]*int(1+ln/2)  # Booleans for odds only, starting with 1 at index 0
	alist=[1+i+i for i in range(int(1+ln/2))] # The values themselves.
	op :i4=len(alist)
	alist[C0]=C0
	maxsieve:i4=int(float(ln)**0.5) +1 # value, not index. value = 1 + 2*index
	maxindex=(2+maxsieve)//C2

	for i in [C3]:
		ep=i   # odd multiples only
		art=C4 # start at 9
		lslice= (op-art+ep-C1 )// ep # (limit - start + step - 1 ) // step 
		alist[art:op:ep]=[C0]*lslice  
	#v=3
	for v in alist[2:maxindex]: # staring from 5, handle the 1 mod 6 and then the 5 mod 6 
		# i is the index, v is the value 
		#v+=2
		if not v: continue
		vstart =v*v  # always 1 mod 6
		art= vstart//2 #(vstart-1)/2
		ep=v*C3
		lslice=(op-art+ep-C1)//ep
		alist[art:op:ep]=[C0]*lslice
		# now start at the next multiple which is 5 mod 6
		while vstart%C6 != C5:
			 vstart+=v+v # skip evens
		art= vstart//2
		lslice=(op-art+ep-C1)//ep
		alist[art:op:ep]=[C0]*lslice  
	rlist=[C2]+[v for v in alist if v ]
	#rlist.extend([v for v in alist if v ]) 
	return rlist 


def SievePritchard(limit):
	""" Pritchard sieve of primes up to limit """
	# from https://rosettacode.org/wiki/Sieve_of_Pritchard#Python
	# modified by dakra 
	members = ndarray(limit + 1, dtype=bool)
	members.fill(False)
	members[1] = True
	steplength, prime, rtlim, nlimit = 1, 2, isqrt(limit), 2
	primes = []
	while prime <= rtlim:
		if steplength < limit:
			for w in range(1, len(members)):
				if members[w]:
					n = w + steplength
					while n <= nlimit:
						members[n] = True
						n += steplength
			steplength = nlimit

		np = 5
		mcpy = members.copy()
		for w in range(1, len(members)):
			if mcpy[w]:
				if np == 5 and w > prime:
					np = w
				n = prime * w
				if n > nlimit:
					break  # no use trying to remove items that can't even be there
				members[n] = False  # no checking necessary now

		if np < prime:
			break
		primes.append(prime)
		prime = 3 if prime == 2 else np
		nlimit = min(steplength * prime, limit)  # advance wheel limit

	newprimes = [i for i in range(2, len(members)) if members[i]]
	return sorted(primes + newprimes)

def SievePritchard1(limit:int):
	""" Pritchard sieve of primes up to limit """
	# from https://rosettacode.org/wiki/Sieve_of_Pritchard#Python
	# modified by dakra 
	C1:int=1
	C2:int=2
	C3:int=3
	C5:int=5
	F:bool=False
	T:bool=True
	#members = ndarray(limit + 1, dtype=bool)
	#members.fill(False)
	# members[1] = True
	members:list[bool]=[F]*(limit+C1)
	members[C1]=T
	#steplength, prime, rtlim, nlimit = C1, C2, isqrt(limit), C2
	steplength:int=C1 
	prime:int=C2 
	rtlim:int=isqrt(limit) 
	nlimit:int = C2
	primes:list[int] = []
	w:int
	n:int
	while prime <= rtlim:
		if steplength < limit:
			for w in range(C1, len(members)):
				if members[w]:
					n = w + steplength
					while n <= nlimit:
						members[n] = T 
						n += steplength
			steplength = nlimit

		np:int = C5
		mcpy = members.copy()
		for w in range(C1, len(members)):
			if mcpy[w]:
				if np == 5 and w > prime:
					np = w
				n = prime * w
				if n > nlimit:
					break  # no use trying to remove items that can't even be there
				members[n] = False  # no checking necessary now

		if np < prime:
			break
		primes.append(prime)
		prime = C3 if prime == C2 else np
		nlimit = min(steplength * prime, limit)  # advance wheel limit

	newprimes = [i for i in range(2, len(members)) if members[i]]
	return sorted(primes + newprimes)



def SieveEuler(n):
#  from euler18 by Mike from https://programmingpraxis.com/2011/02/25/sieve-of-euler/ 
#     modified by dakra for python3
#     requires islice from itertools
	ps = list(range(1,n,2))
	ps[0] = 0
	root_n = int(n**0.5)
	limit = int((root_n - 1)/2 + 1)
	for p in filter(None, islice(ps,limit)):
		for q in filter(None, ps[ int(((n-1)/p - 1)/2 ): int((p-1)/2) - 1 : -1]):
			ps[int((p*q - 1)/2)] = 0
	return [2] + list(filter(None, ps)) # in lieu of   [2]+[i for i in ps if i]



def SieveAtkinGH(nmax):
# from #https://gist.github.com/mineta/7840849 
	"""
	Returns a list of prime numbers below the number "nmax"
	"""
	is_prime = dict([(i, False) for i in range(5, nmax+1)])
	for x in range(1, int(math.sqrt(nmax))+1):
		for y in range(1, int(math.sqrt(nmax))+1):
			n = 4*x**2 + y**2
			if (n <= nmax) and ((n % 12 == 1) or (n % 12 == 5)):
				is_prime[n] = not is_prime[n]
			n = 3*x**2 + y**2
			if (n <= nmax) and (n % 12 == 7):
				is_prime[n] = not is_prime[n]
			n = 3*x**2 - y**2
			if (x > y) and (n <= nmax) and (n % 12 == 11):
				is_prime[n] = not is_prime[n]
	for n in range(5, int(math.sqrt(nmax))+1):
		if is_prime[n]:
			ik = 1
			while (ik * n**2 <= nmax):
				is_prime[ik * n**2] = False
				ik += 1
	primes = []
	for i in range(nmax + 1):
		if i in [0, 1, 4]: pass
		elif i in [2,3] or is_prime[i]: primes.append(i)
		else: pass
	return primes
# assert(atkin(30)==[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])


def SieveAtkinSO(limit):
#slightly modified from # https://stackoverflow.com/questions/21783160/sieve-of-atkin-implementation-in-python code by Zsolt KOVACS  
		P = [2,3]
		r = range(1,int(math.sqrt(limit))+1)
		sieve=[False]*(limit+1)
		for x in r:
			for y in r:
				xx=x*x
				yy=y*y
				xx3 = 3*xx
				n = 4*xx + yy
				if n<=limit and (n%12==1 or n%12==5) : sieve[n] = not sieve[n]
				n = xx3 + yy
				if n<=limit and n%12==7 : sieve[n] = not sieve[n]
				n = xx3 - yy
				if x>y and n<=limit and n%12==11 : sieve[n] = not sieve[n]
		for x in range(5,int(math.sqrt(limit))):
			if sieve[x]:
				xx=x*x
				for y in range(xx,limit+1,xx):
					sieve[y] = False
		#for p in range(5,limit): # original
		#    if sieve[p] : P.append(p) # original
		P=[2,3]+[p for p in range(5,limit) if sieve[p] ] # dakra replacement
		return P


def SieveAtkinG4G(limit):
# from https://www.geeksforgeeks.org/sieve-of-atkin/  by Anuj Rathore # code contributed by Smitha
	# 2 and 3 are known
	# to be prime
	
	# Initialise the sieve
	# array with False values
	sieve = [False] * (limit + 1)
	#for i in range(0, limit + 1): sieve[i] = False
 
	'''Mark sieve[n] is True if     one of the following is True:
	a) n = (4*x*x)+(y*y) has odd    number of solutions, i.e.,
	there exist odd number of     distinct pairs (x, y) that
	satisfy the equation and    n % 12 = 1 or n % 12 = 5.
	b) n = (3*x*x)+(y*y) has    odd number of solutions
	and n % 12 = 7 
	c) n = (3*x*x)-(y*y) has     odd number of solutions,
	x > y and n % 12 = 11 '''
	x = 1
	while x * x <= limit:
		y = 1
		while y * y <= limit:
 
			# Main part of
			# Sieve of Atkin
			n = (4 * x * x) + (y * y)
			if (n <= limit and (n % 12 == 1 or
								n % 12 == 5)):
				sieve[n] ^= True
 
			n = (3 * x * x) + (y * y)
			if n <= limit and n % 12 == 7:
				sieve[n] ^= True
 
			n = (3 * x * x) - (y * y)
			if (x > y and n <= limit and
					n % 12 == 11):
				sieve[n] ^= True
			y += 1
		x += 1
 
	# Mark all multiples of     # squares as non-prime
	r = 5
	while r * r <= limit:
		if sieve[r]:
			for i in range(r * r, limit+1, r * r):
				sieve[i] = False
 
		r += 1
 
		# Print primes
	# using sieve[]
	#for a in range(5, limit+1):
	#    if sieve[a]:
	#        print(a, end=" ")
	r=[i for i in range(5, limit+1) if sieve[i]]
	return [2,3]+r
 
 # SieveAtkinGH , SieveSundaram, SieveAtkinSO, SieveAtkinG4G    
def SieveSundaram(n):
# based on https://www.geeksforgeeks.org/sieve-sundaram-print-primes-smaller-n/	
# and https://en.wikipedia.org/wiki/Sieve_of_Sundaram
# see also https://iq.opengenus.org/sieve-of-sundaram/
# modified to  (a) limit sieving to spans less than sqrt(num), and (b) use numpy slicing rather than a while loop.
#import numpy

	nNew = int((n - 1) / 2);
	marked = np.ones((nNew+1), dtype=bool) # TRUE will be prime. Mark composites as FALSE
	stop=nNew+1 
	maxi=nNew+1 #original
	maxi=(int(n**0.5) - 3) // 2 + 1 # adapted from wikipedia
	maxi=2+int(nNew**0.5) # dakra 
	skipped=0
	for i in range(1, maxi):
		if not marked[i]: continue
		  # skipped+=1
		  # continue
		start= i+i+2*i*i # starts with j=i        
		step= 1+i+i # (1+2*i) is (i+(j+1)+2i(j+1)) and (i+j+2ij)
		# instead of a while loop, with test, mark, and increment. numpy slice assignment allows a scaler on the RHS.
		marked[start:stop:step]=False       
	r= [2]+[i+i+1 for i in range(1,nNew + 1) if marked[i]] # 1 is not prime
	#r.insert(0,2)
	#printalist(["skipped",skipped])
	# print(r[:20:])
	return r

# ======================================

#@njit
def SieveAtkinGHj(nmax:int)->list[int]:
# from #https://gist.github.com/mineta/7840849 
	"""
	Returns a list of prime numbers below the number "nmax"
	"""
	C0:int=0
	C1:int=1
	C2:int=2    
	C3:int=3
	C4:int=4
	C5:int=5
	C6:int=6
	C7:int=7
	C11:int=11
	C12:int=12
	T=True
	F=False
	
	is_prime = dict([(i, F) for i in range(C5, nmax+1)])
	for x in range(C1, int(math.sqrt(nmax))+1):
		for y in range(C1, int(math.sqrt(nmax))+C1):
			n = C4*x**C2 + y**C2
			if (n <= nmax) and ((n % 12 == C1) or (n % C12 == C5)):
				is_prime[n] = not is_prime[n]
			n = C3*x**C2 + y**C2
			if (n <= nmax) and (n % 12 == C7):
				is_prime[n] = not is_prime[n]
			n = 3*x**2 - y**2
			if (x > y) and (n <= nmax) and (n % C12 == C11):
				is_prime[n] = not is_prime[n]
	for n in range(C5, int(math.sqrt(nmax))+C1):
		if is_prime[n]:
			ik = C1
			while (ik * n**C2 <= nmax):
				is_prime[ik * n**C2] = F
				ik += C1
	primes = []
	for i in range(nmax + C1):
		if i in [C0, C1, C4]: pass
		elif i in [C2,C3] or is_prime[i]: primes.append(i)
		else: pass
	return primes
# assert(atkin(30)==[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

#@njit
def SieveAtkinSOj(limit):
#skightly modified from # https://stackoverflow.com/questions/21783160/sieve-of-atkin-implementation-in-python code by Zsolt KOVACS  
		C0:int=0
		C1:int=1
		C2:int=2    
		C3:int=3
		C4:int=4
		C5:int=5
		C6:int=6
		C7:int=7
		C11:int=11
		C12:int=12      
		T=True
		F=False
	
		P = [C2,C3]
		r = range(C1,int(math.sqrt(limit))+C1)
		sieve=[F]*(limit+C1)
		for x in r:
			for y in r:
				xx=x*x
				yy=y*y
				xx3 = C3*xx
				n = C4*xx + yy
				if n<=limit and (n%C12==C1 or n%C12==C5) : sieve[n] = not sieve[n]
				n = xx3 + yy
				if n<=limit and n%C12==C7 : sieve[n] = not sieve[n]
				n = xx3 - yy
				if x>y and n<=limit and n%C12==C11 : sieve[n] = not sieve[n]
		for x in range(C5,int(math.sqrt(limit))):
			if sieve[x]:
				xx=x*x
				for y in range(xx,limit+1,xx):
					sieve[y] = F 
		#for p in range(5,limit): # original
		#    if sieve[p] : P.append(p) # original
		P=[C2,C3]+[p for p in range(5,limit) if sieve[p] ] # dakra replacement
		return P

#@njit
def SieveAtkinG4Gj(limit):
# from https://www.geeksforgeeks.org/sieve-of-atkin/  by Anuj Rathore # code contributed by Smitha
	# 2 and 3 are known
	# to be prime
	
	# Initialise the sieve
	# array with False values
	
	C0:int=0
	C1:int=1
	C2:int=2    
	C3:int=3
	C4:int=4
	C5:int=5
	C6:int=6
	C7:int=7
	C11:int=11
	C12:int=12
	T=True
	F=False
	
	sieve = [F] * (limit + C1)
	#for i in range(0, limit + 1): sieve[i] = False
 
	'''Mark sieve[n] is True if     one of the following is True:
	a) n = (4*x*x)+(y*y) has odd    number of solutions, i.e.,
	there exist odd number of     distinct pairs (x, y) that
	satisfy the equation and    n % 12 = 1 or n % 12 = 5.
	b) n = (3*x*x)+(y*y) has    odd number of solutions
	and n % 12 = 7 
	c) n = (3*x*x)-(y*y) has     odd number of solutions,
	x > y and n % 12 = 11 '''
	x = C1
	while x * x <= limit:
		y = C1
		while y * y <= limit:
 
			# Main part of
			# Sieve of Atkin
			n = (C4 * x * x) + (y * y)
			if (n <= limit and (n % C12 == C1 or
								n % C12 == C5)):
				sieve[n] ^= T
 
			n = (C3 * x * x) + (y * y)
			if n <= limit and n % C12 == C7:
				sieve[n] ^= T
 
			n = (C3 * x * x) - (y * y)
			if (x > y and n <= limit and
					n % C12 == C11):
				sieve[n] ^= T
			y += C1
		x += C1
 
	# Mark all multiples of     # squares as non-prime
	r = C5
	while r * r <= limit:
		if sieve[r]:
			for i in range(r * r, limit+C1, r * r):
				sieve[i] = F
 
		r += C1
 
		# Print primes
	# using sieve[]
	#for a in range(5, limit+1):
	#    if sieve[a]:
	#        print(a, end=" ")
	r=[i for i in range(C5, limit+C1) if sieve[i]]
	return [C2,C3]+r
 

#@njit
def SieveSundaramj(n):
# based on https://www.geeksforgeeks.org/sieve-sundaram-print-primes-smaller-n/
# and https://en.wikipedia.org/wiki/Sieve_of_Sundaram
# see also https://iq.opengenus.org/sieve-of-sundaram/
# modified to  (a) limit sieving to spans less than sqrt(num), but not (b) use numpy slicing rather than a while loop.
#import numpy

	
	C0:int=0
	C1:int=1
	C2:int=2    
	C3:int=3
	C4:int=4
	C5:int=5
	C6:int=6
	C7:int=7
	C11:int=11
	C12:int=12
	T:bool=True
	F:bool=False
	#T=C1
	#F=C0

	nNew:int = int((n - C1) / C2)
	#was marked = np.ones((nNew+C1), dtype=bool) # TRUE will be prime. Mark composites as FALSE
	
	marked:list[bool]=[T]*(nNew+C1) # dakra
	 
	stop:int=nNew+C1 
	maxi:int=nNew+C1 #original
	maxi:int=(int(n**0.5) - C3) // C2 + C1 # adapted from wikipedia
	maxi:int=C2+int(nNew**0.5) # dakra 
	skipped:int=C0
	for i in range(C1, maxi):
		if not marked[i]: continue
		  # skipped+=C1
		  # continue
		start:int= i+i+C2*i*i # starts with j=i        
		step:int= C1+i+i # (1+2*i) is (i+(j+1)+2i(j+1)) and (i+j+2ij)
		lenslice:int  = (stop - start + step - 1 ) // step # instead of a while loop, with test, mark, and increment. numpy slice assignment allows a scaler on the RHS.
		marked[start:stop:step]= [F]* lenslice # F       
	r:list[int]= [C2]+[i+i+C1 for i in range(C1,nNew + C1) if marked[i]] # 1 is not prime
	#r.insert(0,2)
	#printalist(["skipped",skipped])
	# print(r[:20:])
	return r


#@njit
def SieveSundaramwj(n:int)->list[int]:
	#   BROKEN !!!!
	"""The sieve of Sundaram is a simple deterministic algorithm for finding all the prime numbers up to a specified integer."""
	# based on https://en.wikipedia.org/wiki/Sieve_of_Sundaram 
	
	  
	C0:int=0
	C1:int=1
	C2:int=2    
	C3:int=3
	C4:int=4
	C5:int=5
	C6:int=6
	C7:int=7
	C11:int=11
	C12:int=12
	T:bool=True
	F:bool=False
	  
	k:int = (n - C3) // C2 + C1

	integers_list:list[bool] = [T for i in range(k)]

	#ops:int = 0

	for i in range((int(float(n)**0.5) - C3) // C2 + C1):
#        if integers_list[i]: # adding this condition turns it into a SoE!
			p:int = C2 * i + C3
			s:int = (p * p - C3) // C2 # compute cull start

			"""for j in range(s, k, p):
				integers_list[j] = F
				#ops += 1
			"""
			#lenslice:int  = (stop - start + step - 1 ) // step
			 
			integers_list[s:k:p]=[F]*((k - s + p - 1 ) // p )
	#print("Total operations:  ", ops, ";", sep='')

	"""count = 1
	for i in range(k):
		if integers_list[i]:
			#count += 1
	"""
	r:list[int]=[i for i in range(k) if integers_list[i] ]
	#print("Found ", count, " primes to ", n, ".", sep='')
	return r

def kprod(alist: list[i4]) ->int:
	k:i4=1
	for m in alist:
		k*=m
	return k

def modInverse(aa:int, mm:int) -> int: # Returns modulo inverse of aa with # respect to mm 
# This code is contributed by Nikita Tiwari, modified by David Kra. 
# Contribution found at https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/?ref=lbp 
# Iterative Python 3 program to find modul0 inverse using extended Euclid algorithm
# Assumption: a and m are # coprimes, i.e., gcd(a, m) = 1
	#if (mm == 1): return 0
	m:i4=mm
	a:i4=aa%m # newr
	if a==1: return 1
	if a==m-1: return a # (m-1) is is its own inverse.
	m0:i4= m # r
	y:i4= 0  # t
	x:i4= 1  # newt
	
	while (a > 1):		
		q:i4= a // m
		t:i4= m
		# m is remainder now, process same as Euclid's algo
		m:i4= a % m
		a:i4= t
		t:i4= y
		# Update x and y
		y:i4= x - q * y
		x:i4= t
	if (x < 0): x:i4= x + m0 	# Make x positive
	return x
	
def SieveKra11(upToNumm: int=100000000, pnprimorial:i4=7): # primorial with doinverses and smartharvest
	# uses kprod, modInverse
	#C0,C1,C2,C3,C4,C5,C6=0,1,2,3,4,5,6
	C0:i4=0
	C1:i4=1
	C2:i4=2
	C3:i4=3
	C4:i4=4
	C5:i4=5
	C6:i4=6
	T:bool=True
	F:bool=False
	
	nprimorial:i4=19 if pnprimorial>19 else pnprimorial
	baseprimes:list[i4]=  [i for i in [2, 3,  5,   7,   11,    13,     17,      19,        23] if i <=nprimorial]
	#primorials:list[i4]]=            (2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870) # [math.prod(baseprimes[0:i+1]) for i in range(len(baseprimes))]
	
	primorial:i4=int(kprod(baseprimes))
	
	upToNum:i4=max(upToNumm,100)
	upToNum:i4= primorial* ( (upToNum+primorial-1)//primorial )  
	
	rows:i4=(upToNum+1)//primorial
	 
	#modprimorialdo:list[i4]=[i for i in range(primorial) if kprod([i%bp for bp in baseprimes] ) ]
	modprimorialdo:list[i4]=list(range(C1,primorial,C2))  #  about 3x faster  
	for i in baseprimes[C1:]: # skip 2
	   modprimorialdo:list[i4]=[j for j in modprimorialdo if j%i]
	
	modprimorialdos:set[i4]={*modprimorialdo}
	lmodprimorialdo:i4=len(modprimorialdo)
	# calculate the modulo inverse for each do column as a long sparse list
	doinverses:list[i4]=[0]*primorial
	doinverses[C1]=C1 #c the first is always 1.
	doinverses[primorial-C1]=primorial-C1 # The last is always itself.

	for c in modprimorialdo:
		if not doinverses[c]:
			doinverses[c]=ci=modInverse(c, primorial) # the inverse of one is the inverse of the other.
			doinverses[ci]=c

	modelrow:list[bool]=[T if i in modprimorialdos else F for i in range(primorial)]
	
	#print(["nprimorial=",nprimorial," primorial=",primorial," UpToNum=",upToNum," rows=",rows])  
	
	possibleprimes:list[i4]=modelrow*rows # populate all the rows with the model row.
	
	# fix up the beginning nonprimes and primes
	possibleprimes[C0]=F
	possibleprimes[C1]=F    
	for i in baseprimes:  possibleprimes[i]=T 
		
	lenpossibleprimes:i4=len(possibleprimes)
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial)
	# maxcast:i4=int(sqrt(upToNum))+2 # All factors greater than a number's square root have a matching factor less than the square root. There is no need to cast out these larger values.
	
	ln :i4=lenpossibleprimes
	art :i4=C0 #St art; st op; st ep
	op :i4=ln
	ep : int = C2
	lslice :i4= (op-art+ep-C1 ) // ep 
	smax :i4= int(((ln**0.5)//1) +C2 )
	
	alist:i4=possibleprimes  

	for i in range(baseprimes[-1]+C2,smax,C2): # start after the last baseprime
	   if not alist[i]: continue # not prime

	   isq:i4=i*i
	   art:i4=isq  # always 1 mod 6, but not always 1 mod higher primorials. Could be 19 mod 30
	   isqcol:i4=isq%primorial
	   ep:i4=i*primorial #aka stepdown
	   ix2:i4=i+i
	   inversei:i4=doinverses[i%primorial]

	   for d in modprimorialdo: # will consider only do columns 
		   #while not art%primorial in modprimorialdos: 
		   #   art+=ix2 # skip even columns
			
		   # What is x, such that d=(x*i+isq)%primorial ? (d = destination)
		   #         x = (d-isq)*inverse(i), == (d-isqcol)*inverse(i)
		   toshifttocolumnd:i4=i * (( (d-isqcol)*inversei ) % primorial) # ==0 for isqcol.
		   art:i4=isq+toshifttocolumnd # always relative to isq
			 # if start%primorial != d: print([i, isq, d, start%primorial])
		   lslice:i4=(op-art+ep-C1)//ep
		   alist[art:op:ep]=  [F]*lslice # C0 # 
		   #art+=ix2 # skip even columns
	
	#smartharvest 
	r:list[i4]=baseprimes+[ipj for i in range(0,beginlastrow,primorial) for j in modprimorialdo if possibleprimes[ipj:=(i+j)] ]
	
	if r[-1] <= upToNumm: return r
	# Truncate at upToNumm
	lenr:i4=len(r) 
	for i in range(lenr-1,1,-1):
		if r[i]<=upToNumm: break 
	if i<lenr: del r[i+1:]
	return r


def SieveKra12bw(upToNumm:i4=100000000, pnprimorial:i4=7)->list[i4]: # b== boolean entries in possible primes. w== while loop
	# primorial with dense list, 
	# with dense meaning that the entries for uninteresting do-not-do columns are absent.
	# values in the possibleprimes list are True if prime, otherwise False
	#C0,C1,C2,C3,C4,C5,C6=0,1,2,3,4,5,6
	C0:i4=0
	C1:i4=1
	C2:i4=2
	C3:i4=3
	C4:i4=4
	C5:i4=5
	C6:i4=6
	C7:i4=7
	T:bool=True
	F:bool=False
	m1G:i4=-int(10**9) # Out of Range
	oor:i4=m1G  # Out of Range # Must be negative
	#if pnprimorial<C3: pnprimorial=C3 # could test with 2
	nprimorial:i4=C3 if pnprimorial<3 else pnprimorial
	nprimorial:i4=C7 if pnprimorial>13 else pnprimorial
	baseprimes:list[i4]= [i for i in [2, 3,  5,   7,   11,	13,	 17,	  19,		23] if i <=nprimorial]
	#primorials:list[i4]=			(2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870) # [math.prod(baseprimes[0:i+1]) for i in range(len(baseprimes))]
	
	primorial:i4=int(kprod(baseprimes))
	
	upToNum:i4=max(upToNumm,100)
	upToNum:i4= primorial* ( (upToNum+primorial-1)//primorial )  
	
	rows:i4=(upToNum+1)//primorial
	
	# modpriorialdo list creation
	#	modprimorialdo:list[i4]=[i for i in range(primorial) if kprod([i%bp for bp in baseprimes] ) ]
	modprimorialdo:list[i4]=list(range(C1,primorial,C2))
	for i in baseprimes[C1:]:
		modprimorialdo:list[i4]=[j for j in modprimorialdo if j%i]
	# modprimorialdos:set[i4]={*modprimorialdo}
	lmodprimorialdo:i4=len(modprimorialdo)

	# figure out a sparse subscript lookup for which columns of the 2D sparse array hold the DO elements of the full list. #
	#	e.g for [1,5] -> [-1000000000, 0,  -1000000000, -1000000000, -1000000000, 1] where 1000000000 indicates Out Of Range
	# purpose: docolinsparse[d] tells you which column handles numbers which are (d mod primorial)
	# Uninteresting entries have an out of range value that will fail any attempt to access an item in the array by using that value as a subscript
	j:i4=C0
	docolinsparse:list[i4]=[oor]*primorial
	for j in range(lmodprimorialdo):
		docolinsparse[ modprimorialdo[j] ]=j 
  
	# end calculate docolinsparse
		
	# Dense list processing. Items will be the True if prime, otherwise be changed to False
	#create the list
	possibleprimes:list[i4]=[T]* (rows*lmodprimorialdo)
	lpossibleprimes:i4=len(possibleprimes)
	possibleprimes[C0]=F # 1 is not Prime 
	
	# print("Beginning and end of possibleprimes:\n",possibleprimes[:32],"\n",possibleprimes[-32:])
			
	lenpossibleprimes:i4=len(possibleprimes) # length of the dense list
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial)+1 ## in the dense
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial) ## in the sparse 
		
	ln :i4=lenpossibleprimes # length of the dense list
	artv :i4=C0 #STart; st op; st ep #start value
	artp:i4= C0  # STart position
	op :i4=ln   # STop
	ep : int = C2 #STep
	lslice :i4= (op-artp+ep-C1 ) // ep 
	smax :i4= int(upToNum**0.5) +C2  # All factors greater than a number's square root have a matching factor less than the square root. There is no need to cast out these larger values.

	# work to the end of the row containing smax.
	rowsmax:i4=(smax+primorial-C1)//primorial
	smax:i4=rowsmax*primorial # lmodprimorialdo-1
	possmax:i4=rowsmax*lmodprimorialdo # position in dense array of smax
	
	alist:list[i4]=possibleprimes  
	
	# just to get them typed before the loop
	ddocol:i4=C0
	ddocol=C0
	artp:i4=C0
	artp=C0
	fdorowcol:list[i4]=[C0,C0]	
	isq:i4=C0
	ix2:i4=C0
	artv:i4=C0
	i:i4=C0
	j:i4=C0
	ep:i4=C0
			
	for j in range(possmax): 
		if not (alist[j]): continue # not prime
		dr,dc=divmod(j,lmodprimorialdo) # dense row and column
		i=(dr*primorial) + modprimorialdo[dc] # the value
		
		isq=i*i
		ix2=i+i
		artv=isq-ix2 # starting value, will have ix2 added back in the while loop below. Not the starting position.
		
		ep=i*lmodprimorialdo #aka vertical stepdown
		
		#  inversei:i4=doinverses[i%primorial]
		
		for d in modprimorialdo: # will consider only do columns 
			while (
			
					(ddocol:= int(docolinsparse[   
												(fdorowcol:=divmod(
																	(artv:=artv+ix2)
																					,primorial) 
																	) [1]
												])
					)<0
				 ) : pass

			artp:i4=int(ddocol+fdorowcol[C0]*lmodprimorialdo)
			#artp:i4= t2 
			lslice:i4=(op-artp+ep-C1)//ep
			#print("artv=",artv," artp=",artp," fdorowcol=",fdorowcol," ddocol=",ddocol)
			alist[artp:op:ep]=  [F]*lslice

	# smart harvest for entries in possible primes holding the value of the prime or 0 if not prime.
	#	 
	#r:list[i4]=baseprimes+[i for i in possibleprimes if i ]
	
	# harvest dense list with boolean values
	r=baseprimes+[0]*lenpossibleprimes
	pos=len(baseprimes)-1
	for i in range(lenpossibleprimes):
		if not alist[i]: continue
		pos+=1
		dr,dc=divmod(i,lmodprimorialdo) # dense row and dense column
		r[pos]=(dr*primorial)+modprimorialdo[dc]
	r=r[0:pos+1]	
	
	#print(possibleprimes[:20],possibleprimes[-20:])
	if r[-1] <= upToNumm: return r
	
	# to truncate at upToNumm
	lenr:i4=len(r) 
	for i in range(lenr-1,1,-1):
		if r[i]<=upToNumm: break 
	if i<lenr: del r[i+1:]
	return r



def SieveKra12b(upToNumm:i4=100000000, pnprimorial:i4=7)->list[i4]: 
	# b== boolean entries in possible primes. m = multiplicative inverses
	# primorial with dense list, 
	# with dense meaning that the entries for uninteresting do-not-do columns are absent.
	# values in the possibleprimes list are True if prime, otherwise False
	#C0,C1,C2,C3,C4,C5,C6=0,1,2,3,4,5,6
	C0:i4=0
	C1:i4=1
	C2:i4=2
	C3:i4=3
	C4:i4=4
	C5:i4=5
	C6:i4=6
	C7:i4=7
	T:bool=True
	F:bool=False
	m1G:i4=-int(10**10) # Out of Range
	oor:i4=-m1G  # Out of Range # Must be negative
	#if pnprimorial<C3: pnprimorial=C3 # could test with 2
	nprimorial:i4=C3 if pnprimorial<3 else pnprimorial
	nprimorial:i4=C7 if pnprimorial>13 else pnprimorial
	baseprimes:list[i4]= [i for i in [2, 3,  5,   7,   11,	13,	 17,	  19,		23] if i <=nprimorial]
	#primorials:list[i4]=			(2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870) # [math.prod(baseprimes[0:i+1]) for i in range(len(baseprimes))]
	
	primorial:i4=int(kprod(baseprimes))
	
	if hasattr(upToNumm,"__iter__"): upToNumm=upToNumm[0]
	# print("upToNumm= ", upToNumm)
	upToNum:i4=max(upToNumm,100)
	upToNum:i4= primorial* ( (upToNum+primorial-1)//primorial )  
	
	rows:i4=(upToNum+1)//primorial
	
	# modpriorialdo list creation
	#	modprimorialdo:list[i4]=[i for i in range(primorial) if kprod([i%bp for bp in baseprimes] ) ]
	modprimorialdo:list[i4]=list(range(C1,primorial,C2))
	for i in baseprimes[C1:]:
		modprimorialdo:list[i4]=[j for j in modprimorialdo if j%i]
	# modprimorialdos:set[i4]={*modprimorialdo} # unused
	lmodprimorialdo:i4=len(modprimorialdo)

	# figure out a sparse subscript lookup for which columns of the 2D sparse array hold the DO elements of the full list. #
	#	e.g for [1,5] -> [-1000000000, 0,  -1000000000, -1000000000, -1000000000, 1] where 1000000000 indicates Out Of Range
	# purpose: docolinsparse[d] tells you which column handles numbers which are (d mod primorial)
	# Uninteresting entries have an out of range value that will fail any attempt to access an item in the array by using that value as a subscript
	j:i4=C0
	docolinsparse:list[i4]=[oor]*primorial
	for j in range(lmodprimorialdo):
		docolinsparse[ modprimorialdo[j] ]=j   
	# end calculate docolinsparse
	
	# calculate the modulo inverse for each do column as a long sparse list # but not as a dense list
	doinverses:list[i4]=[C0]*primorial
	doinverses[C1]=C1 #c the first is always 1.
	doinverses[primorial-C1]=primorial-C1 # The last is always itself.
	#densdoinverses:list[i4]=[C0]*lmodprimorialdo # currently unused
	for c in modprimorialdo:
		if not doinverses[c]:
			doinverses[c]=ci=modInverse(c, primorial) # the inverse of one is the inverse of the other.
			doinverses[ci]=c
	
	#for i in range(lmodprimorialdo):
		#densdoinverses[i]=doinverses[c]=modInverse(c:=modprimorialdo[i], primorial) #nondo columns in have 0 doinverses
		
	# Dense list processing. Items will be the True if prime, otherwise be changed to False
	#create the list
	possibleprimes:list[i4]=[T]* (rows*lmodprimorialdo)
	lpossibleprimes:i4=len(possibleprimes)
	possibleprimes[C0]=F # 1 is not Prime 
	
	# print("Beginning and end of possibleprimes:\n",possibleprimes[:32],"\n",possibleprimes[-32:])
			
	lenpossibleprimes:i4=len(possibleprimes) # length of the dense list
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial)+1 ## in the dense
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial) ## in the sparse 
	
		
	ln :i4=lenpossibleprimes # length of the dense list
	artv :i4=C0 #STart; st op; st ep #start value
	artp:i4= C0  # STart position
	op :i4=ln   # STop
	ep : int = C2 #STep
	lslice :i4= (op-artp+ep-C1 ) // ep 
	smax :i4= int(upToNum**0.5) +C1  # All factors greater than a number's square root have a matching factor less than the square root. There is no need to cast out these larger values.

	# work to the end of the row containing smax.
	rowsmax:i4=(smax+primorial-C1)//primorial
	smax:i4=rowsmax*primorial # lmodprimorialdo-1
	possmax:i4=rowsmax*lmodprimorialdo # position in dense array of smax
	
	alist:list[i4]=possibleprimes  
	
	# just to get them typed before the loop
	ddocol:i4=C0
	ddocol=C0
	artp:i4=C0
	artp=C0
	fdorowcol:list[i4]=[C0,C0]	
	isq:i4=C0
	ix2:i4=C0
	artv:i4=C0
	i:i4=C0
	j:i4=C0
	ep:i4=C0
	#startvalues:list[int]=[C0]*lmodprimorialdo
	startpositions:list[int]=[C0]*lmodprimorialdo

	dpos=-1
	for rowval in range(0,smax,primorial):
		for dcol in range(lmodprimorialdo):
			dpos+=1
			if not alist[dpos]: continue
			
			i=rowval+modprimorialdo[dcol]
			isq=i*i
			ep=i*lmodprimorialdo #aka vertical stepdown
			isqmodprimorial:int= isq%primorial  # is (p^2 modulo n#)
			mmi:int=doinverses[i%primorial] # yes, the Modulo Multiplicative Inverse of the prime
			
			for k in range(lmodprimorialdo):
				dc=modprimorialdo[k]
				m= (  mmi * (( dc - isqmodprimorial ) %primorial)) %primorial # is the interior %primorial necessary????
				dv=isq+m*i #startvalues[k]=(dv:=isq+m*i)
				#row,fullcol=divmod(dv,primorial)
				row=(dv//primorial)
				startpositions[k]= row*lmodprimorialdo+k # should be the same as docolinsparse[fullcol]
					
			# The following could run in parallel on CPU or GPU
			for artp in startpositions:
				alist[artp:op:ep]= [F]* ((op-artp+ep-C1 ) // ep )	
		
	# harvest dense list with boolean values
	#r=baseprimes+[0]*lenpossibleprimes
	r=baseprimes+[0]*int(1.25*upToNum/math.log(upToNum))
	rpos=len(baseprimes)-1
	dpos=-1
	for rowval in range(0,upToNum,primorial):
		for dcol in range(lmodprimorialdo):
			dpos+=1
			if not alist[dpos]: continue
			rpos+=1
			r[rpos]=rowval+modprimorialdo[dcol]
	# truncate excess elements
	r=r[0:rpos+1] 

	if r[-1] <= upToNumm: return r		

	# truncate at upToNumm
	lenr:i4=len(r) 
	for i in range(lenr-1,1,-1):
		if r[i]<=upToNumm: break 
	r=r[0:i+1]
	return r



def SieveKra12(upToNumm:i4=100000000, pnprimorial:i4=7)->list[i4]: 
	# was SieveKra12qh
	# qh == Quick harvest. Start harvesting while sieving.
	# b== boolean entries in possible primes. m = multiplicative inverses
	# primorial with dense list, 
	# with "dense" meaning that the entries for uninteresting do-not-do columns are absent.
	# values in the possibleprimes list are True if prime, otherwise False
	#C0,C1,C2,C3,C4,C5,C6=0,1,2,3,4,5,6
	C0:i4=0
	C1:i4=1
	C2:i4=2
	C3:i4=3
	C4:i4=4
	C5:i4=5
	C6:i4=6
	C7:i4=7
	T:bool=True
	F:bool=False
	m1G:i4=-int(10**10) # Out of Range
	oor:i4=-m1G  # Out of Range # Must be negative
	#if pnprimorial<C3: pnprimorial=C3 # could test with 2
	nprimorial:i4=C3 if pnprimorial<3 else pnprimorial
	nprimorial:i4=C7 if pnprimorial>13 else pnprimorial
	baseprimes:list[i4]= [i for i in [2, 3,  5,   7,   11,	13,	 17,	  19,		23] if i <=nprimorial]
	#primorials:list[i4]=			(2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870) # [math.prod(baseprimes[0:i+1]) for i in range(len(baseprimes))]
	
	primorial:i4=int(kprod(baseprimes))
	
	if hasattr(upToNumm,"__iter__"): upToNumm=upToNumm[0]
	# print("upToNumm= ", upToNumm)
	upToNum:i4=max(upToNumm,100)
	upToNum:i4= primorial* ( (upToNum+primorial-1)//primorial )  
	
	rows:i4=(upToNum+1)//primorial
	
	# modpriorialdo list creation
	#	modprimorialdo:list[i4]=[i for i in range(primorial) if kprod([i%bp for bp in baseprimes] ) ]
	modprimorialdo:list[i4]=list(range(C1,primorial,C2))
	for i in baseprimes[C1:]:
		modprimorialdo:list[i4]=[j for j in modprimorialdo if j%i]
	# modprimorialdos:set[i4]={*modprimorialdo} # unused
	lmodprimorialdo:i4=len(modprimorialdo)

	# figure out a sparse subscript lookup for which columns of the 2D sparse array hold the DO elements of the full list. #
	#	e.g for [1,5] -> [-1000000000, 0,  -1000000000, -1000000000, -1000000000, 1] where 1000000000 indicates Out Of Range
	# purpose: docolinsparse[d] tells you which column handles numbers which are (d mod primorial)
	# Uninteresting entries have an out of range value that will fail any attempt to access an item in the array by using that value as a subscript
	j:i4=C0
	docolinsparse:list[i4]=[oor]*primorial
	for j in range(lmodprimorialdo):
		docolinsparse[ modprimorialdo[j] ]=j   
	# end calculate docolinsparse
	
	# calculate the modulo inverse for each do column as a long sparse list # but not as a dense list
	doinverses:list[i4]=[C0]*primorial
	doinverses[C1]=C1 #c the first is always 1.
	doinverses[primorial-C1]=primorial-C1 # The last is always itself.
	#densdoinverses:list[i4]=[C0]*lmodprimorialdo # currently unused
	for c in modprimorialdo:
		if not doinverses[c]:
			doinverses[c]=ci=modInverse(c, primorial) # the inverse of one is the inverse of the other.
			doinverses[ci]=c
	
	#for i in range(lmodprimorialdo):
		#densdoinverses[i]=doinverses[c]=modInverse(c:=modprimorialdo[i], primorial) #nondo columns in have 0 doinverses
		
	# Dense list processing. Items will be the True if prime, otherwise be changed to False
	#create the list
	possibleprimes:list[i4]=[T]* (rows*lmodprimorialdo)
	lpossibleprimes:i4=len(possibleprimes)
	possibleprimes[C0]=F # 1 is not Prime 
	
	# print("Beginning and end of possibleprimes:\n",possibleprimes[:32],"\n",possibleprimes[-32:])
			
	lenpossibleprimes:i4=len(possibleprimes) # length of the dense list
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial)+1 ## in the dense
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial) ## in the sparse 
	
		
	ln :i4=lenpossibleprimes # length of the dense list
	artv :i4=C0 #STart; st op; st ep #start value
	artp:i4= C0  # STart position
	op :i4=ln   # STop
	ep : int = C2 #STep
	lslice :i4= (op-artp+ep-C1 ) // ep 
	smax :i4= int(upToNum**0.5) +C1  # All factors greater than a number's square root have a matching factor less than the square root. There is no need to cast out these larger values.

	# work to the end of the row containing smax.
	rowsmax:i4=(smax+primorial-C1)//primorial
	smax:i4=rowsmax*primorial # lmodprimorialdo-1
	possmax:i4=rowsmax*lmodprimorialdo # position in dense array of smax
	
	alist:list[i4]=possibleprimes  
	
	# just to get them typed before the loop
	ddocol:i4=C0
	ddocol=C0
	artp:i4=C0
	artp=C0
	fdorowcol:list[i4]=[C0,C0]	
	isq:i4=C0
	ix2:i4=C0
	artv:i4=C0
	i:i4=C0
	j:i4=C0
	ep:i4=C0
	#startvalues:list[int]=[C0]*lmodprimorialdo
	startpositions:list[int]=[C0]*lmodprimorialdo
	
	# harvest dense list with boolean values
	# initialize
	#r=baseprimes+[0]*lenpossibleprimes
	r=baseprimes+[0]*int(1.25*upToNum/math.log(upToNum))
	rpos=len(baseprimes)-1 # position in the results

	dpos=-1
	for rowvall in range(0,smax,primorial):
		rowval=rowvall
		for dcol in range(lmodprimorialdo):
			dpos+=1
			if not alist[dpos]: continue # not a prime
			
			i=rowval+modprimorialdo[dcol]
			# harvest this prime now
			rpos+=1
			r[rpos]=i
			#
			isq=i*i
			ep=i*lmodprimorialdo #aka vertical stepdown
			isqmodprimorial:int= isq%primorial  # is (p^2 modulo n#)
			mmi:int=doinverses[i%primorial] # yes, the Modulo Multiplicative Inverse of the prime
			
			for k in range(lmodprimorialdo):
				dc=modprimorialdo[k]
				m= (  mmi * (( dc - isqmodprimorial ) %primorial)) %primorial # is the interior %primorial necessary????
				dv=isq+m*i #startvalues[k]=(dv:=isq+m*i)
				#row,fullcol=divmod(dv,primorial)
				row=(dv//primorial)
				startpositions[k]= row*lmodprimorialdo+k # should be the same as docolinsparse[fullcol]
					
			# The following could run in parallel on CPU or GPU
			for artp in startpositions:
				alist[artp:op:ep]= [F]* ((op-artp+ep-C1 ) // ep )	
		
	# harvest dense list with boolean values
	# previously initiated:
	#     r=baseprimes+[0]*lenpossibleprimes
	#     rpos=len(baseprimes)-1
	# dpos=-1 # continue from where left off
	for rowval in range(rowval+primorial,upToNum,primorial):
		for dcol in range(lmodprimorialdo):
			dpos+=1
			if not alist[dpos]: continue
			rpos+=1
			r[rpos]=rowval+modprimorialdo[dcol]
	# truncate excess elements
	r=r[0:rpos+1] 

	if r[-1] <= upToNumm: return r		

	# truncate at upToNumm
	lenr:i4=len(r) 
	for i in range(lenr-1,1,-1):
		if r[i]<=upToNumm: break 
	r=r[0:i+1]
	return r


def SieveKra12np(upToNumm:i4=100000000, pnprimorial:i4=7)->list[i4]: 
	# b== boolean entries in possible primes. m = multiplicative inverses
	# primorial with dense list, 
	# with dense meaning that the entries for uninteresting do-not-do columns are absent.
	# values in the possibleprimes list are True if prime, otherwise False
	#C0,C1,C2,C3,C4,C5,C6=0,1,2,3,4,5,6
	C0:i4=0
	C1:i4=1
	C2:i4=2
	C3:i4=3
	C4:i4=4
	C5:i4=5
	C6:i4=6
	C7:i4=7
	T=np.bool(True)
	F=np.bool(False)
	m1G:i4=-int(10**10) # Out of Range
	oor:i4=-m1G  # Out of Range # Must be negative
	#if pnprimorial<C3: pnprimorial=C3 # could test with 2
	nprimorial:i4=C3 if pnprimorial<3 else pnprimorial
	nprimorial:i4=C7 if pnprimorial>13 else pnprimorial
	baseprimes:list[i4]= [i for i in [2, 3,  5,   7,   11,	13,	 17,	  19,		23] if i <=nprimorial]
	#primorials:list[i4]=			(2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870) # [math.prod(baseprimes[0:i+1]) for i in range(len(baseprimes))]
	
	# alist=np.ones(ln,dtype='?') 
	# C1=np.float128(1)
	# b=np.bool(True)
	# l=np.uint32(1)
	# ll=np.uint64(1)
		
	primorial:i4=int(kprod(baseprimes))
	
	upToNum:i4=max(upToNumm,100)
	upToNum:i4= primorial* ( (upToNum+primorial-1)//primorial )  
	
	rows:i4=(upToNum+1)//primorial
	
	# modpriorialdo list creation
	#	modprimorialdo:list[i4]=[i for i in range(primorial) if kprod([i%bp for bp in baseprimes] ) ]
	modprimorialdo:list[i4]=list(range(C1,primorial,C2))
	for i in baseprimes[C1:]:
		modprimorialdo:list[i4]=[j for j in modprimorialdo if j%i]
	# modprimorialdos:set[i4]={*modprimorialdo} # unused
	lmodprimorialdo:i4=len(modprimorialdo)

	# figure out a sparse subscript lookup for which columns of the 2D sparse array hold the DO elements of the full list. #
	#	e.g for [1,5] -> [-1000000000, 0,  -1000000000, -1000000000, -1000000000, 1] where 1000000000 indicates Out Of Range
	# purpose: docolinsparse[d] tells you which column handles numbers which are (d mod primorial)
	# Uninteresting entries have an out of range value that will fail any attempt to access an item in the array by using that value as a subscript
	j:i4=C0
	docolinsparse:list[i4]=[oor]*primorial
	for j in range(lmodprimorialdo):
		docolinsparse[ modprimorialdo[j] ]=j   
	# end calculate docolinsparse
	
	# calculate the modulo inverse for each do column as a long sparse list # but not as a dense list
	doinverses:list[i4]=[C0]*primorial
	doinverses[C1]=C1 #c the first is always 1.
	doinverses[primorial-C1]=primorial-C1 # The last is always itself.
	#densdoinverses:list[i4]=[C0]*lmodprimorialdo # currently unused
	for c in modprimorialdo:
		if not doinverses[c]:
			doinverses[c]=ci=modInverse(c, primorial) # the inverse of one is the inverse of the other.
			doinverses[ci]=c
	
	#for i in range(lmodprimorialdo):
		#densdoinverses[i]=doinverses[c]=modInverse(c:=modprimorialdo[i], primorial) #nondo columns in have 0 doinverses
		
	# Dense list processing. Items will be the True if prime, otherwise be changed to False
	#create the list
	#possibleprimes:list[i4]=[T]* (rows*lmodprimorialdo)
	possibleprimes=np.ones(rows*lmodprimorialdo,dtype=np.bool)
	lpossibleprimes:i4=len(possibleprimes)
	possibleprimes[C0]=F # 1 is not Prime 
	
	# print("Beginning and end of possibleprimes:\n",possibleprimes[:32],"\n",possibleprimes[-32:])
			
	lenpossibleprimes:i4=len(possibleprimes) # length of the dense list
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial)+1 ## in the dense
	beginlastrow:i4=primorial*(lenpossibleprimes//primorial) ## in the sparse 
	
		
	ln :i4=lenpossibleprimes # length of the dense list
	artv :i4=C0 #STart; st op; st ep #start value
	artp:i4= C0  # STart position
	op :i4=ln   # STop
	ep : int = C2 #STep
	lslice :i4= (op-artp+ep-C1 ) // ep 
	smax :i4= int(upToNum**0.5) +C1  # All factors greater than a number's square root have a matching factor less than the square root. There is no need to cast out these larger values.

	# work to the end of the row containing smax.
	rowsmax:i4=(smax+primorial-C1)//primorial
	smax:i4=rowsmax*primorial # lmodprimorialdo-1
	possmax:i4=rowsmax*lmodprimorialdo # position in dense array of smax
	
	alist:list[i4]=possibleprimes  
	
	# just to get them typed before the loop
	ddocol:i4=C0
	ddocol=C0
	artp:i4=C0
	artp=C0
	fdorowcol:list[i4]=[C0,C0]	
	isq:i4=C0
	ix2:i4=C0
	artv:i4=C0
	i:i4=C0
	j:i4=C0
	ep:i4=C0
	#startvalues:list[int]=[C0]*lmodprimorialdo
	startpositions:list[int]=[C0]*lmodprimorialdo
	dpos=-1


	for rowval in range(0,smax,primorial):
		for dcol in range(lmodprimorialdo):
			dpos+=1
			if not possibleprimes[dpos]: continue
			
			i=rowval+modprimorialdo[dcol]
			isq=i*i
			ep=i*lmodprimorialdo #aka vertical stepdown
			isqmodprimorial:int= isq%primorial  # is (p^2 modulo n#)
			mmi=doinverses[i%primorial] # yes, the Modulo Multiplicative Inverse of the prime
			
			for k in range(lmodprimorialdo):
				dc=modprimorialdo[k]
				m= (  mmi * (( dc - isqmodprimorial ) %primorial)) %primorial # is the interior %primorial necessary????
				dv=isq+m*i #startvalues[k]=(dv:=isq+m*i)
				#row,fullcol=divmod(dv,primorial)
				row=(dv//primorial)
				startpositions[k]= row*lmodprimorialdo+k # should be the same as docolinsparse[fullcol]
					
			# The following could run in parallel on CPU or GPU
			for artp in startpositions:
				possibleprimes[artp:op:ep]= F # [F] * ((op-artp+ep-C1 ) // ep )	 # slower with slide
		
	# harvest dense list with boolean values
	#r=baseprimes+[0]*lenpossibleprimes
	r=np.array(baseprimes+[0]*lenpossibleprimes,dtype=np.uint32)
	rpos=len(baseprimes)-1
	dpos=-1
	for rowval in range(0,upToNum,primorial):
		for dcol in range(lmodprimorialdo):
			dpos+=1
			if not possibleprimes[dpos]: continue
			rpos+=1
			r[rpos]=rowval+modprimorialdo[dcol]
	# truncate excess elements
	r=r[0:rpos+1] 

	if r[-1] <= upToNumm: return r		

	# truncate at upToNumm
	lenr:i4=len(r) 
	for i in range(lenr-1,1,-1):
		if r[i]<=upToNumm: break 
	r=r[0:i+1]
	return r


# ======================================

"""
def pritchard0(limit):
def pritchard1(limit:int):
def euler(n):
def SieveAtkinGH(nmax):
def SieveAtkinSO(limit):
def SieveAtkinG4G(limit):
def SieveKra11(upToNumm: int=100000000, pnprimorial:i4=7):
def sieveKra12i(upToNumm:i4=100000000, pnprimorial:i4=7)->list[i4]: # primorial with dense list,
def SieveSundaram(n):
def SieveAtkinGHj(nmax:int)->list[int]:
def SieveAtkinSOj(limit):
def SieveAtkinG4Gj(limit):
def SieveSundaramj(n):
#def SieveSundaramwj(n:int)->list[int]:

"""

def A_Main():
	pass

def _Main():
	pass
	
A= lambda:None # Holds attributes
_=lambda:None  # Holds attributes

sieves=[SieveEratosthenes7, SieveSundaramj, SieveEuler, SieveAtkinSOj,	SievePritchard   ]
sievesKra=[PrimeSieveKra, SieveKra12 ] #,
unsieves=[	SieveSundaramj,
		SieveEuler, 
		SieveAtkinSOj,		 
		#SieveAtkinG4G,, SieveAtkinGHj, SieveAtkinG4Gj, SieveAtkinGH, 
		SieveEratosthenes0,SieveEratosthenes1,SieveEratosthenes2, SieveEratosthenes3,
		SieveEratosthenes4,SieveEratosthenes5, SieveEratosthenes6, SieveEratosthenes7,
		SievePritchard        
		]


	

tin=1
#n= 100000000
#n=  10000000
#n=   1000000
n= 1000000000 # 10^9
n=  100000000 # 10^8
n= 100059937 # last prime up to 100059960
n=  100059960 # 196 * 510,510, where 510,510 is 17#.
M=1000000
r10=10.0**0.5 # = 3.1622776601683795
A.rates=[]
A.results=[]
A.lens=[]
ns=[M, int(r10*M), 10*M, int(r10*10*M), 100*M, int(r10*100*M), 1000*M, int(2*1000*M)]
# printboth(ns)
#ns=int(r10*1000*M)
#ns=[2000*M]

ns=[10000, 100000, 1000000, 10000000, 100000000, 100059960, 1000000000]# ,  1300000000, 2000000000,  ]
ns = [100059960]

for n in ns:
	for trials in range(1):
		printnow()

		for s in sievesKra:
			for primorial in [7,11]:
				gc.collect()
				t=timeit('A.result=s(n,primorial)',number=tin,globals=globals())
				count=len(A.result)
				rate= float(n) / (1000000.0*t)
				print(f"MIPS=\t{rate:6.3f}\t{s.__name__}@{primorial}\t{count} primes upto {n} {A.result[0:5]} , {A.result[-5:] } " );
				del A.result

		for s in sieves:
				gc.collect()
				t=timeit('A.result=s(n)',number=tin,globals=globals())
				count=len(A.result)
				rate= float(n) / (1000000.0*t)
				print(f"MIPS=\t{rate:6.3f}\t{s.__name__}\t{count}\t primes upto \t{n}\t {A.result[0:5]}\t{A.result[-5:] }" );
				del A.result

	#print(" ")
printnow()


quit()


import big_o
# https://pypi.org/project/big-O/ 
# https://github.com/pberkes/big_O
ALL_EXCEPT_CUBIC_QUADRATIC = [big_o.complexities.Constant, 
					big_o.complexities.Logarithmic, 
					big_o.complexities.Linear, 
					big_o.complexities.Linearithmic, 
					big_o.complexities.NLogLogN,
					big_o.complexities.NLogNLogLogN,
					#big_o.complexities.Quadratic, 
					big_o.complexities.Polynomial, 
					big_o.complexities.Exponential]
nsteps=100
# multiples of 30, 210, 2310, 30030, or 510510
art=2*510510 # 2x = 1021020 6 x= 3063060
op=100*art   #  # 3M is OK, 30M is OK at 100 steps

#sieves=[SieveKra12, SieveEratosthenes4, SieveSundaramj]
for s in sieves:
	#best, others = big_o.big_o(SieveKra12, n_, min_n=art, max_n=op, n_measures=nsteps)
	best, others = big_o.big_o(s, big_o.datagen.n_, classes=ALL_EXCEPT_CUBIC_QUADRATIC, min_n=art, max_n=op, n_measures=nsteps, verbose=True)
	printboth(f"\n{s.__name__}\tstart=\t{art}\tstop=\t{op}\tsteps={nsteps}\t{best}")
	for class_, residuals in others.items():
		printboth(f"residuals=\t{residuals:6.3e}\tclass=\t{class_}\t{s.__name__}")



quit()

# SieveAtkinGH , SieveSundaram, SieveAtkinSO, SieveAtkinG4G

# 
#print("jit compilations and functional test")
# slower print("SieveSundaramwj",n,len(primes:=SieveSundaramj(n)), primes[0:5], primes[-5:])
print("SieveSundaramj",n,len(primes:=SieveSundaramj(n)),  primes[0:5], primes[-5:])
#print("SieveAtkinGHj",n,len(primes:=SieveAtkinGHj(n)),   primes[0:5], primes[-5:])
#print("SieveAtkinSOj",n,len(primes:=SieveAtkinSOj(n)),   primes[0:5], primes[-5:])
print("SieveAtkinG4Gj",n,len(primes:=SieveAtkinG4Gj(n)),  primes[0:5], primes[-5:])

n=100059960 # 196  * 510,510, where 510,510 is 17#.
n=100000000
n=10210200  # 20*510510 # 20 * 17#
n=106696590 # =  11  * 19#
n=100059960 # = 196  * 17# = 196 * 510510
n=10000000
"""
print()
print("SieveAtkinGH",n,len(primesl:=SieveAtkinGH(n)), primesl[-10:])
#print("SieveSundaram",n,len(primesl:=SieveSundaram(n)), primesl[-10:])
#print("SieveAtkinSO",n,len(primesl:=SieveAtkinSO(n)), primesl[-10:])
print("SieveAtkinG4G",n,len(primesl:=SieveAtkinG4G(n)), primesl[-10:])
print()
print('SieveAtkinGH ', n, f"{timeit('SieveAtkinGH(n)',number=tin,globals=globals()):.3f}" );
print('SieveSundaram ', n, f"{timeit('SieveSundaram(n)',number=tin,globals=globals()):.3f}" );
print('SieveAtkinSO ', n, f"{timeit('SieveAtkinSO(n)',number=tin,globals=globals()):.3f}" );
print('SieveAtkinG4G ', n, f"{timeit('SieveAtkinG4G(n)',number=tin,globals=globals()):.3f}" );
"""
print()
# very slow print('SieveAtkinGHj ', n, f"{timeit('SieveAtkinGHj(n)',number=tin,globals=globals()):.5f}" );
print('SieveSundaramj ', n, f"{timeit('SieveSundaramj(n)',number=tin,globals=globals()):.5f}" );
# print('SieveSundaramwj ', n, f"{timeit('SieveSundaramwj(n)',number=tin,globals=globals()):.5f}" );
# print('SieveAtkinSOj ', n, f"{timeit('SieveAtkinSOj(n)',number=tin,globals=globals()):.3f}" );
print('SieveAtkinG4Gj ', n, f"{timeit('SieveAtkinG4Gj(n)',number=tin,globals=globals()):.5f}" );

