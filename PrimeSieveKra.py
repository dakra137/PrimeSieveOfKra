#!/bin/python3.12
# Copyright © 2025 David A. Kra      dakra137@gmail.com  https://www.linkedin.com/in/dakra/ 
# License:  Creative Commons Attribution-ShareAlike 4.0  https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt 
# One function, modInverse, is a derivative of a work © by Nikita Tiwari, modified by David A. Kra. 
#     The original is found at https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/	

from numba import int32, i4, int64, i8, b1
import math

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
       This sieve is faster than the sieves of Atkins, Eratosthenes, Euler, Pritchard, and Sundaram
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


def modInverse(aa:int, mm:int) -> int: # Returns modulo inverse of aa with respect to mm 
# This code is contributed by Nikita Tiwari, modified by David A. Kra. 
# Contribution found at https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/?ref=lbp 
# Iterative Python 3 program to find modul0 inverse using extended Euclid algorithm
# Assumptions: a and m are coprimes, i.e., gcd(a, m) = 1
	#          m not equals 1    # if (mm == 1): return 0
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
	

if __name__ == "__main__":
	p=7
	for n in [1000, 10000, 100000, 1000000, 10000000, 100000000, 100059960]:
		pk=PrimeSieveKra(n,p)
		print ( len(pk), n, p, 	pk[0:5],	pk[-5:] )
		
	
"""
Expected results:
168 1000 7 [2, 3, 5, 7, 11] [971, 977, 983, 991, 997]
1229 10000 7 [2, 3, 5, 7, 11] [9931, 9941, 9949, 9967, 9973]
9592 100000 7 [2, 3, 5, 7, 11] [99929, 99961, 99971, 99989, 99991]
78498 1000000 7 [2, 3, 5, 7, 11] [999953, 999959, 999961, 999979, 999983]
664579 10000000 7 [2, 3, 5, 7, 11] [9999937, 9999943, 9999971, 9999973, 9999991]
5761455 100000000 7 [2, 3, 5, 7, 11] [99999931, 99999941, 99999959, 99999971, 99999989]
5764697 100059960 7 [2, 3, 5, 7, 11] [100059853, 100059857, 100059863, 100059893, 100059937]
"""
