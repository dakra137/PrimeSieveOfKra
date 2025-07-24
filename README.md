# PrimeSieveOfKra
Code and Documentation for the Prime Number Sieve of Kra, including comparison to other Prime Number Sieves.

Code for other prime number sieves, and ModuloInverse adapted from public sources.

    Prime number sieve algorithm by David A. Kra 	
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
       This sieve is faster than the sieves of Atkins, Eratosthenes, Euler, Pritchard, and Sundaram for large upper limits.
       At 7 primorial, this sieve, compared these other sieves, uses only 23% of the memory to hold candidate numbers.
