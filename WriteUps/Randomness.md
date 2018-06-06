# Randomness

1. Python 

   ```python
   import random
   random.seed()

   # The below shows how the Mercene Twister generates two sequences of same numbers given the same state

   s = random.getstate()
   print(random.getrandbits(3),random.getrandbits(3),random.getrandbits(3))
   random.setstate(s)
   print(random.getrandbits(3),random.getrandbits(3),random.getrandbits(3))

   # randome.randrange(start,stop,step), only for integer
   print(random.randrange(3,10,2))

   # random.randint(a,b), a<=i<=b
   print(random.randint(-1,2))

   # random choices with 
   print(random.choices([10,45,23,49,87,100],weights=[0,0,1,1,1,0],k=4))

   ```

   ​

2. Mathematical Background

   1. Conditional Probability: $P(A|B) = {P(A \cup B) \over P(B)}$ , indicates the probability of event $A$ happening given that event $B$ happened. Since B has already happened , the sample space reduces to B, so the probability is over $P(B)$

   2. Bayes's Formula: $P(A|B) = {P(B|A)P(B) \over P(B)}$

   3. Random Variables: A random variable is actually a function that maps outcome of a random event (like coin toss) to a real value.

   4. Linearity of Expectation: Let R1and R2 be two discrete random variables on some probability space, then $E[R1 + R2] = E[R1] + E[R2] $

   5. **Expected Number of Trials until Success** If probability of success is p in every trial, then expected number of trials until success is 1/p. For example, consider 6 faced fair dice is thrown until a ‘5’ is seen as result of dice throw. The expected number of throws before seeing a 5 is 6. Note that 1/6 is probability of getting a 5 in every trial. So number of trials is 1/(1/6) = 6.

   6. Binomial Random Variables:  A specific type of **discrete** random variable that counts how often a particular event occurs in a fixed number of tries or trials.

      For a variable to be a binomial random variable, ALL of the following conditions must be met:

      1. There are a fixed number of trials (a fixed sample size).
      2. On each trial, the event of interest either occurs or does not.
      3. The probability of occurrence (or not) is the same on each trial.
      4. Trials are independent of one another.
      5. Probability: 	(Number of ways to achieve k success and n-k failure) * (probability for each way to achieve k success and n-k failure) = ![img](http://contribute.geeksforgeeks.org/wp-content/uploads/3.png)

3. Algorithm and Problems: 

   1. Problem Statement : Given an unsorted array A[] of n numbers and ε > 0, compute an element whose rank (position in sorted A[]) is in the range [(1 – ε)n/2, (1 + ε)n/2]. For ½ Approximate Median Algorithm ; is 1/2 => rank should be in the range [n/4, 3n/4]. Following steps represent an algorithm that is O((Log n) x (Log Log n)) time and produces incorrect result with probability less than or equal to 2/n^2.

      1. Randomly choose k elements from the array where k=c log n (c is some constant)
      2. Insert then into a set.
      3. Sort elements of the set.
      4. Return median of the set i.e. (k/2)th element from the set

   2. Classification: Randomized algorithms are classified in two categories.**Las Vegas:** These algorithms always produce correct or optimum result.Time complexity of these algorithms is based on a random value and time complexity is evaluated as expected value.  **Monte Carlo:** Produce correct or optimum result with some probability. These algorithms have deterministic running time and it is generally easier to find out worst case time complexity.

   3. Generate [1-7] from [1-5]with equal probability:  Given a function foo() that returns integers from 1 to 5 with equal probability, write a function that returns integers from 1 to 7 with equal probability using foo() only. Minimize the number of calls to foo() method. Also, use of any other library function is not allowed and no floating point arithmetic allowed: 

      If we somehow generate integers from 1 to a-multiple-of-7 (like 7, 14, 21, …) with equal probability, we can use modulo division by 7 followed by adding 1 to get the numbers from 1 to 7 with equal probability.

      We can generate from 1 to 21 with equal probability using the following expression.

      ```python
       #5*foo() + foo() -5 
      ```

      Let us see how above expression can be used.

      1. For each value of first foo(), there can be 5 possible combinations for values of second foo(). So, there are total 25 combinations possible.
      2. The range of values returned by the above equation is 1 to 25, each integer occurring exactly once.
      3. If the value of the equation comes out to be less than 22, return modulo division by 7 followed by adding 1. Else, again call the method recursively. The probability of returning each integer thus becomes 1/7.

   4. Make a fair coin from a biased coin: You are given a function foo() that represents a biased coin. When foo() is called, it returns 0 with 60% probability, and 1 with 40% probability. Write a new function that returns 0 and 1 with 50% probability each. 

      If we can somehow get two cases with equal probability, then we are done. We call foo() two times. Both calls will return 0 with 60% probability. So the two pairs (0, 1) and (1, 0) will be generated with equal probability from two calls of foo(). Let us see how.

      **(0, 1):** The probability to get 0 followed by 1 from two calls of foo() = 0.6 * 0.4 = 0.24
      **(1, 0):** The probability to get 1 followed by 0 from two calls of foo() = 0.4 * 0.6 = 0.24

      So the two cases appear with equal probability. The idea is to return consider only the above two cases, return 0 in one case, return 1 in other case. For other cases [(0, 0) and (1, 1)], recur until you end up in any of the above two cases.

   5. Shuffle a given array: Given an array, write a program to generate a random permutation of array elements.

      ```python 
      To shuffle an array a of n elements (indices 0..n-1):
        for i from n - 1 downto 1 do
             j = random integer with 0 <= j <= i
             exchange a[j] and a[i]
      ```

   6. [Reservoir sampling](http://en.wikipedia.org/wiki/Reservoir_sampling) is a family of randomized algorithms for randomly choosing *k* samples from a list of *n* items,where *n* is either a very large or unknown number.It **can be solved in O(n) time**. The solution also suits well for input in the form of stream. The idea is similar to [this ](https://www.geeksforgeeks.org/archives/25111)post. Following are the steps.

      **1)** Create an array *reservoir[0..k-1]* and copy first *k* items of *stream[]* to it.
      **2)** Now one by one consider all items from *(k+1)*th item to *n*th item.
      	**a)** Generate a random number from 0 to *i* where *i* is index of current item in *stream[]*. Let the generated random number is *j*.
      	**b)** If *j* is in range 0 to *k-1*, replace *reservoir[j]* with *arr[i]*

   7. Select random number from stream with O(1) extra space: Given a stream of numbers, generate a random number from the stream. You are allowed to use only O(1) space and the input is in the form of stream, so can’t store the previously seen numbers.

      **1)** Initialize ‘count’ as 0, ‘count’ is used to store count of numbers seen so far in stream.
      **2)** For each number ‘x’ from stream, do following
      …..**a)** Increment ‘count’ by 1.
      …..**b)** If count is 1, set result as x, and return result.
      …..**c)** Generate a random number from 0 to ‘count-1’. Let the generated random number be i.
      …..**d)** If i is equal to ‘count – 1’, update the result as x.

   8. Random Number generator in arbitrary distribution: Given n numbers, each with some frequency of occurrence. Return a random number with probability proportional to its frequency of occurrence.

      1. Use a cumsum() to generate a prefix array, randomly generate number up to the last element of the prefix array
      2. Return number according to the range of the random number generated falls into.

   9. Function to generate one of 3 numbers according to given probabilities:  You are given a function rand(a, b) which generates equiprobable random numbers between [a, b] inclusive. Generate 3 numbers x, y, z with probability P(x), P(y), P(z) such that P(x) + P(y) + P(z) = 1 using the given rand(a,b) function.

      ```java
      // This function generates 'x' with probability px/100, 'y' with 
      // probability py/100  and 'z' with probability pz/100:
      // Assumption: px + py + pz = 100 where px, py and pz lie 
      // between 0 to 100 
      int random(int x, int y, int z, int px, int py, int pz)
      {       
              // Generate a number from 1 to 100
              int r = rand(1, 100);
            
              // r is smaller than px with probability px/100
              if (r <= px)
                  return x;
       
               // r is greater than px and smaller than or equal to px+py 
               // with probability py/100 
              if (r <= (px+py))
                  return y;
       
               // r is greater than px+py and smaller than or equal to 100 
               // with probability pz/100 
              else
                  return z;
      }
      ```

   10. Find the kth smallest element in an array:

       1. We can also use Max Heap for finding the k’th smallest element. Following is algorithm.
          1) Build a Max-Heap MH of the first k elements (arr[0] to arr[k-1]) of the given array. O(k)

          2) For each element, after the k’th element (arr[k] to arr[n-1]), compare it with root of MH.
          ……a) If the element is less than the root then make it root and call heapify for MH
          ……b) Else ignore it.
          // The step 2 is O((n-k)*logk)

          3) Finally, root of the MH is the kth smallest element.

          Time complexity of this solution is O(k + (n-k)*Logk)

       2. This is an optimization over method 1 if [QuickSort ](http://geeksquiz.com/quick-sort/)is used as a sorting algorithm in first step. In QuickSort, we pick a pivot element, then move the pivot element to its correct position and partition the array around it. The idea is, not to do complete quicksort, but stop at the point where pivot itself is k’th smallest element. Also, not to recur for both left and right sides of pivot, but recur for one of them according to the position of pivot. The worst case time complexity of this method is O(n2), but it works in O(n) on average.

       3. The idea is to randomly pick a pivot element. To implement randomized partition, we use a random function, [rand()](http://www.cplusplus.com/reference/cstdlib/rand/) to generate index between l and r, swap the element at randomly generated index with the last element, and finally call the standard partition process which uses last element as pivot. Linear time expected time complexity.

   11. Loading Balance on Servers:  Pick a random server and assign the request to it , every time a new request comes in. The average load on each server is (TotalLoad)/n

   12. **How to select a random node with only one traversal allowed?**
       The idea is to use [Reservoir Sampling](https://www.geeksforgeeks.org/reservoir-sampling/). Following are the steps. This is a simpler version of [Reservoir Sampling](https://www.geeksforgeeks.org/reservoir-sampling/) as we need to select only one key instead of k keys.

       ```python
       (1) Initialize result as first node
          result = head->key 
       (2) Initialize n = 2
       (3) Now one by one consider all nodes from 2nd node onward.
           (3.a) Generate a random number from 0 to n-1. 
                Let the generated random number is j.
           (3.b) If j is equal to 0 (we could choose other fixed number 
                 between 0 to n-1), then replace result with current node.
           (3.c) n = n+1
           (3.d) current = current->next
       ```

       ​