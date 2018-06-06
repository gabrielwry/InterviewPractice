"""
You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

Solution: two 1 step equal one 2 step, do substitution of all 2 steps solution

"""

import math
def combination(n,m):
    return math.factorial(n)/(math.factorial(m)*(math.factorial(n-m)))
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n%2:
            return



solution = Solution()
print combination(8,2)
print solution.climbStairs(10)