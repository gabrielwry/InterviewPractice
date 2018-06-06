"""
Set of n! permutation, consists of number from 1-n
Find the kth permutation

solution:
recursive method, deal with edge case
"""
import math
def helper(n, k, result,nums):
    if n == 0:
        return result
    else:
        if k%(math.factorial(n-1))==0:
            result +=str(nums.pop((k/(math.factorial(n-1))-1)))
            return helper (n-1,math.factorial(n-1),result,nums)
        else:
            result += str(nums.pop(k/(math.factorial(n-1))))
            return helper(n-1,k%(math.factorial(n-1)),result,nums)
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        return helper(n,k,'',range(1,n+1))



solution = Solution()
print solution.getPermutation(3,1)
