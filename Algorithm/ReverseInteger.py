"""
Reverse digits of an integer.

Example1: x = 123, return 321
Example2: x = -123, return -321

"""
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        rev = ''
        for i in range(len(str(abs(x)))-1,-1,-1):
            rev+=str(abs(x))[i]
        result = int(rev) if (abs(int(rev)) <= 2**31-1) else 0
        return result if x>=0 else -result
        
        