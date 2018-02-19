"""Given two strings representing two complex numbers.

You need to return a string representing their multiplication. Note i2 = -1 according to the definition."""
class Solution(object):
    def complexNumberMultiply(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        a_real, a_complex = a.split('+')
        b_real, b_complex = b.split('+')
        result_real = int(a_real) * int(b_real) - int(a_complex[:-1]) * int(b_complex[:-1])
        result_complex = int(a_real) * int(b_complex[:-1]) + int(b_real) * int(a_complex[:-1])
        result = str(result_real)+'+'+str(result_complex)+'i'
        return result
        
# easy peasy, lemon squeezy