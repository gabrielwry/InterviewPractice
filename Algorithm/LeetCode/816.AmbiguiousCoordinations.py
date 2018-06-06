"""
We had some 2-dimensional coordinates, like "(1, 3)" or "(2, 0.5)".  Then, we removed all commas, decimal points, and spaces, and ended up with the string S.  Return a list of strings representing all possibilities for what our original coordinates could have been.

Our original representation never had extraneous zeroes, so we never started with numbers like "00", "0.0", "0.00", "1.0", "001", "00.01", or any other number that can be represented with less digits.  Also, a decimal point within a number never occurs without at least one digit occuring before it, so we never started with numbers like ".1".

The final answer list can be returned in any order.  Also note that all coordinates in the final answer have exactly one space between them (occurring after the comma.)

Example 1:
Input: "(123)"
Output: ["(1, 23)", "(12, 3)", "(1.2, 3)", "(1, 2.3)"]
Example 2:
Input: "(00011)"
Output:  ["(0.001, 1)", "(0, 0.011)"]
Explanation: 
0.0, 00, 0001 or 00.01 are not allowed.
Example 3:
Input: "(0123)"
Output: ["(0, 123)", "(0, 12.3)", "(0, 1.23)", "(0.1, 23)", "(0.1, 2.3)", "(0.12, 3)"]
Example 4:
Input: "(100)"
Output: [(10, 0)]
Explanation: 
1.0 is not allowed.

Note:

4 <= S.length <= 12.
S[0] = "(", S[S.length - 1] = ")", and the other elements in S are digits.
"""

# Medium, split to two substring and loop through possible dot pos, validate result and append 

class Solution:

	def validate_cord(self,x):

		if x[-1] == '.':
			x =  x[:-1]
		print(x,x[-1])
		if len(x) == 1:
			return (True,x)
		else:
			if x[0] == '0' and  x[1] != '.':
				return (False,x)
			if '.' in x and x[-1] == '0':
				return (False,x)
		return (True,x)

	def ambiguousCoordinates(self, S):
		"""
		:type S: str
		:rtype: List[str]
		"""
		candidate = S[1:-1]
		result = []

		for i in range(1,len(candidate)):
			x = candidate[:i]
			y = candidate[i:]
			# examine x
			x_cord = None
			y_cord = None
			for x_pos in range(len(x)):
				x_cord = x[:x_pos+1]+'.'+x[x_pos+1:]
				for y_pos in range(len(y)):
					y_cord = y[:y_pos+1]+'.'+y[y_pos+1:]
					if self.validate_cord(x_cord)[0] and self.validate_cord(y_cord)[0]:
						str_ = '('+self.validate_cord(x_cord)[1]+', '+\
						self.validate_cord(y_cord)[1]+')'
						result.append(str_)
		return result

    
        	




solution = Solution()
print(solution.ambiguousCoordinates('(100)'))