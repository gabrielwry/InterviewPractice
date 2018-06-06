"""
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.
Thought: expand each center, O(N^2) time complexity, O(1) space
"""
def testPalindrome(s):
    	for i in range(0,len(s)):
    		if s[i] != s[-i-1]:
    			return False
    	return True
def countChar(s):
	count = 0
	for each in s:
		if each!='!':
			count+=1
	return count

class Solution(object):


    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        str_list = list(''.join(map(lambda x: x+'!',s)))[:-1]
        max_ = -float('inf')
        sub_str=str_list[0]
        for i in range(1,len(str_list)-1):

        	test_ = list(str_list[i])
        	lo = i
        	hi = i
        	while(testPalindrome(test_)):
        		print test_
        		if countChar(test_) > max_:
        			max_ = countChar(test_)
        			sub_str = test_[:]
        			print 'sub_str now is ', sub_str
        		lo-=1
        		hi+=1
        		if lo==-1 or hi==len(str_list):
        			break
        		test_.insert(0,str_list[lo])
        		test_.insert(len(test_),str_list[hi])
       	#print sub_str
    	return ''.join(sub_str).replace('!','')

solution = Solution()
#print testPalindrome()
print solution.longestPalindrome('abb')