"""
Given two strings s and t, write a function to determine if t is an anagram of s.

For example,
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false.

Note:
You may assume the string contains only lowercase alphabets.

Follow up:
What if the inputs contain unicode characters? How would you adapt your solution to such case?
"""

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        s_dic = {}
        t_dic = {}
        for each in s:
            if each not in s_dic:
                s_dic[each] = 1
            else:
                s_dic[each]+=1
        for each in t:
            if each not in s_dic:
                return False
            else:
                s_dic[each] -= 1
                if s_dic[each]<0:
                    return False
        return True


solution = Solution()
print solution.isAnagram('cat','tac')
