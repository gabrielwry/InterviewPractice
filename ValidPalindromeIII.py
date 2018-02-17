"""Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome."""
# greedy solution: when see a missmatch, compare the two scenario of whether next char matched
class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        i = 0
        j = len(s) - 1
        while i <= j:
            if s[i] != s[j]:
                if s[i + 1] == s[j]:

                    if checkPalindrome(s[i:j]):
                        return True
                elif s[i] == s[j - 1]:

                    print s[i:j], j
                    return checkPalindrome(s[i:j])
                else:
                    return False
            i += 1
            j -= 1
        return True


def checkPalindrome(s):
    i = 0
    j = len(s) - 1
    while i <= j:
        if s[i] != s[j]:
            return False
        else:
            i += 1
            j -= 1
    return True
