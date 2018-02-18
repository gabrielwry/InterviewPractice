"""
Given a sequence of integers, return the length of the longest subsequence that is a wiggle sequence.
A subsequence is obtained by deleting some number of elements (eventually, also zero) from the original sequence,
leaving the remaining elements in their original order.
"""
class Solution(object):
# This solution use a linear dynamic programming
# The max wiggle length is the previous longest plus one if the next element follows the rule
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        up = 0
        down = 0
        for i in range(1,len(nums)):
            if nums[i-1] > nums[i]:
                down = up+1
            elif nums[i-1] < nums[i]:
                up = down+1
        return max(up,down)