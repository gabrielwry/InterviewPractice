"""
Follow up for "Remove Duplicates":
What if duplicates are allowed at most twice?

For example,
Given sorted array nums = [1,1,1,2,2,3],

Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3. It doesn't matter what you leave beyond the new length.

Runs perfectly on local, index out of range error on server
"""
class Solution(object):
    """
        def removeDuplicates(self, nums):
        length = 0
        while len(nums)!= 0:
            if len(nums) == 1:
                length += 1
                break

            i = nums[0]
            j = nums[1]
            if i == j:
                length+=2
                while len(nums)>0 and nums[0] == i:
                    nums.remove(i)
            else:
                length += 1
        return length

    """

    def removeDuplicates(self, nums):
        i = 0
        for n in nums:
            if i < 2 or n > nums[i - 2]:
                nums[i] = n
                i += 1
        return i


solution= Solution()
print solution.removeDuplicates([1,1])
