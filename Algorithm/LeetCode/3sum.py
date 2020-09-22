"""
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Notice that the solution set must not contain duplicate triplets.

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Example 2:

Input: nums = []
Output: []
Example 3:

Input: nums = [0]
Output: []

Constraints:

0 <= nums.length <= 3000
-105 <= nums[i] <= 105
"""


class Solution:
    def threeSum(self, nums):
        """
        Input: nums = [-1,0,1,2,-1,-4]
        Output: [[-1,-1,2],[-1,0,1]]

        """

        def _is_duplicate(i, j, two_sums):
            # check for duplicate
            # i, j and 0-i-j
            if i in two_sums:
                if j in two_sums[i] or 0-i-j in two_sums[i]:
                    return True
            if j in two_sums:
                if i in two_sums[j] or 0-i-j in two_sums[j]:
                    return True
            if 0-i-j in two_sums:
                if i in two_sums[0-i-j] or j in two_sums[0-i-j]:
                    return True
            return False

        two_sums = {}
        results = []
        for index, i in enumerate(nums):
            if i not in two_sums:  # do not process duplicated entry
                two_sums[i] = {}
                rem = nums[index + 1::]
                for _index, j in enumerate(rem):  # prevent duplicates
                    if 0 - (j + i) in rem[_index + 1::]:
                        print(i, j, two_sums, _is_duplicate(i, j, two_sums))
                        if not _is_duplicate(i, j, two_sums):
                            two_sums[i][j] = 0 - (j + i)
                            results.append([i, j, 0 - (j + i)])
        return results
    """
    class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res, dups = set(), set()
        seen = {}
        for i, val1 in enumerate(nums):
            if val1 not in dups:
                dups.add(val1)
                for j, val2 in enumerate(nums[i+1:]):
                    complement = -val1 - val2
                    if complement in seen and seen[complement] == i:
                        res.add(tuple(sorted((val1, val2, complement))))
                    seen[val2] = i
        return res
    """


print(Solution().threeSum([-1,0,1,2,-1,-4,-2,-3,3,0,4]))
