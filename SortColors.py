"""
Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
"""


class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n_0 = 0
        n_1 = 0
        n_2 = 0
        for i in range(0,len(nums)):
            print 'n_0 is %d, n_1 is %d, n_2 is %d'%(n_0,n_1,n_2)
            if nums[i] == 0:
                n_0 +=1
                nums.insert(n_0-1,nums.pop(i))
                print nums
                continue
            if nums[i] == 1:
                n_1 +=1
                nums.insert(n_0+n_1-1,nums.pop(i))
                print nums
                continue
            if nums[i] == 2:
                n_2 +=1
                nums.insert(n_0+n_1+n_2-1,nums.pop(i))
                print nums
                continue


solution = Solution()
test = [0,1,0,2,1,2,0,0,0,0]
solution.sortColors(test)
print test