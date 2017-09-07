class Solution(object):
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums)<3:
            return False
        min_1 = float('inf')
        min_1_list = []
        min_2 = nums[-1]
        min_2_list = []
        found = False
        for each in nums:
            if each < min_1:
                min_1_list.append(each)
                min_1 = each
            else: 
                min_1_list.append(min_1)
        print nums
        print min_1_list
        for i in range(0,len(nums)-1):
            print nums[len(nums)-1-i]
            if nums[i]> min_2 and nums[i] > min_1_list[i] and min_2 > min_1:
                return True
            if nums[i] < min_2:
                min_2 = nums[i]
        return False
                