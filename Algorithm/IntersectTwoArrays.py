class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        dic_1 = {}
        dic_2 = {}
        result = []
        for each in nums1:
            if each not in dic_1:
                dic_1[each] = 1
            else:
                dic_1[each]+=1
        for each in nums2:
            if each not in dic_2:
                dic_2[each]= 1
            else:
                dic_2[each]+=1
        for each in dic_1:
            if each in dic_2:
                for k in range(0,min(dic_1[each],dic_2[each])):
                    result.append(each)
        return result