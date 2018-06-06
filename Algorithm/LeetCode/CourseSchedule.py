"""
There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

Solution: Basically detect cycle in a directed graph
"""
class Solution(object):
    def helper(self, course, taken, ancestor, prerequisites):
        taken[course] = 1
        ancestor[course] = 1
        for each in prerequisites:
            if each[1] == course:
                if not taken[each[0]]:
                    if self.helper(course,taken,ancestor,prerequisites):
                        return True
                elif ancestor[each[0]]:
                    return True
        ancestor[course] = 0
        return False

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        taken = [0] * numCourses
        ancestor = [0] * numCourses
        for course in range(numCourses):
            if not taken[course]:
                if self.helper(course,taken,ancestor,prerequisites):
                    return False
        return True


solution = Solution()
print solution.canFinish(2,[[1,0],[0,1]])