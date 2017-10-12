"""
There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

Solution: Basically detect cycle in a directed graph
"""
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        acyclic = True
        taken = [0] * numCourses
        ancestor = [0] * numCourses
        for course in range(0,numCourses):
            if not taken[course]:
                acyclic = helper(self,course,taken,ancestor,prerequisites)

        return acyclic

def helper(self,course,taken,ancestor,prerequisites):
    taken[course] = 1
    for each in prerequisites:
        if each[1] == course:
            ancestor[course] = 1
            print course,ancestor,taken
            if ancestor[each[0]] == 1:
                return False
            elif not taken[each[0]]:
                ancestor[each[0]] = 1
                taken[each[0]] = 1
                return helper(self,each[0],taken,ancestor,prerequisites)
    ancestor[course] = 0
    return True



solution = Solution()
print solution.canFinish(3,[[0,2],[1,2],[2,0]])