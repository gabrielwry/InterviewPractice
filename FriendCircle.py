"""There are N students in a class. Some of them are friends, while some are not.

 Their friendship is transitive in nature. For example, if A is a direct friend of B, and B is a direct friend of C, then A is an indirect friend of C.
 And we defined a friend circle is a group of students who are direct or indirect friends.

Given a N*N matrix M representing the friend relationship between students in the class. If M[i][j] = 1,
then the ith and jth students are direct friends with each other, otherwise not. And you have to output the total number of friend circles among all the students."""

# use dfs, not optimal
class Solution(object):
    def helper(self, M, visited, student):  # use helper to dfs every students
        visited.append(student)
        for x in range(0, len(M[student])):
            if x not in visited:
                if M[student][x] == 1:
                    print 'at column', x
                    self.helper(M, visited, x)

    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        N = len(M)  # student number
        students = range(0, N)
        visited = []
        count = 0
        while len(visited) != N:
            student = students.pop()
            print 'at row', student
            if student not in visited:
                count += 1  # initiate a new friend circle
                self.helper(M, visited, student)
        return count



