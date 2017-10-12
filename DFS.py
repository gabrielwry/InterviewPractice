# Python program to print DFS traversal from a
# given given graph
from collections import defaultdict


# This class represents a directed graph using
# adjacency list representation
class Graph:
    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def helper(self,v,visited):
        visited[v] = 1
        print v
        for each in self.graph[v]:
            if not visited[each]:
                self.helper(each,visited)
    def DFS(self,v):
        """

        :param v:
        :return:
        visited = [0] * len(self.graph)
        stack = list(self.graph.keys())
        while stack:
            current = stack.pop(0)
            visited[current] = 1
            print current
            for each in self.graph[current]:
                if not visited[each]:
                    stack.insert(0,each)
        """
        visited = [0] * len(self.graph)
        for each in list(self.graph.keys()):
            if not visited[each]:
                self.helper(each,visited)


# Driver code
# Create a graph given in the above diagram
g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print "Following is DFS from (starting from vertex 2)"
g.DFS(2)

# This code is contributed by Neelam Yadav