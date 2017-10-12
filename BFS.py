# Program to print BFS traversal from a given source
# vertex. BFS(int s) traverses vertices reachable
# from s.
from collections import defaultdict


# This class represents a directed graph using adjacency
# list representation
class Graph:
    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # Function to print a BFS of graph
    def BFS(self, s):
        visited = [0] * len(self.graph) # boolean array to keep track of every vertices visited
        queue = [s] # queue for bfs search
        while(len(queue)!=0):
            v = queue.pop(0)
            if not visited[v]:
                print v
                visited[v] = 1
                for each in self.graph[v]:
                    queue.append(each)
# Driver code
# Create a graph given in the above diagram
g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print "Following is Breadth First Traversal (starting from vertex 2)"
g.BFS(2)