# Python Program for union-find algorithm to detect cycle in a undirected graph
# we have one egde for any two vertex i.e 1-2 is either 1-2 or 2-1 but not both

from collections import defaultdict


# This class represents a undirected graph using adjacency list representation
class Graph:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = defaultdict(list)  # default dictionary to store graph

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def find_parent(self,parent,i):
        if parent[i] == -1:
            return i
        else:
            return self.find_parent(parent,parent[i])

    def union(self,parent,x,y):
        parent_of_x = self.find_parent(parent,x)
        parent_of_y = self.find_parent(parent,y)
        parent[parent_of_y] = parent_of_x

    def isCyclic(self):
        parent = [-1] * self.V
        for vertex in range(self.V):
            for edge in self.graph[vertex]:
                if self.find_parent(parent,vertex) == self.find_parent(parent,edge):
                    return True
                self.union(parent,vertex,edge)

# Create a graph given in the above diagram
g = Graph(4)
g.addEdge(0, 1)
g.addEdge(2, 0)
g.addEdge(0,3)
g.addEdge(3,1)

if g.isCyclic():
    print "Graph contains cycle"
else:
    print "Graph does not contain cycle "

    # This code is contributed by Neelam Yadav