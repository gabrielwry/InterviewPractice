# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX
import sys


class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printSolution(self, dist):
        print "Vertex tDistance from Source"
        for node in range(self.V):
            print node, "t", dist[node]

    def minDistVertex(self,dist,sptSet):
        min_ = float('inf')
        min_v = None
        for v in range(self.V):
            if v not in sptSet:
                if dist[v] < min_:
                    min_ = dist[v]
                    min_v = v
        return min_v


    def dijkstra(self,src):
        sptSet = []
        dist = [float('inf')]*self.V
        dist[src] = 0
        min_v = src
        while(len(sptSet)!= self.V):
            for vertex in range(self.V):
                if self.graph[min_v][vertex] != 0:
                    print vertex
                    if dist[vertex] > self.graph[min_v][vertex] + dist[min_v]:
                        dist[vertex] = self.graph[min_v][vertex]+dist[min_v]
            sptSet.append(min_v)
            print dist
            min_v = self.minDistVertex(dist,sptSet)
        self.printSolution(dist)




# Driver program
g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
           [4, 0, 8, 0, 0, 0, 0, 11, 0],
           [0, 8, 0, 7, 0, 4, 0, 0, 2],
           [0, 0, 7, 0, 9, 14, 0, 0, 0],
           [0, 0, 0, 9, 0, 10, 0, 0, 0],
           [0, 0, 4, 14, 10, 0, 2, 0, 0],
           [0, 0, 0, 0, 0, 2, 0, 1, 6],
           [8, 11, 0, 0, 0, 0, 1, 0, 7],
           [0, 0, 2, 0, 0, 0, 6, 7, 0]
           ];

g.dijkstra(0);

# This code is contributed by Divyanshu Mehta