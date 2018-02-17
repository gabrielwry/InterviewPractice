# A Python program for Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix representation of the graph

import sys  # Library for INT_MAX


class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printMST(self, parent):
        print "Edge \tWeight"
        for i in range(1, self.V):
            print parent[i], "-", i, "\t", self.graph[i][parent[i]]

    def minVertex(self,dist,mstSet):
        min_ = float('inf')
        min_v = None
        for vertex in range(self.V):
            if vertex not in mstSet:
                if dist[vertex] < min_:
                    min_ = dist[vertex]
                    min_v = vertex
        return min_v

    def primMST(self):
        mstSet = []
        edges = []
        dist = [float('inif')] * self.V
        min_v = 0
        dist[min_v] = 0
        while(len(mstSet)!= self.V):
            mstSet.append(min_v)
            min_v = self.minVertex(dist,mstSet)
            for each in range(self.V):
                if self.graph[min_v][each]!=0:
                    # TODO: finish this implementation
                    return

g = Graph(5)
g.graph = [[0, 2, 0, 6, 0],
           [2, 0, 3, 8, 5],
           [0, 3, 0, 0, 7],
           [6, 8, 0, 0, 9],
           [0, 5, 7, 9, 0],
           ]

g.primMST();

# Contributed by Divyanshu Mehta