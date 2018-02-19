## Algorithm



**1. Graph** 

1. [Shortest Path from source to all vertices **Dijkstra** ](https://www.geeksforgeeks.org/greedy-algorithms-set-6-dijkstras-shortest-path-algorithm/)

   - variation: 
     - negative weight (Bellman-Ford)- relax all edges $|V|-1$  times instead of using a priority queue
     - path: keep a parent array, update the array every time find a smaller distance

   ```python
   """Given a graph and a source vertex in graph, find shortest paths from source to all vertices in the given graph."""
   graph = Graph()
   V = graph.vertices()
   def find_min_index(dist,sptSet): # use a Priority Queue to keep poping out the one with min dist vertices for test
       min_dist = float('inf')
       min_index = 0
       for i in range(len(dist)):
           if i not in sptSet:
               if dist[i]<min_dist:
                   min_dist = dist[i]
                   min_index = i
       return min_index            
                   
   def dijkstra(graph, s):
       sptSet = Set()
       vertices = V #create two empty sets, one with finialized shortes path, one with not computed vertices
       dist = [float'inf']*len(V)
       dist[s] = 0
       
       while V:
           u = find_min_index(dist,sptSet)
           sptSet.append(u)
           V.remove(u)
           for v in V:
               if isConnected(v,u):
                   if dist[v]>(graph[u][v]+dist[u]):
                       dist[v] = graph[u][v]+dist[u]
       return dist
   ```

   ​

2. [Shortest Path from every vertex to every other vertex **Floyd Warshall**](https://www.geeksforgeeks.org/dynamic-programming-set-16-floyd-warshall-algorithm/)

   ```python
   """The problem is to find shortest distances between every pair of vertices in a given edge weighted directed Graph."""
   graph = Graph()
   V = graph.vertices()
   # pick all vertices one-by-one and consider it as an intermediate vertex of some shortest path (consider all shortest path (i,j) if dist[i][j]>dist[i][k]+dist[k][j])
   def floyd_warshall(graph):
       dist=map(lambda i: map(lambda j:j , i), graph) # initate the solution matrix as the graph, lambda function intiate through row and columns;
      	for k in range(len(graph)):
           for i in range(len(graph)):
               for j in range:
                   dist[i][j] = min(dist[i][j],
                                   dist[i][k]+dist[k][j])
      	return dist 
   ```

   ​

3. [To detect cycle in a Graph **Union Find**](https://www.geeksforgeeks.org/union-find/)

   ```python
   """A disjoint-set data structure is a data structure that keeps track of a set of elements partitioned into a number of disjoint (non-overlapping) subsets.  Two vertices of an edge in the same sub set will indicate a loop was detected"""
   graph = Graph()
   V = graph.vertices()
   E = graph.edges()
   parent = [-1]*len(graph)
   def quick_union(s,v):
       for i in range(len(parent)):
           if parent[i] == s:
               parent[i] = v
   def quick_find(s):
       return parent[s] 
   #============The above implementation is a quick-find==============#
   def union(s,v):
       if parent[s] != -1:
           return union(parent[s],v) #track down to the first ancestor
       else:
           parent[s] = v #set the parent of the first ancestor to v
       
   def find(s):
       if parent[s] != -1:
           return find([parent[s]])
       else:
           return s # find the first ancestor
   #===============Union and Find with O(log(N))================#
   def detect_cycle(graph):
       isCyclic = False
       for edge in E:
           if find(edge[0]) == find(edge[1]):
               isCyclic = True
               break
       return isCyclic
   ```

   ​

4. [Minimum Spanning tree **Prim** ](https://www.geeksforgeeks.org/greedy-algorithms-set-5-prims-minimum-spanning-tree-mst-2/)

   ```python
   """For MST: Prim grows a connected graph; Kruskal grow forest until V-1 edges; Prim works in dense graph where edges are far more than vertices, Kruskal works well in sparse graph;
   """
   graph = Graph()
   V = graph.vertices()
   E = graph.edges()
   def find_min_index(dist,V)
   	min_index = 0
       min_dist = float('inf')
       for v in V:
           if dist[v] < min_dist:
               min_dist = dist[v]
               min_index = v
       return min_dist
       
       
   def prim(graph):
       mstSet = Set()
       parent = [-1]*len(graph)
       dist= [float('inf')]*len(graph) # initiate all dist as inf
       dist[0] = 0 # pick a key to start with
       u=0
       while V: #iterate through all vertices
           mstSet.append(u) #update the mstSet with the examined vertex
           V.remove(u)
       	for v in V:
               if isConnected(v,u):
                   if dist[v]>graph[v][u]+dist[u]:#check if the current v is larger than dist[u] + the edge weight
                       dist[v]=graph[v][u]+dist[u] #update the distance if larger
                       parent[v] = u #update the parent 
            u = find_min_index(dist,V) #examine the next vertex

     	return parent

   for v in V:
       print parent[v],"-",v
       
                       
   ```

   ​

5. [Minimum Spanning tree **Kruskal** ](https://www.geeksforgeeks.org/greedy-algorithms-set-2-kruskals-minimum-spanning-tree-mst/)

   ```python
   """What is Minimum Spanning Tree?
   Given a connected and undirected graph, a spanning tree of that graph is a subgraph that is a tree and connects all the vertices together. A single graph can have many different spanning trees. A minimum spanning tree (MST) or minimum weight spanning tree for a weighted, connected and undirected graph is a spanning tree with weight less than or equal to the weight of every other spanning tree. The weight of a spanning tree is the sum of weights given to each edge of the spanning tree."""
   graph = Graph()
   V = graph.vertices()
   E = graph.edges()
   parents = [-1]*len(graph)
   def union(s,v):
       if parent[s] != -1:
           return union(parent[s],v) #track down to the first ancestor
       else:
           parent[s] = v #set the parent of the first ancestor to v
       
   def find(s):
       if parent[s] != -1:
           return find([parent[s]])
       else:
           return s # find the first ancestor
       
   def detect_cycle(E,v,u):
       union(v,u)
       isCyclic = False
       for edge in E:
           if find(edge[0]) == find(edge[1]):
               isCyclic = True
               break
       return isCyclic
   #=================Utility Function to detect loop======================#
   def kruskal(graph):
       edges = Set()
       E = sorted(E,key = lambda x: x.weight)# sort edges for greedy adding to the Set
       while len(edges) < len(graph)-1: # need V-1 edges for the MST
       	e = E.pop(0)
           if find(e[0]) != find(e[1]):
               edges.append(e)
               union(e[0],e[1])
       return edges
           
   ```

   ​

**2. Linked List**

- Basic:

  - Linear Data Structure, connected with pointers;
  - Pro: Dynamic size, easy to insert and delete; Con: Have to access element from the start, extra memory for pointer

- Problem:

  1. [Insertion of a node in Linked List (On the basis of some constraints)](https://www.geeksforgeeks.org/given-a-linked-list-which-is-sorted-how-will-you-insert-in-sorted-way/)

     ```python
     """Given a sorted linked list and a value to insert, write a function to insert the value in sorted way."""
     def insert(linked_list,node):
         n = linked_list.head
         if !n:
             node = head
             return
         if node.value< n.value:
             node.next = head
             head = node
             return
         else:
             flag = False
             while n and not flag:
                 if n.value < node.value:
                     if n.next.value > node.value:
                         node.next = n.next
                         n.next = node
                         flag = True
                         return
                     else:
                         n = node.next
              	else:
                     #this should never happen
                     return
                     
                     
     ```

     ​

  2. [Delete a given node in Linked List (under given constraints)](https://www.geeksforgeeks.org/delete-a-given-node-in-linked-list-under-given-constraints/)

     ```python
     """Given a Singly Linked List, write a function to delete a given node. Your function must follow following constraints:
     1) It must accept pointer to the start node as first parameter and node to be deleted as second parameter i.e., pointer to head node is not global.
     2) It should not return pointer to the head node.
     3) It should not accept pointer to pointer to head node.
     """
     def delete(linked_list,node):
         n = linked_list.head
         if n.value == node.value:
             head = head.next
         while n:
             if n.next.value == node.value:
                 n.next = n.next.next
                 
     ```

     ​

  3. [Add Two Numbers Represented By Linked Lists](https://www.geeksforgeeks.org/sum-of-two-linked-lists/)

     ```python
     """Given two numbers represented by two linked lists, write a function that returns sum list. The sum list is linked list representation of addition of two input numbers. It is not allowed to modify the lists. Also, not allowed to use explicit extra space (Hint: Use Recursion)."""
     def count_size(linked_list):
        	count = 0
         stack = []
         n = linked_list.head
         while n:
             count+=1
             n=n.next
             stack.insert(0,n.value)# push the last digit to the top of stack
         return count, stack
     def add_two_numbers(n_1,n_2):
         size_1,stack_1 = count_size(n_1)
         size_2,stack_2 = count_size(n_2)
         result = []
         a = n_1.head
         b = n_2.head
         carry = False
         if size_1 > size_2:
             #fill n_2 with 0
             for i in range(size_1-size_2):
                 n_2.insert(-1,0)
         if size_1 == size_2:
             
             for i in range(size_1):
                 c = a+b
                 if carry:
                 	c = c+1
                     carry = False
                 if c>=10:
                     carry = True
                     c = c-10
                 result.insert(-1,c)
            	return result
           	
                 
     ```

     ​

  4. [Reverse A List In Groups Of Given Size](https://www.geeksforgeeks.org/reverse-a-list-in-groups-of-given-size/)

  5. [Union And Intersection Of 2 Linked Lists](https://www.geeksforgeeks.org/union-and-intersection-of-two-linked-lists/)

  6. [Detect And Remove Loop In A Linked List](https://www.geeksforgeeks.org/detect-and-remove-loop-in-a-linked-list/)

  7. [Merge Sort For Linked Lists](https://www.geeksforgeeks.org/merge-sort-for-linked-list/)

  8. [Select A Random Node from A Singly Linked List](https://www.geeksforgeeks.org/select-a-random-node-from-a-singly-linked-list/)

1. **3. Dynamic Programming**

   Dynamic Programming solves a given complex problem by breaking into subproblems and store the results of subproblems to avoid computing the same result again. Two main properties of a problem suggests that it can solved by a dynamic programming 1) Overlapping Subproblems 2) Optimal Substructure 

   1. [Longest Common Subsequence](https://www.geeksforgeeks.org/dynamic-programming-set-4-longest-common-subsequence/)

      ```python
      """Given two sequences, find the length of longest subsequence present in both of them. A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous. For example, “abc”, “abg”, “bdf”, “aeg”, ‘”acefg”, .. etc are subsequences of “abcdefg”. So a string of length n has 2^n different possible subsequences."""
      # This can be solved by recursively checking if the last character of the two array match, if it is match, then the max length is L(X[0..m-1], Y[0..n-1]) = 1 + L(X[0..m-2], Y[0..n-2]), if it doesn't match L(X[0..m-1], Y[0..n-1]) = MAX ( L(X[0..m-2], Y[0..n-1]), L(X[0..m-1], Y[0..n-2])
      ```

```
  ```
```

```
  ​
```

1. [Longest Increasing Subsequence](https://www.geeksforgeeks.org/dynamic-programming-set-3-longest-increasing-subsequence/)

   ```python
   """The Longest Increasing Subsequence (LIS) problem is to find the length of the longest subsequence of a given sequence such that all elements of the subsequence are sorted in increasing order. For example, the length of LIS for {10, 22, 9, 33, 21, 50, 41, 60, 80} is 6 and LIS is {10, 22, 33, 50, 60, 80}."""
   #  This can be solved by recursively considering i th element as the last of LIS, and L(i) = 1 + max( L(j) ) where 0 < j < i and arr[j] < arr[i]; or L(i) = 1, if no such j exists.
   ```

   ​

2. [Edit Distance](https://www.geeksforgeeks.org/dynamic-programming-set-5-edit-distance/)

   ```python
   """Given two strings str1 and str2 and below operations that can performed on str1. Find minimum number of edits (operations) required to convert ‘str1’ into ‘str2’.

   Insert
   Remove
   Replace"""
   #If last characters of two strings are same, nothing much to do. Ignore last characters and get count for remaining strings. So we recur for lengths m-1 and n-1.Else (If last characters are not same), we consider all operations on ‘str1’, consider all three operations on last character of first string, recursively compute minimum cost for all three operations and take minimum of three values.Insert: Recur for m and n- 1Remove: Recur for m-1 and n Replace: Recur for m-1 and n-1
   ```

   ​

3. [Minimum Partition](https://www.geeksforgeeks.org/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum/)

4. [Ways to Cover a Distance](https://www.geeksforgeeks.org/count-number-of-ways-to-cover-a-distance/)

5. [Longest Path In Matrix](https://www.geeksforgeeks.org/find-the-longest-path-in-a-matrix-with-given-constraints/)

6. [Subset Sum Problem](https://www.geeksforgeeks.org/dynamic-programming-subset-sum-problem/)

7. [Optimal Strategy for a Game](https://www.geeksforgeeks.org/dynamic-programming-set-31-optimal-strategy-for-a-game/)

8. [0-1 Knapsack Problem](https://www.geeksforgeeks.org/dynamic-programming-set-10-0-1-knapsack-problem/)

9. [Boolean Parenthesization Problem](https://www.geeksforgeeks.org/dynamic-programming-set-37-boolean-parenthesization-problem/)

**4. Sorting And Searching**

1. [Merge Sort](http://geeksquiz.com/merge-sort/)

   ```python
   #Divide and Conquer, split the array to two halves and keep sorting and merging them
   def merge(arr, l,m,r): #always use arr_2 as the longer array
       arr_1 = arr[l:m]
       arr_2 = arr[m:r]
       result = [None]*(len(arr_1)+len(arr_2)) # the result array
       index = 0
       i = 0
       j = 0
       while i < len(arr_1) and j < len(arr_2):
           if arr_1[i] > arr_2[j]:
               result[index] = arr_2[j]
               j += 1
           else:
               result[index] = arr_1[i]
               i += 1
               
           index += 1
       # after existing the while loop, copy the remaining of either array into result
       for x in range(i,len(arr_1)):
           result[index] = arr_1[x]
           index+=1
       for x in range(j, len(arr_2)):
           result[index] = arr_2[x]
           index+=1
           
      	return result

    def merge_sort(arr_,l,r):
       
       
       if l<r:
           m=(l+(r-1))/2
       	merge_sort(arr_,l,m)
           merge_sort(arr_,m,r)
      		merge(arr_,l,m,r)     
               
       
   ```

   ​

2. [Heap Sort (Binary Heap)](http://geeksquiz.com/heap-sort/)

   ```python
   # 1. Construct Heap -> keep removing the largest item
   # Python program for implementation of heap Sort
    
   # To heapify subtree rooted at index i.
   # n is size of heap
   def heapify(arr, n, i):
       largest = i  # Initialize largest as root
       l = 2 * i + 1     # left = 2*i + 1
       r = 2 * i + 2     # right = 2*i + 2
    
       # See if left child of root exists and is
       # greater than root
       if l < n and arr[i] < arr[l]:
           largest = l
    
       # See if right child of root exists and is
       # greater than root
       if r < n and arr[largest] < arr[r]:
           largest = r
    
       # Change root, if needed
       if largest != i:
           arr[i],arr[largest] = arr[largest],arr[i]  # swap
    
           # Heapify the root.
           heapify(arr, n, largest)
    
   # The main function to sort an array of given size
   def heapSort(arr):
       n = len(arr)
    
       # Build a maxheap.
       for i in range(n, -1, -1):
           heapify(arr, n, i)
    
       # One by one extract elements
       for i in range(n-1, 0, -1):
           arr[i], arr[0] = arr[0], arr[i]   # swap
           heapify(arr, i, 0)
    
   # Driver code to test above
   arr = [ 12, 11, 13, 5, 6, 7]
   heapSort(arr)
   n = len(arr)
   print ("Sorted array is")
   for i in range(n):
       print ("%d" %arr[i]),
   ```

   ​

3. [Quick Sort](http://geeksquiz.com/quick-sort/)

   ```python
   #Divide and Conquer
   #partition around an element, so that the partition element is inplace and recursively call the method on the new arrays

   ```

   ​

**5. Tree / Binary Search Tree**

1. [Find Minimum Depth of a Binary Tree](https://www.geeksforgeeks.org/find-minimum-depth-of-a-binary-tree/)
2. [Maximum Path Sum in a Binary Tree](https://www.geeksforgeeks.org/find-maximum-path-sum-in-a-binary-tree/)
3. [Check if a given array can represent Preorder Traversal of Binary Search Tree](https://www.geeksforgeeks.org/check-if-a-given-array-can-represent-preorder-traversal-of-binary-search-tree/)
4. [Check whether a binary tree is a full binary tree or not](https://www.geeksforgeeks.org/check-whether-binary-tree-full-binary-tree-not/)
5. [Bottom View Binary Tree](https://www.geeksforgeeks.org/bottom-view-binary-tree/)
6. [Print Nodes in Top View of Binary Tree](https://www.geeksforgeeks.org/print-nodes-top-view-binary-tree/)
7. [Remove nodes on root to leaf paths of length < K](https://www.geeksforgeeks.org/remove-nodes-root-leaf-paths-length-k/)
8. [Lowest Common Ancestor in a Binary Search Tree](https://www.geeksforgeeks.org/lowest-common-ancestor-in-a-binary-search-tree/)
9. [Check if a binary tree is subtree of another binary tree](https://www.geeksforgeeks.org/check-binary-tree-subtree-another-binary-tree-set-2/)
10. [Reverse alternate levels of a perfect binary tree](https://www.geeksforgeeks.org/reverse-alternate-levels-binary-tree/)

**6. String / Array**

1. [Reverse an array without affecting special characters](https://www.geeksforgeeks.org/reverse-an-array-without-affecting-special-characters/)

   1. iterate the array from head `arr[i]` and tail `arr[j]`
   2. if `arr[i]` is not `alph`,`i++`
   3. if `arr[j]` is not `alph`, `j--`
   4. if both `arr[i] arr[j]` are alph, swap`arr[i] arr[j]` `i++ j--`

2. [All Possible Palindromic Partitions](https://www.geeksforgeeks.org/given-a-string-print-all-possible-palindromic-partition/)

   1. ```python
      #1. checkPalindrome method:
      #   1. start from arr[i] and arr[j] which are the head and tail of array
      #   2. compare arr[i] arr[j] until i>=j
      #2. iterate through all the substring of string
      #	1. i=0 j=len(arr)
      #		1. print all char from arr[i] till arr[j]
      #		2. i++, j--

      ```

3. [Count triplets with sum smaller than a given value](https://www.geeksforgeeks.org/count-triplets-with-sum-smaller-that-a-given-value/)

   ```python
   #1. Sort the array
   #2. run a loop from i=0 to n-2 to find all the triplets with arr[i] as the first element
   #	1. j=i+1 k = -1;
   #	2. while j<=k
   #	3. compute sum = arr[i]+arr[j]+arr[k]
   #		1. if 	sum <= K, then all k-j is smaller than K, count+=(k-j), break, j++
   #		2. else k--
   #	4. print count	
   ```

   ​

4. [Convert array into Zig-Zag fashion](https://www.geeksforgeeks.org/convert-array-into-zig-zag-fashion/)

   ```python
   # Need to consider at most three elements together
   FLAG = True
   for i in range(n-1):
       if FLAG:# < expected
           if arr[i]>arr[i+1]:# A>B
               tmp=arr[i+1]
               arr[i+1]=arr[i]
               arr[i]=tmp #now A<B
      	else: # > expected
       	if arr[i]<arr[i+1]: # A<B<C, because the previous relation is <
               arr[i],arr[i+1] = arr[i+1], arr[i]
       FLAG = not Flag
   print arr
               
   ```

   ​

5. [Generate all possible sorted arrays from alternate elements of two given sorted arrays](https://www.geeksforgeeks.org/generate-all-possible-sorted-arrays-from-alternate-elements-of-two-given-arrays/)

   ```python
   """
   Given two sorted arrays A and B, generate all possible arrays such that first element is taken from A then from B then from A and so on in increasing order till the arrays exhausted. The generated arrays should end with an element from B.
   """
   # arr: A, arr: B
     				
           
   ```

   ​

6. [Pythagorean Triplet in an array](https://www.geeksforgeeks.org/find-pythagorean-triplet-in-an-unsorted-array/)

   ```python
   """
   Given an array of integers, write a function that returns true if there is a triplet (a, b, c) that satisfies a^2 + b^2 = c^2.
   """
   def is_pythagorean(arr):
       arr = list(map(lambda x: x**2, arr)).sort() # first sort the square of the elements
       for i in range(n-1,1,-1):# fix the last digit
           j=0
           k=i-1 #test the head and tail of the arr
           while(j<k):
               if arr[i]-arr[j] == arr[k]:
                   return True # find
               else:
                   if arr[i]-arr[k] > arr[j]: #the difference is too large
                   	j++
                   else:# the difference is too small
                       k--
        return False
   ```

   ​

   ​

7. [Length of the largest subarray with contiguous elements](https://www.geeksforgeeks.org/length-largest-subarray-contiguous-elements-set-1/)

   ```python
   """
   Given an array of distinct integers, find length of the longest subarray which contains numbers that can be arranged in a continuous sequence.
   """
   def find_contiguous(arr):# test all subarray's max-min
       for i in range(len(arr)-1):# every first element 
           for j in range(len(arr)-1,i,-1):# every last element
               if max(arr[i:j])-min(arr[i:j]) == j-i:# because all elements are distinct the diffrence between max element and min element has to equal to the indices difference
                   return j-i
   ```

   ​

8. [Find the smallest positive integer value that cannot be represented as sum of any subset of a given array](https://www.geeksforgeeks.org/find-smallest-value-represented-sum-subset-given-array/)

   ```python
   """Given a sorted array (sorted in non-decreasing order) of positive numbers, find the smallest positive integer value that cannot be represented as sum of elements of any subset of given set. 
   Expected time complexity is O(n)."""
   def find_smallest(arr):
       result = 1
       for i in range(len(arr)-1):
           if result+arr[i] < arr[i+1]:
               return result+arr[i]
           result+=arr[i]
   ```

   ​

9. [Smallest subarray with sum greater than a given value](https://www.geeksforgeeks.org/minimum-length-subarray-sum-greater-given-value/)

   ```python
   """Given an array of integers and a number x, find the smallest subarray with sum greater than the given value."""
   def find_shortest(arr,X):
       sum = 0, min_length = len(arr)+1
       start = 0, end = 0
       for i in range(len(arr)):
           sum+=arr[i]#add the next element to the sum
           if sum>X:
               while sum>X and start<=i:
                   if start == i:#one element is larger than X
                       return 1
                   sum-=arr[start] #keep removing the leading element
                   start+=1
               if i-start+1 <= min_length:# update the new length
                   min_length = i-start+1
                   
       if min_length > len(arr):
           return 'Not Found'
       return min_length
               
   ```

   ​

10. [Stock Buy Sell to Maximize Profit](https://www.geeksforgeeks.org/stock-buy-sell/)

    ```python
    """The cost of a stock on each day is given in an array, find the max profit that you can make by buying and selling in those days. For example, if the given array is {100, 180, 260, 310, 40, 535, 695}, the maximum profit can earned by buying on day 0, selling on day 3. Again buy on day 4 and sell on day 6. If the given array of prices is sorted in decreasing order, then profit cannot be earned at all."""
    def max_profit(arr):
        local_min = arr[0], min_index = 0
        local_max = arr[0], max_index = 0
        total_profit = 0, local_profit = 0
        for i in range(1,len(arr)):
            if arr[i] > local_max:
                local_max = arr[i]
                max_index = i
            if arr[i] < local_min:
                if max_index > min_index:
                    total_profit+=arr[max_index]-arr[min_index]
                else:
                    total_profit+=arr[i-1]-arr[min_index]
                local_min = arr[i],local_max = arr[i]
                min_index = i, max_index = i
      	return total_profit
                
                
    ```

**7. Multithreading**



