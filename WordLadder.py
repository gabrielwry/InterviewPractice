"""
Construct a tree of this word list, the most similar word is the first child, each child node has one letter differencce with the parent
bfs to find the shallowest match word

"""
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        copy =wordList[:]
        begin_queue = [beginWord]
        begin_level = 0
        begin_level_length = 0
        end_queue = [endWord]
        end_level = 0
        end_level_length = 0
        end = False
        while(not end):
        	begin_node = begin_queue.pop(0)
        	end_node = end_queue.pop(0)

        	if begin_node == '/':
        		begin_level +=1
        	if end_node == '/':
        		end_level +=1
        	else:
        		if begin_node in end_queue or end_node in begin_queue:
	        		print begin_node, end_queue
	        		return begin_level + end_level
        		for each in wordList:
	        		if stringDiff(begin_node,each) == 1:
	        			begin_queue.append(each)
	        			end = False

	        		if stringDiff(end_node,each) == 1:
	        			end_queue.append(each)
	        			end = False

        	begin_queue.append('/')
        	end_queue.append('/')
	        


def stringDiff(word_1,word_2):
	diff = 0
	for i in range(0,len(word_1)):
		if word_1[i] != word_2[i]:
			diff+=1
	return diff

solution = Solution()
print solution.ladderLength('hit','cog',["hot","dot","dog","lot","cog"])