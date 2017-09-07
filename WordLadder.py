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
        child = []
        level = 0
        current = [beginWord]
        node = beginWord
        counter = 0
        while(stringDiff(node,endWord)!=0):
        	print counter,current
        	counter+=1
        	node = current.pop(0)
	        for each in wordList:
	        	if stringDiff(node,each) == 1:
	        		child.append(each)
	        		wordList.remove(each)
	        if len(current)==0:
	        	current = child
	        	level+=1
        return level

        	
	        


def stringDiff(word_1,word_2):
	diff = 0
	for i in range(0,len(word_1)):
		if word_1[i] != word_2[i]:
			diff+=1
	return diff

solution = Solution()
print solution.ladderLength('hellp','helao',['tellp','tello','hello','helao'])