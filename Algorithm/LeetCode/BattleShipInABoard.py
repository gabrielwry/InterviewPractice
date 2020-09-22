""""
Given an 2D board, count how many battleships are in it. The battleships are represented with 'X's, empty slots are represented with '.'s. You may assume the following rules:
You receive a valid board, made of only battleships or empty slots.
Battleships can only be placed horizontally or vertically. In other words, they can only be made of the shape 1xN (1 row, N columns) or Nx1 (N rows, 1 column), where N can be of any size.
At least one horizontal or vertical cell separates between two battleships - there are no adjacent battleships.
Example:
X..X
...X
...X
In the above board there are 2 battleships.
Invalid Example:
...X
XXXX
...X
This is an invalid board that you will not receive - as battleships will always have a cell separating between them.
Follow up:
Could you do it in one-pass, using only O(1) extra memory and without modifying the value of the board?
"""


class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        '''
        brainstorm?
        - keep counter
        - go through board. if see X, up counter and change this whole BS to '.''s lol

        We say that board has M rows, N columns
        Runtime: O(MN)
        Space: O(1)

        This solution has the characteristics of the follow-up:
            - one-pass
            - using only O(1) extra memory
            - does not modify the value of the board

        '''

        counter = 0
        M, N = len(board), len(board[0])

        for row in range(M):
            for col in range(N):
                # check if we've already counted?
                if board[row][col] == 'X':
                    '''
                    - case1: left is inbounds and an 'X' -> do nothing
                        - if this condition is true, then we don't need to look top b/c we know top won't be an 'X'
                    - case2: top is inbounds and an 'X' -> do nothing
                        - if this condition is true, then we don't need to look left, b/c we know left won't be an 'X'
                    - if case1 or case2 don't hit, then we for sure are not a continuation of any battleships. 
                        - We count it, b/c this is the beginning of a bship. 
                    '''
                    partOfLeftBattleShip = col - 1 != -1 and board[row][col - 1] == 'X'
                    partOfTopBattleShip = row - 1 != -1 and board[row - 1][col] == 'X'
                    if not partOfLeftBattleShip and not partOfTopBattleShip: counter += 1
        return counter
