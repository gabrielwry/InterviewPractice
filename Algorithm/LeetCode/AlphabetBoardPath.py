"""
On an alphabet board, we start at position (0, 0), corresponding to character board[0][0].

Here, board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"], as shown in the diagram below.



We may make the following moves:

'U' moves our position up one row, if the position exists on the board;
'D' moves our position down one row, if the position exists on the board;
'L' moves our position left one column, if the position exists on the board;
'R' moves our position right one column, if the position exists on the board;
'!' adds the character board[r][c] at our current position (r, c) to the answer.
(Here, the only positions that exist on the board are positions with letters on them.)

Return a sequence of moves that makes our answer equal to target in the minimum number of moves.  You may return any path that does so.



Example 1:

Input: target = "leet"
Output: "DDR!UURRR!!DDD!"
Example 2:

Input: target = "code"
Output: "RR!DDRR!UUL!R!"


Constraints:

1 <= target.length <= 100
target consists only of English lowercase letters.
"""


class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        step = ""

        def get_index(c):
            r = (ord(c) - ord("a")) // 5
            c = (ord(c) - ord("a")) % 5
            return (r, c)

        def get_step(start, end):
            if start == end:
                return "!"
            step = ""
            end_with_z = False
            if start == (5, 0):
                step = "U"
                start = (4, 0)
            if end == (5, 0):
                end = (4, 0)
                end_with_z = True
            r_diff = end[0] - start[0]
            c_diff = end[1] - start[1]
            if r_diff < 0:
                step += abs(r_diff) * "U"
            if r_diff > 0:
                step += abs(r_diff) * "D"
            if c_diff < 0:
                step += abs(c_diff) * "L"
            if c_diff > 0:
                step += abs(c_diff) * "R"
            if end_with_z:
                step += "D"
            step += "!"
            print(start, end, step)
            return step

        last_index = (0, 0)
        for c in target:
            c_index = get_index(c)
            print(c, last_index, c_index)
            step += get_step(last_index, c_index)
            last_index = c_index
        return step
