"""There is a brick wall in front of you. The wall is rectangular and has several rows of bricks.
The bricks have the same height but different width. You want to draw a vertical line from the top to the bottom and cross the least bricks.
The brick wall is represented by a list of rows. Each row is a list of integers representing the width of each brick in this row from left to right.
If your line go through the edge of a brick, then the brick is not considered as crossed. You need to find out how to draw the line to cross the least bricks and return the number of crossed bricks.
You cannot draw a line just along one of the two vertical edges of the wall, in which case the line will obviously cross no bricks."""
# Correct algorithm, bad implementation
class Solution(object):
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        sums_ = {}
        sum_wall = []
        count = 0
        for row in wall:
            row_sum = [row[0]]
            if row[0] not in sums_:
                sums_[row[0]] = 1
            else:
                sums_[row[0]] += 1
            if len(row) <= 2:
                if len(row) == 1:
                    count += 1

            for cell in row[1:-1]:
                print cell
                if cell + row_sum[-1] not in sums_:
                    sums_[cell + row_sum[-1]] = 1
                else:
                    sums_[cell + row_sum[-1]] += 1
                row_sum.append(cell + row_sum[-1])

            sum_wall.append(row_sum)
        # print sums_
        cut = max(sums_.keys(), key=lambda x: sums_[x])

        for i in range(0, len(sum_wall)):
            # print sum_wall[i],cut
            if cut not in sum_wall[i]:
                # print wall[i]
                if len(wall[i]) != 1:
                    count += 1
        return count


