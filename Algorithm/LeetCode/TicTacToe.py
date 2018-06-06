def checkTicTacToe(matrix):
    #  If any player want to win the game
    #   one of the cell in their winning strategy will be in the 
    #   diagonal cells. So we only need to check the diagonal cells for winning
    row_ = len(matrix)
    col_ = len(matrix)
    has_0 = False
    for i in range(row_):
        if len(set(matrix[i])) == 1:
            tmp = set(matrix[i]).pop()
            if tmp != 0:
                return tmp
            
    for j in range(col_):

        if len(set(row[j] for row in matrix)) == 1:

            tmp = set(row[j] for row in matrix).pop()

            if tmp != 0:
                return tmp

        has_0 = 0 in set(row[j] for row in matrix)
            
    if len(set([matrix[i][i] for i in range(row_)])) == 1:
        # print(set([matrix[i][i] for i in range(row_)]))
        tmp = set([matrix[i][i] for i in range(row_)]).pop()
        if tmp != 0:
            return tmp
        
    if len(set(row[-i-1] for i, row in enumerate(matrix)))== 1:

        tmp = set(row[-i-1] for i, row in enumerate(matrix)).pop()
        if tmp != 0:
            return tmp
        
    if  has_0:
        return 0        
    else:
        return 2
            
print(checkTicTacToe(
    [[0, 0, 0], 
    [1, 1, 1], 
    [-1, 0, 1]]))