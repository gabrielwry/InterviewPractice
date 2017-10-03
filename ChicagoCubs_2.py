# Complete the function below.

def canBeCaught(ballTrajectoryX, ballTrajectoryY, outfieldPlayerPositions, outfieldPlayerReaches):
    print ballTrajectoryX,ballTrajectoryY,outfieldPlayerPositions,outfieldPlayerReaches
    if ballTrajectoryY != 0 and ballTrajectoryX != 0:
        slope = float(ballTrajectoryX)/float(ballTrajectoryY)
    if ballTrajectoryY == 0 and ballTrajectoryX != 0:
        slope = float('inf')
    if ballTrajectoryY != 0 and ballTrajectoryX == 0:
        slope = 0
    print len(outfieldPlayerReaches)
    for i in range(0,len(outfieldPlayerPositions)):
        print outfieldPlayerPositions[i]
        x_pos = outfieldPlayerPositions[i][0]
        y_pos = outfieldPlayerPositions[i][1]
        x_intersect = x_pos * slope
        if slope != 0 and slope != float('inf'):
            x_intersect = x_pos * slope
            y_intersect = y_pos / slope
        if slope == 0:
            x_intersect = float('inf')
            y_intersect = 0
        if slope == float('inf'):
            x_intersect = 0
            y_intersect = float('inf')
        print x_intersect,y_intersect
        if abs(x_intersect - y_pos) <= outfieldPlayerReaches[i]:
            if int(abs(x_intersect - y_pos)) - abs(x_intersect - y_pos) == 0:
                return True
        if abs(y_intersect - x_pos) <= outfieldPlayerReaches[i]:
            if int(abs(y_intersect - x_pos)) - abs(y_intersect - x_pos) == 0:
                return True

    return False

print canBeCaught(3,4,[[2,2],[6,9]],[2,1])