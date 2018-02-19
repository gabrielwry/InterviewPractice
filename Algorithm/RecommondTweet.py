# Complete the function below.
# followGraph_edges is a list of tuples (userId, userId)
# likeGraph_edges is also a list of tuples (userId, tweetId)

def getRecommendedTweets(followGraph_edges, likeGraph_edges, targetUser, minLikeThreshold):
    result = []
    follow =[]
    liked_by_follow = {}
    for follow_edge in followGraph_edges:
        if follow_edge[0] == targetUser:
            follow.append(follow_edge[1])
    for like_edge in likeGraph_edges:
        user = like_edge[0]
        tweet = like_edge[1]
        if tweet not in result:
            if user in follow:
                if tweet in liked_by_follow:
                    liked_by_follow[tweet] += 1
                    if liked_by_follow[tweet] == minLikeThreshold:
                        result.append(tweet)
                else:
                    liked_by_follow[tweet] = 1
    return result 