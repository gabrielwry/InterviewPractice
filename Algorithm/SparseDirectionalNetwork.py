def broadcast_delivery_time(origin_id, adj_matrix):
    # Enter your code here.
    total = len(adj_matrix)
    queue = [origin_id]
    cycle = []
    delivered = [0]*total
    time = [float('inf')]*total
    time[origin_id] = 0
    for each in queue:
        delivered[each] = 1
        row = adj_matrix[each]
        for _id in range(0,total):
            if _id != each and row[_id] is not None:
                if (_id,each) not in cycle:
                    cycle.append((each,_id))
                    queue.append(_id)
                    if row[_id] < time[_id]:
                        time[_id] = row[_id]
    if 0 in delivered:
        return None
    else:
        total_time = 0
        for each in time:
            total_time += each
        return total_time

print broadcast_delivery_time(0,[[None,None,122,None],[None,None,None,50],[341,None,None,205],[456,None,186,None]])