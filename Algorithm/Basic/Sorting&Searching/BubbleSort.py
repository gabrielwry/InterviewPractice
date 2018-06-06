arr_ = [2,34,52,1,7,82,1234,8472,22,245,138]
def bubbleSort(arr):
	sorted_ = False
	while not sorted_:
		sorted_ = True
		for i in range(1,len(arr)):
			tmp = arr[i-1]
			if arr[i-1] > arr[i]:
				arr[i-1] = arr[i]
				arr[i] = tmp
				sorted_ = False
	return arr
print(bubbleSort(arr_))