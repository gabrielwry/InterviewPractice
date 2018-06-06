arr_ = [2,34,52,1,7,82,1234,8472,22,245,138]
def insertionSort(arr):
	for i in range(1,len(arr)):
		tmp = arr[i]
		j = i - 1
		while j >= 0 and tmp < arr[j]:
			arr[j+1] = arr[j]
			j-=1
		arr[j+1] = tmp
	return arr
print(insertionSort(arr_))