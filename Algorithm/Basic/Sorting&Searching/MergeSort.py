arr_ = [2,34,52,1,7,82,1234,8472,22,245,138]


def merge(arr_,lo,mid,hi):
	arr_1 = arr_[lo:mid+1]
	arr_2 = arr_[mid+1:hi+1]
	result = [None]*(hi-lo)
	i = 0
	j = 0
	n = 0
	while i<len(arr_1) and j<len(arr_2):
		if arr_1[i] < arr_2[j]:
			result[n] = arr_1[i]
			n += 1
			i += 1
		elif arr_1[i] > arr_2[j]:
			result[n] = arr_2[j]
			n += 1
			j += 1
		elif arr_1[i] == arr_2[j]:
			result[n] = arr_1[i]
			result[n+1] = arr_2[j]
			n += 2
			i += 1
			j += 1
	if i == len(arr_1): # copy the rest of arr_2
		result[n::] = arr_2[j::]
	if j == len(arr_2):
		result[n::] = arr_1[i::]
	arr_[lo:hi+1] = result



def mergeSort(arr_,lo,hi):
	if lo < hi:
		mid = (lo+hi-1)//2
		mergeSort(arr_,lo,mid)
		mergeSort(arr_,mid+1,hi)
		merge(arr_,lo,mid,hi)
mergeSort(arr_,0,len(arr_)-1)
print(arr_)