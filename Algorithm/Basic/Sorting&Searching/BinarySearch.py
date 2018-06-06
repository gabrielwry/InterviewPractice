# Given a sorted array arr[] of n elements, write a function to search a given element x in arr[].
arr = [1,2,3,4,5,6,7,8,91,100]
def search(arr_,element,low,high):
	mid = arr_[int((low+high)/2)]
	if low+1 < high:
		if mid < element:
			low = int((low+high)/2)
			return search(arr_,element,low,high)
		if mid > element:
			high = int((low+high)/2)
			return search(arr_,element,low,high)
		if mid == element:
			return int((low+high)/2)
	else:
		print('NOT FOUND')
		return
print(search(arr,91,0,len(arr)))