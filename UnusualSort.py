# Complete the function below.


def unusual_sort(numbers):
    _len = len(numbers)
    numbers = sorted(numbers)
    for i in range(0,_len/2):
        tmp = numbers.pop(-1)
        numbers.insert(i*2+1,tmp)
    print numbers


unusual_sort([1,3,4,2,5,2,3,6,8,9,10,2,5,4])