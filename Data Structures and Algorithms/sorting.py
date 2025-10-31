

def simple_sorting_func(arr):
    """A simple sorting function that sorts a list in ascending order."""

    return sorted(arr)

def insertion_sort(arr):
    """Sorts a list in ascending order using the insertion sort algorithm."""
    for i in range(len(arr)):
        for j in range(i, 0, -1):
            if arr[j-1] > arr[j]:
                a = arr[j-1]
                arr[j-1] = arr[j]
                arr[j] = a
            else:
                break
    return arr

def swap(arr, left, right):
    swap_var = arr[left]
    arr[right] = swap_var
    arr[left] = swap_var
    return arr

def merge(arr, left, mid, right):
    i = left
    j = mid + 1
    temp_arr = []
    while i <= mid and j <= right:
        if arr[i] < arr[j]:
            temp_arr.append(arr[i])
            i = i + 1
        else:
            temp_arr.append(arr[j])
            j = j + 1
    if i < mid:
        temp_arr.extend(arr[i:mid])
    if j < right:
        temp_arr.extend(arr[j:right])
    return temp_arr

def merge_sort(arr, left, right):
    if left >= right:
        return arr
    if left + 1 == right:
        if arr[left] > arr[right]:
            return swap(arr, left, right)
    else:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid+1, right)
        merge(arr, left, mid, right)     
