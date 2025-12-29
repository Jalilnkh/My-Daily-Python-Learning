

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
    arr[left], arr[right] = arr[right], arr[left]
    return arr

def merge(arr, left, mid, right):
    i = left
    j = mid + 1
    temp_arr = []
    while i <= mid and j <= right:
        if arr[i] < arr[j]:
            temp_arr.append(arr[i])
            i += 1
        else:
            temp_arr.append(arr[j])
            j += 1
    while i <= mid:
        temp_arr.append(arr[i])
        i += 1
    while j <= right:
        temp_arr.append(arr[j])
        j += 1
    # Copy temp_arr back to arr
    for idx, val in enumerate(temp_arr):
        arr[left + idx] = val

def merge_sort(arr, left, right):
    if left >= right:
        return
    mid = (left + right) // 2
    merge_sort(arr, left, mid)
    merge_sort(arr, mid + 1, right)
    merge(arr, left, mid, right)

def lomuto_partition(arr, left, right):
    """
        Partition using the last element as pivot (Lomuto)
    """
    pivot = arr[right]
    i = left
    for j in range(left, right):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[right] = arr[right], arr[i]
    return i

def quicksort_lomuto(arr, left=0, right=None):
    """Quick sort algorithm using lomuto partition scheme"""
    if right is None:
        right = len(arr) - 1
    if left < right:
        p = lomuto_partition(arr, left, right)
        quicksort_lomuto(arr, left, p-1)
        quicksort_lomuto(arr, p+1, right)
    return arr
    