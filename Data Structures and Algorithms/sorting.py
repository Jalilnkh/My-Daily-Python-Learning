

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