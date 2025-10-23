

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