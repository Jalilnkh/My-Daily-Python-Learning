# Codes for bubble up and bubble down

def parent(index):
    return (index -1 ) // 2
def swap(H, i, j):
    temp = H[i]
    H[i] = H[j]
    H[j] = temp
    return H

def heap_bubble_up(H, index):
    while index > 0 and H[index] < H[parent(index)]:
        parent_index = parent(index)
        swap(H, index, parent_index)
        index = parent_index

    return H

def left_child(index:int)->int:
    return 2 * index + 1

def right_child(index:int)->int:
    return 2 * index + 2

def heap_bubble_down(H:list, index:int, heap_size:int)->list:
    smallest_child_index = index
    left_child_index = left_child(index)
    right_child_index = right_child(index)

    # check if the left child exist and is smaller than the current smallest
    if left_child_index < heap_size and H[smallest_child_index] > H[left_child_index]:
        smallest_child_index = left_child_index
    # check if the right child exist and is smaller than the current smallest
    if right_child_index < heap_size and H[smallest_child_index] > H[right_child_index]:
        smallest_child_index = right_child_index
    
    # if the smallest is not the current index, swap and continue dashing down
    if smallest_child_index != index:
        swap(H, index, smallest_child_index)
        heap_bubble_down(H, smallest_child_index, heap_size)

    return H

def heap_insert(heap_arr, new_value, index):
    heap_arr.append(new_value)  # Add the new value at the end
    new_heap_arr = heap_bubble_up(heap_arr, index)
    return new_heap_arr

def heap_delete(heap_arr, del_index, heap_size):
    heap_arr = swap(heap_arr, del_index, heap_size - 1)  # Swap with the last element
    heap_arr.pop()  # Remove the last element
    if heap_arr[del_index] > heap_arr[parent(del_index)]:
        heap_bubble_down(heap_arr, del_index, len(heap_arr))
    if heap_arr[del_index] < heap_arr[parent(del_index)]:
        heap_bubble_up(heap_arr, del_index)
    return heap_arr

def array_to_heap(arr):
    arr_size = len(arr)
    for i in range(arr_size//2, 0, -1):
        arr = heap_bubble_down(arr, i, arr_size)
    return arr