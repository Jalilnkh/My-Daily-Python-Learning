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