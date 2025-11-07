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


def minheap_bubble_down(H:list, index:int, heap_size:int)->list:
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
        minheap_bubble_down(H, smallest_child_index, heap_size)

    return H

def maxheap_bubble_down(H:list, index:int, heap_size:int)->list:
    largest_child_index = index
    left_child_index = left_child(index)
    right_child_index = right_child(index)

    # Check if left child exists and is LARGER than current largest
    if left_child_index < heap_size and H[largest_child_index] < H[left_child_index]:
        largest_child_index = left_child_index
    
    # Check if right child exists and is LARGER than current largest  
    if right_child_index < heap_size and H[largest_child_index] < H[right_child_index]:
        largest_child_index = right_child_index
    
    if largest_child_index != index:
        swap(H, index, largest_child_index)
        maxheap_bubble_down(H, largest_child_index, heap_size)

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

def array_to_minheap(arr):
    arr_size = len(arr)
    for i in range(arr_size//2-1, -1, -1):
        arr = minheap_bubble_down(arr, i, arr_size)
    return arr

def array_to_maxheap(arr):
    arr_size = len(arr)
    for i in range(arr_size//2-1, -1, -1):
        arr = maxheap_bubble_down(arr, i, arr_size)
    return arr

def heap_sort(arr_to_sort):
    n = len(arr_to_sort)
    heap_max_array = array_to_maxheap(arr_to_sort)
    for i in range(n-1, 0, -1):
        swap(heap_max_array, 0, i)
        maxheap_bubble_down(heap_max_array, 0, i)
    return heap_max_array

# HashTable creation
class HashTableTest:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        """Simple hash function using module"""
        if isinstance(key, str):
            return sum(ord(char) for char in key) % self.size
        return hash(key) % self.size

    def insert(self, key, value):
        """Insert key-value pair into hash table"""
        index = self._hash(key)
        bucket = self.table[index]

        # Check if hey already exists, update value
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                print(f"Updated: {key} -> {value} at index {index}")
                return
        
        # Key does not exist, add new entry
        bucket.append((key, value))
        print(f"Inserted: {key} -> {value} at index {index}")

    def delete(self, key):
        """Delete key-value pair from hash table"""
        index = self._hash(key)
        bucket = self.table[index]

        for i, (k, _) in enumerate(bucket):
            if k == key:
                deleted_value = bucket.pop(i)
                print(f"Deleted: {key} -> {deleted_value[1]} from index {index}")
                return deleted_value[1]
        
        print(f"Key '{key}' not found")
        return None

    def get(self, key):
        """Retrieve value by key"""
        index = self._hash(key)
        bucket = self.table[index]

        for k, v in bucket:
            if k == key:
                return v
        return None

    def display(self):
        """Display the entire hash table"""
        print("\n--- Hash Table Contents ---")
        for i, bucket in enumerate(self.table):
            if bucket:
                print(f"Index {i}: {bucket}")
            else:
                print(f"Index {i}: []")
        print("----------------------------\n")
