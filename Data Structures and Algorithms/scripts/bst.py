class Node():
    def __init__(val):
        self.left = None
        self.right = None
        self.val = val

class BST:
    def __init__():
        self.root = None
    
    def insert(val):
        if self.root is None:
            self.root = Node(val)
        else:
            return self._insert_recursive(self.root, val)
    
    def _insert_recursive(current, val):
        if current.val > val:
            if current.left is None:
                current.left = Node(val)
            else:
                return self._insert_recursive(current.left, val)
        else:
            if current.right is None:
                current.right = Node(val)
            else:
                return self._insert_recursive(current.right, val)

    def find(val):
        self._find_recursive(self.root, val)

    def _find_recursive(self, current, val):
        if current is None:
            return False
        if current.val == val:
            return True
        elif current.val > val:
            return self._find_recursive(current.left, val)
        else:
            return self._find_recursive(current.right, val)
    
    def in_order_traversal(self):
        results = []
        self._in_order(self.root, results)
        return results
    
    self._in_order(self, current, results):
        if current is None:
            return
        return self._in_order(current.left, results)
        results.append(current.val)
        return self._in_order(current.right, results)



            