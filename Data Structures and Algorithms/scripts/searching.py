# Here is the code of binary search:

def bineary_search(lst, elt, left, right):
    if left > right:
        return None
    else:
        mid = (left + right)//2
        if lst[mid] == elt:
            return mid
        elif lst[mid] < elt:
            return bineary_search(lst, elt, mid+1, right)
        else:
            return bineary_search(lst, elt, left, mid-1)

