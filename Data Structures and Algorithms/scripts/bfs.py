from collections import deque


def bfs_order(start: Hashable, graph: Graph) -> List[Hashable]:
    """Return nodes in BFS visitation order from 'start'."""
    visited = set([start])
    q = deque([start])
    order = []

    while q:
        node = q.popleft()
        order.append(node)
        for nbr in graph.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                q.append(nbr)
    return order