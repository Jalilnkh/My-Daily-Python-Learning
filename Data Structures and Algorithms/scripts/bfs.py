from collections import deque
from typing import Dict, List, Optional, Hashable, Iterable, Tuple

def bfs_shortest_path(
    start: Hashable,
    goal: Hashable,
    graph: Dict[Hashable, Iterable[Hashable]]
) -> Tuple[Optional[List[Hashable]], Dict[Hashable, int]]:
    """
    Returns (path, distance) where:
      - path is the shortest path from start to goal (list of nodes) or None if unreachable
      - distance maps each visited node to its distance from start
    """
    visited = set([start])
    parent: Dict[Hashable, Optional[Hashable]] = {start: None}
    dist: Dict[Hashable, int] = {start: 0}
    q = deque([start])

    while q:
        u = q.popleft()
        if u == goal:
            # reconstruct
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path)), dist

        for v in graph.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                dist[v] = dist[u] + 1
                q.append(v)

    return None, dist  # goal not reachable


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