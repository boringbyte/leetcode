from collections import defaultdict, deque

from leetcode.utils import ListNode


def remove_nth_node_from_end_of_list(head: ListNode | None, n: int) -> ListNode | None:
    # https://leetcode.com/problems/remove-nth-node-from-end-of-list
    # https://leetcode.com/problems/remove-nth-node-from-end-of-list/solutions/4813340/beat-10000-full-explanation-with-picture-sg8y/
    dummy = ListNode(0)
    dummy.next = head

    first = second = dummy

    for _ in range(n + 1):
        first = first.next

    while first:
        first = first.next
        second = second.next

    second.next = second.next.next

    return dummy.next


def diagonal_traversal_ii(nums):
    # https://leetcode.com/problems/diagonal-traverse-ii
    diagonal_map = defaultdict(list)

    for i in range(len(nums)):
        for j in range(len(nums[i])):
            diagonal_map[i + j].append(nums[i][j])

    result = []

    for i in diagonal_map.keys():
        result.extend(diagonal_map[i][::-1])

    return result


def koko_eating_bananas(piles: h):
    # https://leetcode.com/problems/koko-eating-bananas
    pass


def all_nodes_distance_k_in_binary_tree(root, target, k):
    # https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree

    def build_graph_from_binary_tree(root):
        graph = defaultdict(list)

        def dfs(child, parent):
            if parent:
                graph[parent].append(child)
                graph[child].append(parent)
            if child.left:
                dfs(child.left, child)
            if child.right:
                dfs(child.right, child)
        dfs(root, None)
        return graph

    graph = build_graph_from_binary_tree(root)
    visited = set()
    result = []

    def add_nodes_from_target_node_using_dfs(node, distance):
        if distance == k:
            result.append(node.val)
        else:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    add_nodes_from_target_node_using_dfs(neighbor, distance + 1)

    add_nodes_from_target_node_using_dfs(root, 0)
    return result


def shortest_bridge(grid):
    # https://leetcode.com/problems/shortest-bridge
    n = len(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = set()
    queue = deque()

    def dfs(x, y):
        visited.add((x, y))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited:
                if grid[nx][ny] == 0:
                    visited.add((nx, ny))
                    queue.append((nx, ny, 1))
                else:
                    dfs(nx, ny)

    found = False
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j)
                found = True
                break
        if found:
            break

    while queue:
        x, y, distance = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited:
                if grid[nx][ny] == 1:
                    return distance
                visited.add((nx, ny))
                queue.append((nx, ny, distance + 1))
    return 0