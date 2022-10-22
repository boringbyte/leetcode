import collections


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def binary_tree_maximum_path_sum(root):
    result = [0]

    def dfs(node):
        if node is None:
            return 0
        l, r = dfs(node.left), dfs(node.right)
        result[0] = max(result[0], l + r + node.val)
        return max(0, l + node.val, r + node.val)

    dfs(root)
    return result[0]


def range_sum_binary_search_tree(root, low, high):
    result = [0]

    def dfs(node):
        if node:
            if low <= node.val <= high:
                result[0] += node.val
            if low < node.val:
                dfs(node.left)
            if high > node.val:
                dfs(node.right)

    dfs(root)
    return result[0]


def binary_tree_right_side_view1(root):
    if not root:
        return []
    result, queue = [], collections.deque([root])
    while queue:
        for i in range(len(queue)):
            current = queue.popleft()
            if i == 0:
                result.append(current.val)
            if current.right:
                queue.append(current.right)
            if current.left:
                queue.append(current.left)
    return result


def binary_tree_right_side_view2(root):
    if not root:
        return []
    result, visited_level = [], set()

    def dfs(node, level):
        if not node:
            return
        if level not in visited_level:
            result.append(node.val)
        dfs(node.right, level + 1)
        dfs(node.left, level + 1)

    dfs(root, 0)
    return result


def diameter_of_binary_tree(root):
    result = [0]

    def dfs(node):
        if not node:
            return 0
        l, r = dfs(node.left), dfs(node.right)
        result[0] = max(result[0], l + r)
        return max(l, r) + 1

    dfs(root)
    return result[0]


def lowest_common_ancestor_of_binary_tree1(root, p, q):
    if root is None or root == p or root == q:
        return root
    l = lowest_common_ancestor_of_binary_tree1(root.left, p, q)
    r = lowest_common_ancestor_of_binary_tree1(root.right, p, q)
    if l and r:
        return root
    return l or r


def lowest_common_ancestor_of_binary_tree2(root, p, q):
    stack, parent_dict, ancestors = [root], {root: None}, set()
    while p not in parent_dict or q in parent_dict:
        current = stack.pop()
        if current.left:
            stack.append(current.left)
            parent_dict[current.left] = current
        if current.right:
            stack.append(current.right)
            parent_dict[current.right] = current

    while p:
        ancestors.add(p)
        p = parent_dict[p]

    while q not in ancestors:
        q = parent_dict[q]
    return q


class Codec:
    i = 0

    def serialize(self, root):
        values = []

        def dfs(node):
            if node is None:
                values.append('#')
            else:
                values.append(str(node.val))
                dfs(node.left)
                dfs(node.right)

        dfs(root)
        return ' '.join(values)

    def deserialize(self, data):
        values = data.split()

        def dfs():
            if self.i == len(values):
                return None
            value = values[self.i]
            self.i += 1
            if value == '#':
                return None
            node = TreeNode(int(value))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()


def build_graph_from_binary_tree(root):
    graph = collections.defaultdict(list)

    def dfs(child, parent):
        if parent:
            graph[child].append(parent)
        if child.left:
            graph[child].append(child.left)
            dfs(child.left, child)
        if child.right:
            graph[child].append(child.right)
            dfs(child.right, child)

    dfs(root, None)
    return graph


def all_nodes_distance_k_binary_tree1(root, target, k):
    visited, result = {target}, []
    graph = build_graph_from_binary_tree(root)

    def dfs(node, distance):
        if distance == 0:
            result.append(node.val)
        else:
            visited.add(node)
            for neighbour in graph[node]:
                if neighbour not in visited:
                    dfs(neighbour, distance - 1)

    dfs(target, k)
    return result


def all_nodes_distance_k_binary_tree2(root, target, k):
    visited, result, queue = set(), [], collections.deque([(target, 0)])
    graph = build_graph_from_binary_tree(root)
    while queue:
        current, distance = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if k == distance:
            result.append(current.val)
        elif distance < k:
            for child in graph[current]:
                queue.append((child, distance + 1))
    return result


def balance_a_binary_search_tree(root):
    values = []

    def dfs(node):
        if node:
            dfs(node.left)
            values.append(node.val)
            dfs(node.right)

    dfs(root)

    def build_binary_search_tree(l, r):
        if l > r:
            return None
        m = (l + r) // 2
        node = TreeNode(values[m])
        node.left = build_binary_search_tree(l, m - 1)
        node.right = build_binary_search_tree(m + 1, r)
        return node

    return build_binary_search_tree(0, len(values) - 1)
