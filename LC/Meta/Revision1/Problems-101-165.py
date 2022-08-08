import heapq
import collections
import random
from functools import lru_cache
from LC.LCMetaPractice import ListNode, TreeNode, RandomPointerNode


def longest_common_prefix1(strs):
    if not strs:
        return ''
    prefix, n = [], len(strs)
    for chars in zip(*strs):
        if len(set(chars)) == 1:
            prefix.append(chars[0])
        else:
            break
    return ''.join(prefix)


def longest_common_prefix2(strs):
    if not strs:
        return ''
    shortest = min(strs, key=len)
    for i, char in enumerate(shortest):
        for other in strs:
            if other[i] != char:
                return shortest[:i]
    return shortest


def meeting_rooms_2(intervals):
    n, heap = len(intervals), []
    if n <= 1:
        return n
    for interval in sorted(intervals):
        if heap and interval[0] >= heap[0]:
            heapq.heappushpop(heap, interval[1])
        else:
            heapq.heappush(heap, interval[1])
    return len(heap)


def validate_binary_search_tree(root):

    def dfs(node, low=float('-inf'), high=float('inf')):
        if not root:
            return True
        if not low < node.val < high:
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

    return dfs(root)


def diagonal_traversal(matrix):
    m, n, result = len(matrix), len(matrix[0]), []
    diagonal_map = collections.defaultdict(list)

    for i in range(m):
        for j in range(n):
            diagonal_map[i + j].append(matrix[i][j])

    for key in sorted(diagonal_map):
        if key % 2 == 0:
            result.extend(diagonal_map[key][::-1])
        else:
            result.extend(diagonal_map[key])
    return result
    # itertools.chain(*[v if k % 2 else v[::-1] for k, v in d.items()])


def check_completeness_of_a_binary_tree(root):
    queue = collections.deque([root])
    prev_node = root
    while queue:
        current = queue.popleft()
        if current:
            if not prev_node:
                return False
            queue.append(current.left)
            queue.append(current.right)
        prev_node = current
    return True


def nested_list_weight_sum(nested_list):
    def dfs(current_list, depth):
        total = 0
        for value in current_list:
            if isinstance(value, int):
                total += (value * depth)
            else:
                total += dfs(value, depth + 1)
        return total
    return dfs(nested_list, 1)


def permutations(nums):
    result, n = [], len(nums)
    visited = [0] * n

    def backtrack(sofar):
        if len(sofar) == n:
            result.append(sofar[:])
        else:
            for i in range(n):
                if visited[i] != 1:
                    chosen, visited[i] = nums[i], 1
                    backtrack(sofar + [chosen])
                    visited[i] = 0
    backtrack(sofar=[])
    return result


def course_schedule(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    for curr, prev in prerequisites:
        graph[prev].append(curr)
        in_degree[curr] += 1

    queue = collections.deque(v for v in range(num_courses) if in_degree[v] == 0)
    n = len(queue)
    while queue and n != num_courses:
        current_course = queue.popleft()
        for next_course in graph[current_course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                n += 1
                queue.append(next_course)
    return n == num_courses


