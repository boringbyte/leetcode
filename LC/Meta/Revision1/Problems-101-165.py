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


def reverse_linked_list(head):
    prev = None
    while head:
        current = head
        head = head.next
        current.next = prev
        prev = current
    return prev


def palindrome_linked_list(head):
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    slow = reverse_linked_list(slow)
    while slow:
        if slow.val != head.val:
            return False
        slow = slow.next
        head = head.next
    return True


def strobogrammatic_number(num):
    number_map = {('0', '0'), ('1', '1'), ('6', '9'), ('8', '8'), ('9', '6')}
    i, j = 0, len(num) - 1
    while i <= j:
        if (num[i], num[j]) not in number_map:
            return False
        i, j = i + 1, j - 1
    return True


def first_missing_positive(nums):
    pass


def construct_binary_tree_from_string(s):
    if not s or len(s) == 0:
        return None

    def dfs(s, i):
        start = i
        if s[start] == '-':
            i += 1
        while i < len(s) and s[i].isdigit():
            i += 1
        node = TreeNode(int(s[start: i]))

        if i < len(s) and s[i] == '(':
            i += 1
            node.left, i == dfs(s, i)
            i += 1

        if i < len(s) and s[i] == '(':
            i += 1
            node.right, i == dfs(s, i)
            i += 1

        return node, i

    root, idx = dfs(s, 0)
    return root


def generate_parentheses(n):
    result = []

    def backtrack(sofar, left, right):
        if len(sofar) == 2 * n:
            result.append(sofar)
        else:
            if left < n:
                backtrack(sofar + '(', left + 1, right)
            if right < left:
                backtrack(sofar + ')', left, right + 1)
    backtrack('', 0, 0)
    return result


def median_of_two_sorted_arrays(nums1, nums2):
    pass


def missing_ranges(nums, lower, upper):
    result, n, previous = [], len(nums), lower - 1

    def get_range(left, right):
        if left == right:
            return f'{left}'
        return f'{left}->{right}'

    if not nums:
        gap = get_range(lower, upper)
        result.append(gap)
        return result

    for num in nums:
        if previous + 1 != num:
            gap = get_range(previous + 1, num - 1)
            result.append(gap)
        previous = num

    if nums[-1] < upper:
        gap = get_range(nums[-1] + 1, upper)
        result.append(gap)

    return result


class NumMatrix:
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.total = [[0] * (n + 1) for _ in range(m + 1)]
        for r in range(1, m + 1):
            for c in range(1, n + 1):
                self.total[r][c] = self[r - 1][c] + self.total[r][c - 1] - self.total[r - 1][c - 1] + matrix[r - 1][c - 1]

    def sum_region(self, r1, c1, r2, c2):
        r1, c1, r2, c2 = r1 + 1, c1 + 1, r2 + 1, c2 + 1
        return self.total[r2][c2] - self[r1 - 1][c1] - self.total[r1][c1 - 1] + self.total[r1 - 1][c1 - 1]


def populating_next_right_pointer_in_each_node1(root):
    if root is None:
        return root

    queue = collections.deque([root])
    while queue:
        size = k = len(queue)
        while k > 0:
            current = queue.popleft()
            if k == size:
                current.next = None
            else:
                current.next = prev
            prev = current

            if current.right:
                queue.append(current.right)
            if current.left:
                queue.append(current.left)

            k -= 1
    return root


def populating_next_right_pointer_in_each_node2(root):
    if root is None:
        return root

    queue = collections.deque([root])
    while queue:
        right_node = None
        for _ in range(len(queue)):
            current = queue.popleft()
            current.next = right_node
            right_node = current

            if current.right:
                queue.append(current.right)
            if current.left:
                queue.append(current.left)

    return root


def reverse_integer(x):
    result, sign = 0, 1
    if x < 0:
        sign, x = -sign, -x

    while x:
        result = result * 10 + x % 10
        x /= 10

    return 0 if result > pow(2, 31) else result * sign


def minimum_cost_for_tickets(days, costs):
    days_map, last_day = set(days), days[-1]
    dp = [0] * (last_day + 1)

    for day in range(1, last_day + 1):
        if day not in days_map:
            dp[day] = dp[day - 1]
        else:
            dp[day] = min(
                dp[max(0, day - 1)] + costs[0],  # per days value
                dp[max(0, day - 7)] + costs[1],  # per week value
                dp[max(0, day - 30)] + costs[2]  # per year value
            )
    return dp[-1]


def minimum_knight_moves():
    pass


def sum_root_to_leaf_numbers1(root):

    def dfs(node, value):
        if not node:
            return 0
        value = value * 10 + node.val
        if not node.left and not node.right:
            return value
        return dfs(node.left, value) + dfs(node.right, value)
    return dfs(root, 0)


def sum_root_to_leaf_numbers2(root):
    stack, result = [(root, 0)], 0
    while stack:
        current, value = stack.pop()
        value = value * 10 + current.val
        if not current.left and not current.right:
            result += value
        if current.right:
            stack.append((current.right, value))
        if current.left:
            stack.append((current.left, value))
    return result


