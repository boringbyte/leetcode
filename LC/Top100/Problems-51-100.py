import collections
import heapq
import random
from functools import lru_cache
from LC.LCMetaPractice import TreeNode


def maximum_product_subarray(nums):
    # https://leetcode.com/problems/maximum-product-subarray/discuss/48276/Python-solution-with-detailed-explanation
    result = big = small = nums[0]
    for num in nums[1:]:
        a, b, c = num, num * big, num * small
        big, small = max(a, b, c), min(a, b, c)
        result = max(result, big)
    return result


class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, val):
        min_val = self.get_min()
        if min_val is None or val < min_val:
            min_val = val
        self.stack.append((val, min_val))

    def pop(self):
        self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1][0]

    def get_min(self):
        if self.stack:
            return self.stack[-1][1]


def intersection_of_two_linked_lists1(headA, headB):
    # comments of
    # https://leetcode.com/problems/intersection-of-two-linked-lists/discuss/49798/Concise-python-code-with-comments
    if headA and headB:
        last = headA
        while last.next:
            last = last.next
        last.next = headB

        slow = fast = headA
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                break
        else:
            last.next = None  # As not modifying is a condition
            return

        slow = headA
        while slow != fast:
            slow = slow.next
            fast = fast.next
        last.next = None  # As not modifying is a condition
        return slow


def intersection_of_two_linked_lists2(headA, headB):
    # https://leetcode.com/problems/intersection-of-two-linked-lists/discuss/49798/Concise-python-code-with-comments
    if not headA or not headB:
        return
    pa, pb = headA, headB
    while pa is not pb:
        pa = headB if pa is None else pa.next
        pb = headA if pb is None else pb.next
    return pa


def majority_element(nums):
    result, count = None, 0
    for num in nums:
        if count == 0:
            result = num
        if result == num:
            count += 1
        else:
            count -= 1
    return result


def house_robber1(nums):
    # https://leetcode.com/problems/house-robber/discuss/1605797/C%2B%2BPython-4-Simple-Solutions-w-Explanation-or-Optimization-from-Brute-Force-to-DP
    n = len(nums)

    @lru_cache
    def recursive(i):
        if i >= n:
            return 0
        else:
            return max(recursive(i + 1), nums[i] + recursive(i + 2))
    return recursive(0)


def house_robber2(nums):
    n = len(nums)
    if n == 1:
        return nums[0]
    dp = nums[:]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
    return dp[-1]


def house_robber3(nums):
    prev2 = prev = curr = 0
    for num in nums:
        curr = max(prev, num + prev2)
        prev2, prev = prev, curr
    return curr


def reverse_linked_list1(head):
    prev = None
    while head:
        current = head
        head = head.next
        current.next = prev
        prev = current
    return prev


def reverse_linked_list2(head):
    # https://leetcode.com/problems/reverse-linked-list/discuss/58127/Python-Iterative-and-Recursive-Solution
    def recursive(node, prev=None):
        if not node:
            return prev
        next_node = node.next
        node.next = prev
        return recursive(next_node, node)
    return recursive(head)


def course_schedule(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    for current, previous in prerequisites:
        graph[previous].append(current)
        in_degree[current] += 1

    queue = collections.deque(course for course in range(num_courses) if in_degree[course] == 0)
    n = len(queue)

    while queue and n != num_courses:
        current = queue.popleft()
        for next_course in graph[current]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                n += 1
                queue.append(next_course)
    return n == num_courses


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                return False
        return node.is_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return False
        return True


def course_schedule_2(num_courses, prerequisites):
    # https://leetcode.com/problems/course-schedule-ii/discuss/1642354/C%2B%2BPython-Simple-Solutions-w-Explanation-or-Topological-Sort-using-BFS-and-DFS
    graph = [[] for _ in range(num_courses)]
    in_degree, result = [0] * num_courses, []

    for current, previous in prerequisites:
        graph[previous].append(current)
        in_degree[current] += 1

    queue = collections.deque(course for course in range(num_courses) if in_degree[course] == 0)

    while queue:
        current = queue.popleft()
        result.append(current)
        for next_course in graph[current]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return result if len(result) == num_courses else []


def kth_largest_element_in_an_array(nums, k):
    n, k = len(nums), len(nums) - k

    def swap(i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def partition(left, right, p_index):
        pivot = nums[p_index]
        swap(p_index, right)
        p_index = left
        for i in range(left, right):
            if nums[i] <= pivot:
                swap(i, p_index)
                p_index += 1
        swap(p_index, right)
        return p_index

    def quick_select(left, right):
        p_index = random.randint(left, right)
        p_index = partition(left, right, p_index)
        if k == p_index:
            return
        if k < p_index:
            quick_select(left, p_index - 1)
        if k > p_index:
            quick_select(p_index + 1, right)

    quick_select(0, n - 1)
    return nums[k]


def maximal_square1(matrix):
    # https://leetcode.com/problems/maximal-square/discuss/1632376/C%2B%2BPython-6-Simple-Solution-w-Explanation-or-Optimizations-from-Brute-Force-to-DP
    pass


def invert_binary_tree(root):
    if root:
        root.left, root.right = invert_binary_tree(root.right), invert_binary_tree(root.left)
        return root


def kth_smallest_element_in_a_bst(root, k):
    stack, current = [], root
    while True:
        if current:
            stack.append(current)
            current = current.left
        elif stack:
            current = stack.pop()
            k -= 1
            if k == 0:
                return current.val
            current = current.right
        else:
            break


def reverse_linked_list(head):
    prev = None
    while head:
        current = head
        head = head
        current.next = prev
        prev = current
    return prev


def palindrome_linked_list(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    list2, list1 = reverse_linked_list(slow), head
    while list2:
        if list2.val != list1.val:
            return False
        list2, list1 = list2.next, list1.next
    return True


def sliding_window_maximum(nums, k):
    pass


def search_a_2d_matrix_2(matrix, target):
    # https://leetcode.com/problems/search-a-2d-matrix-ii/discuss/1079154/Python-O(m-%2B-n)-solution-explained
    m = len(matrix)
    x, y = len(matrix[0]) - 1, 0
    while x > 0 and y < m:
        if matrix[y][x] > target:
            x -= 1
        elif matrix[y][x] < target:
            y += 1
        else:
            return True
    return False


def meeting_rooms_2():
    pass


def perfect_squares(n):
    # https://leetcode.com/problems/perfect-squares/discuss/275311/Python-DP-and-BFS
    # https://leetcode.com/problems/perfect-squares/discuss/71475/Short-Python-solution-using-BFS
    squares = [i * i for i in range(1, (n ** 0.5) + 1)]
    depth, queue, new_queue = 1, {n}, set()
    while queue:
        for current in queue:
            for square in squares:
                if current == square:
                    return depth
                if current < square:
                    break
                new_queue.add(current - square)
        queue, new_queue, depth = new_queue, set(), depth + 1


def longest_increasing_subsequence(nums):
    pass


def merge_two_binary_trees1(root1, root2):
    # https://leetcode.com/problems/merge-two-binary-trees/discuss/426243/PythonRecursive-Solution-Beats-100
    if not root1 or not root2:
        return root1 or root2
    else:
        node = TreeNode(root1.val + root2.val)
        node.left = merge_two_binary_trees1(root1.left, root2.left)
        node.right = merge_two_binary_trees1(root1.right, root2.right)
        return node


def merge_two_binary_trees2(root1, root2):
    # https://leetcode.com/problems/merge-two-binary-trees/discuss/104429/Python-BFS-Solution
    # https://leetcode.com/problems/merge-two-binary-trees/discuss/173640/Python-Simple-Iterative-using-ListStack
    # comments
    if not root1 or not root2:
        return root1 or root2
    stack = [(root1, root2)]
    while stack:
        node1, node2 = stack.pop()
        if not node2:
            continue
        node1.val += node2.val
        if not node1.left:
            node1.left = node2.left
        else:
            stack.append((node1.left, node2.left))
        if not node1.right:
            node1.right = node2.right
        else:
            stack.append((node1.right, node2.right))
    return root1


def task_scheduler1(tasks, n):
    # https://leetcode.com/problems/task-scheduler/discuss/104507/Python-Straightforward-with-Explanation
    task_counts = collections.Counter(tasks)
    most_frequent_task, most_frequent_task_count = task_counts.most_common(1)[0]
    tasks_with_max_frequency = sum(task_counts[key] == most_frequent_task_count for key in task_counts.keys())
    return max(len(tasks), (most_frequent_task_count - 1) * (n + 1) + tasks_with_max_frequency)


def task_scheduler2(tasks, n):
    # https://www.youtube.com/watch?v=s8p8ukTyA2I
    counts, result = collections.Counter(tasks), 0
    max_heap = [-c for c in counts.values()]
    heapq.heapify(max_heap)

    queue = collections.deque()  # pairs of [-cnt, idle_time]
    while max_heap or queue:
        result += 1
        if max_heap:
            cnt = 1 + heapq.heappop(max_heap)
            if cnt:
                queue.append([cnt, result + n])
        if queue and queue[0][1] == result:
            heapq.heappush(max_heap, queue.popleft()[0])
    return result


def palindromic_substrings(s):
    # https://www.youtube.com/watch?v=4RACzI5-du8
    # https://leetcode.com/problems/palindromic-substrings/discuss/105687/Python-Straightforward-with-Explanation-(Bonus-O(N)-solution)
    n, result = len(s), [0]

    def count_palindrome(s, l, r):
        while l >= 0 and r < n and s[l] == s[r]:
            result[0], l, r = result[0] + 1, l - 1, r + 1

    for i in range(n):
        count_palindrome(s, i, i)
        count_palindrome(s, i, i + 1)

    return result[0]


def daily_temperatures(temperatures):
    # https://leetcode.com/problems/daily-temperatures/discuss/1574808/C%2B%2BPython-3-Simple-Solutions-w-Explanation-Examples-and-Images-or-2-Monotonic-Stack-Approaches
    result, stack = [0] * len(temperatures), []
    for i, temperature in enumerate(temperatures):
        while stack and temperature > temperatures[stack[-1]]:
            last_index = stack[-1]
            result[last_index] = i - last_index
            stack.pop()
        stack.append(i)
    return result
