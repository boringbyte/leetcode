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


def remove_invalid_parentheses(s):
    pass


def best_time_to_buy_and_sell_stock_with_cooldown(prices):
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/75924/Most-consistent-ways-of-dealing-with-the-series-of-stock-problems
    pass


def burst_balloons(nums):
    pass


def coin_change1(coins, amount):
    # https://leetcode.com/problems/coin-change/discuss/1475250/Python-4-solutions%3A-Top-down-DP-Bottom-up-DP-Space-O(amount)-Clean-and-Concise
    n = len(coins)

    @lru_cache
    def dfs(total):
        if total == 0:
            return 0
        ans = float('inf')
        for coin in coins:
            if total >= coin:
                ans = min(coin, dfs(total - coin) + 1)
        return ans
    result = dfs(amount)
    return result if result != float('inf') else -1


def coin_change2(coins, amount):
    n, coins = len(coins), sorted(coins)
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for amt in range(1, amount + 1):
        for coin in coins:
            if amt >= coin:
                dp[amt] = min(dp[amt], dp[amt - coin] + 1)
            else:
                break
    return dp[amount] if dp[amount] != float('inf') else -1


def odd_even_linked_list(head):
    if not head and not head.next:
        return head
    odd = even = even_head = head

    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = even_head
    return head


def house_robber_3(root):
    pass


def counting_bits(n):
    # https://leetcode.com/problems/counting-bits/discuss/656849/Python-Simple-Solution-with-Clear-Explanation
    counter = [0]
    for i in range(1, n + 1):
        counter.append(counter[i // 2] + i % 2)
    return counter


def top_k_frequent_elements1(nums, k):
    counter_dict, heap, result = collections.Counter(nums), [], []
    for val, priority in counter_dict.items():
        heapq.heappush(heap, (-priority, val))

    for _ in range(k):
        _, val = heapq.heappop(heap)
        result.append(val)
    return result


def top_k_frequent_elements2(nums, k):
    counter_dict, freq_dict = collections.Counter(nums), collections.defaultdict(list)
    for key, val in counter_dict.items():
        freq_dict[key].append(val)

    result = []
    for times in reversed(range(len(nums) + 1)):
        result.extend(freq_dict[times])
        if len(result) >= k:
            return result[:k]
    return result[:k]


def decode_string(s):
    pass


def evaluate_division():
    pass


def queue_construction_by_height(people):
    pass


def partition_equal_subset_sum(nums):
    # https://leetcode.com/problems/partition-equal-subset-sum/discuss/1624939/C%2B%2BPython-5-Simple-Solutions-w-Explanation-or-Optimization-from-Brute-Force-to-DP-to-Bitmask
    n, total = len(nums), sum(nums)

    @lru_cache
    def recursive(total, k):
        if total == 0:
            return True
        if k >= n or total < 0:
            return False
        return recursive(total - nums[k], k + 1) or recursive(total, k + 1)

    return total & 1 == 0 and recursive(total // 2, 0)


def subarray_sum_equals_k(nums, k):
    prefix_sum, prefix_sum_counts, result = 0, {0: 1}, 0
    for num in nums:
        prefix_sum = prefix_sum + num
        diff = prefix_sum - k
        if diff in prefix_sum_counts:
            result += prefix_sum_counts[diff]
        prefix_sum_counts[prefix_sum] = prefix_sum_counts.get(prefix_sum, 0) + 1
    return result


def shortest_unsorted_continuous_subarray(nums):
    # https://leetcode.com/problems/shortest-unsorted-continuous-subarray/discuss/264474/Python-O(n)-2-Loops-and-O(1)-space
    # comments
    n = len(nums)
    if n <= 1:
        return 0

    start, end = -1, 0
    left, right = nums[end], nums[start]

    for i in range(1, n):
        if left > nums[i]:
            end = i
        else:
            left = nums[i]
        if right < nums[~i]:
            start = ~i
        else:
            right = nums[~i]
    if end != 0:
        return end - (n + start) + 1
    else:
        return 0


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

    def count_palindrome(l, r):
        while l >= 0 and r < n and s[l] == s[r]:
            result[0], l, r = result[0] + 1, l - 1, r + 1

    for i in range(n):
        count_palindrome(i, i)
        count_palindrome(i, i + 1)

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
