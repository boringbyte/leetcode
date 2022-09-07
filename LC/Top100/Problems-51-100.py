import collections
from functools import lru_cache


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

