from collections import deque, Counter


def plus_one(digits):
    # https://leetcode.com/problems/plus-one
    n = len(digits)

    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        else:
            digits[i] = 0

    return [1] + digits


def sliding_window_maximum(nums, k):
    # https://leetcode.com/problems/sliding-window-maximum
    """
    n = len(nums)
    result = []

    for i in range(n - k + 1):
        result.append(max(nums[i:i + k]))

    return result
    """
    index_queue = deque()                                   # Stores indices and the front of the deque is always the maximum of the current window.
    result = []

    for i, num in enumerate(nums):
        while index_queue and nums[index_queue[-1]] < num:  # Remove smaller elements from the back of the deque
            index_queue.pop()

        index_queue.append(i)

        if index_queue[0] == i - k:                         # Remove elements that leave the window
            index_queue.popleft()

        if i >= k - 1:                                      # Add max to result (when window is valid)
            result.append(nums[index_queue[0]])

    return result


def cutting_ribbons(ribbons, k):
    # https://leetcode.com/problems/cutting-ribbons
    # https://algo.monster/liteproblems/1891
    def feasible(size):
        count = sum(ribbon // size for ribbon in ribbons)
        return count >= k

    left, right = 0, max(ribbons)
    first_true_index = 0

    while left <= right:
        mid = (left + right) // 2
        if feasible(mid):
            first_true_index = mid
            left = mid + 1
        else:
            right = mid - 1

    return first_true_index


def friends_of_appropriate_age(ages):
    # https://leetcode.com/problems/friends-of-appropriate-ages

    requests = 0
    age_counter = Counter(ages)

    for x in age_counter:
        for y in age_counter:
            if y <= 0.5 * x + 7:
                continue
            if y > x:
                continue
            if y > 100 and x < 100:
                continue

            if x == y:
                requests += age_counter[x] * (age_counter[x] - 1)       # No self requests
            else:
                requests += age_counter[x] * age_counter[y]             # Everyone is valid

    return requests


def palindrome_linked_list(head):
    # https://leetcode.com/problems/palindrome-linked-list

    def reverse_linked_list(head):
        prev = None

        while head:
            current = head
            head = head.next
            current.next = prev
            prev = current

        return prev

    slow = fast = head

    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    slow = reverse_linked_list(slow)            # Reversing the second part

    while slow:                                 # Only looping on slow as it might be smaller than entire linked list
        if slow.val != head.val:
            return False
        slow = slow.next
        head = head.next

    return True


def check_completeness_of_a_binary_tree(root):
    # https://leetcode.com/problems/check-completeness-of-a-binary-tree
    """
    If you traverse the tree level by level (left → right), once you see a None, every node after that must also be None.
    """
    queue = deque([root])
    seen_none = False

    while queue:
        current = queue.popleft()

        if current is None:
            seen_none = True
        else:
            if seen_none:
                return False
            queue.append(current.left)
            queue.append(current.right)

    return True


class RangeSumQuery:
    # https://leetcode.com/problems/check-completeness-of-a-binary-tree/description/

    def __init__(self, nums):
        self.n = len(nums)
        self.running_sum = [0] * self.n
        self.running_sum[0] = nums[0]

        for i in range(1, self.n):
            self.running_sum[i] = self.running_sum[i - 1] + nums[i]

    def sum_range(self, left: int, right: int) -> int:
        if left == 0:
            return self.running_sum[right]
        else:
            return self.running_sum[right] - self.running_sum[left - 1]


def best_time_to_buy_and_sell_stock_ii(prices):
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii
    profit = 0

    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]

    return profit


def can_place_flowers(flower_bed, n):
    # https://leetcode.com/problems/can-place-flowers
    if n == 0:                                      # There are no new flowers to plant
        return True

    count = 0
    flower_bed = [0] + flower_bed + [0]

    for i in range(1, len(flower_bed) - 1):
        if flower_bed[i - 1] == flower_bed[i] == flower_bed[i + 1] == 0:
            flower_bed[i] = 1
            count += 1
            if count >= n:                          # Early check and return result
                return True

    return False


def boundary_of_binary_tree(root):
    # https://leetcode.com/problems/boundary-of-binary-tree
    # https://algo.monster/liteproblems/545
    # https://takeuforward.org/data-structure/boundary-traversal-of-a-binary-tree
    if not root:
        return []

    result = [root.val]

    def is_leaf(node):
        return node and node.left is None and node.right is None

    # 1. Left boundary (excluding leaves)
    def add_left_boundary(node):
        while node:
            if not is_leaf(node):
                result.append(node.val)
            if node.left:
                node = node.left
            else:
                node = node.right

    # 2. Add all leaves
    def add_leaves(node):
        if is_leaf(node):
            result.append(node.val)
        else:
            if node.left:
                add_leaves(node.left)
            if node.right:
                add_leaves(node.right)

    # 3. Right boundary (excluding leaves, reversed)
    def add_right_boundary(node):
        stack = []
        while node:
            if not is_leaf(node):
                stack.append(node.val)
            if node.right:
                node = node.right
            else:
                node = node.left

        while stack:
            result.append(stack.pop())

    add_left_boundary(root.left)
    add_leaves(root)
    add_right_boundary(root.right)

    return result
