from functools import cache
from collections import deque, defaultdict


def merge_sorted_array(nums1, nums2, m, n):
    # https://leetcode.com/problems/merge-sorted-array

    while m > 0 and n > 0:
        if nums1[m - 1] >= nums2[n - 1]:
            nums1[m + n - 1] = nums1[m - 1]
            m - 1
        else:
            nums1[m + n - 1] = nums2[n - 1]
            n - 1

    if n > 0:
        nums1[:n] = nums2[:n]


def pascals_triangle(num_rows):
    # https://leetcode.com/problems/pascals-triangle/description/
    row, result = [1], [[1]]

    for _ in range(1, num_rows):
        left = [0] + row
        right = row + [0]
        row = [x + y for x, y in zip(left, right)]
        result.append(row)
    return result


def best_time_to_buy_and_sell_stock(prices):
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
    """
    This is an application of Kadane's algorithm.
    To apply kadane's algorithm, convert the stock prices to daily stock price differences
    """
    max_profit = current_profit = 0
    diffs = [b - a for a, b in zip(prices, prices[1:])]
    for diff in diffs:
        current_profit = max(0, current_profit + diff)  # Either start fresh or extend the previous subarray.
        max_profit = max(max_profit, current_profit)
    return max_profit


def binary_tree_maximum_path_sum(root):
    # https://leetcode.com/problems/binary-tree-maximum-path-sum
    """
    There is at least 1 node in the root. So base condition is not necessary.

    """
    result = [float('-inf')]

    def dfs(node):
        if not node:
            return 0

        # Recursively compute max contribution from left and right
        left, right = max(dfs(node.left), 0),  max(dfs(node.right), 0)  # Ignore negative paths
        result[0] = max(result[0], left + right + node.val)
        return max(left + node.val, right + node.val) # Return max path SLOPING down from this node

    dfs(root)
    return result[0]


def valid_palindrome(s):
    # https://leetcode.com/problems/valid-palindrome
    s = s.strip()
    left, right = 0, len(s) - 1

    while left < right:
        if not s[left].isalnum():
            left += 1
        elif not s[right].isalnum():
            right -= 1
        else:
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
    return True


def word_ladder(begin_word, end_word, word_list):
    # https://leetcode.com/problems/word-ladder
    if not begin_word or not end_word or not word_list or len(begin_word) != len(end_word) or begin_word not in word_list or end_word not in word_list:
        return 0

    n = len(begin_word)
    hashmap = defaultdict(list)
    for word in word_list:
        for i in range(n):
            intermediate_word = word[:i] + "*" + word[i + 1:]
            hashmap[intermediate_word].append(word)

    queue, visited = deque([(begin_word, 1)]), {begin_word}
    while queue:
        current_word, level = queue.popleft()
        for i in range(n):
            intermediate_word = current_word[:i] + "*" + current_word[i + 1:]
            for word in hashmap[intermediate_word]:
                if word == end_word:
                    return level + 1
                if word not in visited:
                    visited.add(word)
                    queue.append((word, level + 1))
    return 0


def sum_root_to_leaf_numbers(root):
    # https://leetcode.com/problems/sum-root-to-leaf-numbers
    stack, result = [(root, 0)], 0
    while stack:
        current, val = stack.pop()
        val = val * 10 + current.val
        if current.left is None and current.right is None:
            result += val
        if current.left:
            stack.append((current.left, val))
        if current.right:
            stack.append((current.right, val))
    return result


def word_break(s, word_dict):
    # https://leetcode.com/problems/word-break/description/
    """
    queue = deque([0])
    visited = {0}
    word_set = set(word_dict)
    n =  len(s)

    while len(queue) > 0:
        current = queue.popleft()
        for i in range(current + 1, n + 1):
            if i in visited:
                continue
            if s[current: i] in word_set:
                if i == n:
                    return True
                queue.append(i)
                visited.add(i)
    return False
    """
    word_set, n = set(word_dict), len(s)

    @cache
    def backtrack(k):
        if k == n:
            return True
        for i in range(k, n):
            chosen = s[k: i + 1]
            if chosen in word_set and backtrack(i + 1):
                return True
        return False

    return backtrack(k=0)