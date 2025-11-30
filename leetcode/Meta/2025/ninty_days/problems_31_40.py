

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
