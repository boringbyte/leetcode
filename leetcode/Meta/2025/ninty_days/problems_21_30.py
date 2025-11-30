from collections import defaultdict, deque


def power_x_n(x, n):
    # https://leetcode.com/problems/powx-n
    if abs(x) < 1e-40:
        return 0

    if n < 0:
        return power_x_n(1 / x, -n)
    elif n == 0:
        return 1
    else:
        a = power_x_n(x, n // 2)
        if n % 2 == 0:
            return a * a
        else:
            return a * a * x


def maximum_subarray(nums):
    # https://leetcode.com/problems/maximum-subarray
    """
    This is a classic Kadane Algorithm.
    As I walk through the array, I'm carrying a running sum.
    If my sum ever becomes negative, it is harmful to keep it. I should drop it and start fresh.
    """
    result = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)  # Either start fresh or extend the previous subarray.
        result = max(result, current_sum)
    return result


def merge_intervals(intervals):
    # https://leetcode.com/problems/merge-intervals
    intervals = sorted(intervals, key=lambda x: x[0])
    result = [intervals[0]]
    for start, end in intervals[1:]:
        if result[-1][-1] >= start:
            result[-1][-1] = max(result[-1][-1], end)
        else:
            result.append([start, end])
    return result


def interval_list_intersections(first_list, second_list):
    # https://leetcode.com/problems/interval-list-intersections
    pass


def vertical_order_traversal_of_a_binary_tree(root):
    # https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree
    column_dict = defaultdict(list)
    row = col = 0
    queue = deque([(root, row, col)])
    result = []

    while queue:
        current, row, col = queue.popleft()
        column_dict[col].append((row, current.val))
        if current.left:
            queue.append((current.left, row + 1, col - 1))
        if current.right:
            queue.append((current.right, row + 1, col + 1))

    for col_key in sorted(column_dict.keys()):
        row_val_pairs = column_dict[col_key]
        row_val_pairs = sorted(row_val_pairs, key=lambda x: (x[0], x[1]))
        values = [val for row, val in row_val_pairs]
        result.append(values)

    return result


def valid_number():
    # https://leetcode.com/problems/valid-number
    pass


def climbing_stairs(n):
    # https://leetcode.com/problems/climbing-stairs
    """
    This is recursive solution and by using @cache decorator from functools
    if n <= 3:
        return n
    else:
        return climbing_stairs(n - 1) + climbing_stairs(n - 2)
    """
    if n <= 3:
        return n

    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2

    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[-1]


def simplify_path(path):
    stack = []
    components = path.split('/')
    for component in components:
        if component in ['', '.']:
            continue
        elif component == '..':
            if stack:
                stack.pop()
        else:
            stack.append(component)
    return '/' + '/'.join(stack)


def sort_colors(nums):
    # https://leetcode.com/problems/sort-colors

    pass



def subsets(nums):
    # https://leetcode.com/problems/subsets
    result, n = [], len(nums)

    def backtrack(sofar, start):
        result.append(sofar[:])

        for i in range(start, n):
            chosen = nums[i]
            backtrack(sofar + [chosen], i + 1)

    backtrack(sofar=[], start=0)
    return result