import string
from collections import defaultdict, deque


def power_x_n(x, n):
    # https://leetcode.com/problems/powx-n
    """
    Condition 1: Think if x is very small then, just return 0
    Next conditions on n:
        - If n is negative, flip x to 1/x and make n positive.
        - If n is zero, the result is 1.
        - Otherwise, compute power(x, n // 2) once
            - Square it if n is even,
            - multiply by x only if n is odd.
    """
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
    As I walk through the array, I'm keep a running sum.
    If my sum ever goes negative, it is harmful to keep it, so I drop it and start fresh from the current number.
    """
    result = running_sum = nums[0]
    for num in nums[1:]:
        running_sum = max(num, running_sum + num)  # Either start fresh or extend the previous subarray.
        result = max(result, running_sum)
    return result


def merge_intervals(intervals):
    # https://leetcode.com/problems/merge-intervals
    """
    Interval merging strategy:

    1. Always sort intervals by start time first.
    2. Keep a result bucket that stores merged intervals.
    3. For each new interval:
       - Compare its start with the end of the last interval in result.
       - If they overlap (last_end >= current_start), merge by extending the end to max(last_end, current_end).
       - Otherwise, no overlap → add the interval as a new entry.

    Think of it as extending the last block if possible; otherwise, start a new block.
    """
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
    """
    Walk through both interval lists using two pointers.

    At each step:
    - The overlap (if any) is the max of the starts and the min of the ends.
    - If start <= end, we found an intersection.

    Move forward the interval that ends first, because it cannot overlap with anything further.
    """
    i = j = 0
    result = []

    while i < len(first_list) and j < len(second_list):
        start = max(first_list[i][0], second_list[j][0])
        end = min(first_list[i][1], second_list[j][1])

        if start <= end:
            result.append([start, end])

        # Advance the pointer with the smaller end time
        if first_list[i][1] < second_list[j][1]:
            i += 1
        else:
            j += 1

    return result


def vertical_order_traversal_of_a_binary_tree(root):
    # https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree
    """
    As the expected output is based on vertical traversal, group nodes based on column.
    Final ordering rules:
        1. Read columns from left to right.
        2. Within each column, sort nodes by row (top to bottom).
        3. If rows are equal, sort by node value.
    """
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


def valid_number(s):
    # https://leetcode.com/problems/valid-number
    """
    Check if a string is a valid number.

    Rules:
    1. Optional sign (+/-) at the start or immediately after 'e'.
    2. At most one decimal point (.) and it must be before 'e'.
    3. 'e' can appear at most once and must follow a number; it requires digits after it.
    4. Only digits, signs, '.', and 'e' are allowed.
    5. At least one digit must appear (either before or after 'e').

    Approach:
    - Scan left to right, tracking if we've seen a dot, 'e', and a digit.
    - Reject invalid positions of signs, dots, or 'e'.
    - Return True only if we have at least one valid digit at the end.
    """
    s = s.strip()
    met_dot = met_e = met_digit = False
    for i, char in enumerate(s):
        if char in '+-':
            if i > 0 and s[i - 1].lower() != 'e':
                return False
        elif char == '.':
            if met_dot or met_e:
                return False
            met_dot = True
        elif char.lower() == 'e':
            if met_e or not met_digit:
                return False
            met_e, met_digit = True, False
        elif char in string.digits:
            met_digit = True
        else:
            return False
    return met_digit


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
    """
    This problem is not about sorting. It is about partitioning into 3 groups in one scan.
    [ all 0s | all 1s | all 2s ]

    low: boundary between 0-region and 1-region
    mid: current element
    high: boundary between 2-region and unexplored region

    [ 0s | 1s | ??? | 2s ]
       ↑     ↑        ↑
      low   mid      high

    if nums[mid] which means current element.
    1. If nums[mid] == 0 then, move it to the front region. Swap it with nums[low], expand both regions.
    2. If nums[mid] == 1 then, it is already in the correct region. Just move mid forward.
    3. If nums[mid] == 2 then, move it to the back region. Swap it with nums[high], shirk high.
        But don't advance mid, because the number swapped in must be rechecked.
    """
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[high], nums[mid] = nums[mid], nums[high]
            high -= 1


def subsets(nums):
    # https://leetcode.com/problems/subsets
    result, n = [], len(nums)

    def backtrack(sofar, start):
        result.append(sofar[:])

        for i in range(start, n):
            chosen = nums[i]
            # As sofar is a list, a new copy is sent to recursive function,
            # and we don't have to explicitly remove chosen from the sofar list as we not append to sofar list.
            backtrack(sofar + [chosen], i + 1)

    backtrack(sofar=[], start=0)
    return result
