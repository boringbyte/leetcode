import collections
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def number_of_islands(grid):
    result, m, n = 0, len(grid), len(grid[0])
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(x, y):
        if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
            grid[x][y] = '0'
            for dx, dy in directions:
                dfs(x + dx, y + dx)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                result += 1
    return result


def spiral_matrix(matrix):

    result = []
    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1

        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        if top <= bottom:
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result


def add_two_numbers(list1, list2):
    if list1 and list2:
        head = current = ListNode()
        carry = 0

        while list1 or list2 or carry:
            if list1:
                carry += list1.val
                list1 = list1.next
            if list2:
                carry += list2.val
                list2 = list2.next
            carry, digit = divmod(carry, 10)
            current.next = ListNode(digit)
            current = current.next
        return head.next
    else:
        return list1 or list2


def rotate_image(matrix):
    n = len(matrix)
    top, bottom = 0, n - 1

    while top < bottom:  # This is for inverting the rows along the horizontal axis
        matrix[top], matrix[bottom] = matrix[bottom], matrix[top]
        top += 1
        bottom -= 1

    for row in range(n):  # This is for transposing along the diagonal
        for col in range(row + 1, n):
            matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]

    return matrix


def meeting_rooms_2(intervals):
    # https://walkccc.me/LeetCode/problems/253/#__tabbed_1_3
    if not intervals:
        return 0

    heap, n = [], len(intervals)
    intervals = sorted(intervals, key=lambda x: x[0])
    heapq.heappush(heap, intervals[0][1])

    for start, end in intervals[1:]:
        if start >= heap[0]:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

    return len(heap)


def word_search(board, word):
    directions, visited = [(1, 0), (0, 1), (-1, 0), (0, -1)], set()
    m, n, k = len(board), len(board[0]), len(word)

    def dfs(x, y, idx):
        if idx == k:
            return True
        if x < 0 or x >= m or y < 0 or y >= n or board[x][y] != word[idx]:
            return False

        board[x][y], temp = '#', board[x][y]
        for dx, dy in directions:
            if dfs(x + dx, y + dy, idx + 1):
                return True
        board[x][y] = temp
        return False

    for i in range(m):
        for j in range(n):
            dfs(i, j, 0)
    return False


def best_time_to_buy_and_sell_stock_2(prices):
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/solutions/5816678/video-sell-a-stock-immediately
    total_profit = 0

    for i in range(1, len(prices)):
        buy_price, sell_price = prices[i - 1], prices[i]
        if sell_price > buy_price:
            total_profit += sell_price - buy_price
    return total_profit


def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # base case

    for x in range(1, amount + 1):
        for coin in coins:
            if x - coin >= 0:
                dp[x] = min(dp[x], 1 + dp[x - coin])

    return dp[amount] if dp[amount] != float('inf') else -1


def rotate_array(nums, k):
    # https://leetcode.com/problems/rotate-array/solutions/1730142/java-c-python-a-very-very-well-detailed-explanation
    def reverse(nums, l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

    n = len(nums)
    if n == k:  # If the length of the array and the k are same then we don't need to rotate
        return

    k = k % n  # If the k > n, it creates a k such that it falls between 0 and n

    reverse(nums, 0, n - 1)
    reverse(nums, 0, k - 1)
    reverse(nums, k, n - 1)


def simplify_path(path):
    stack, components = [], path.split('/')
    for component in components:
        if component in ['', '.']:
            continue
        elif component == '..':
            if stack:
                stack.pop()
        else:
            stack.append(component)
    return '/' + '/'.join(stack)


def length_of_longest_common_prefix(arr1, arr2):
    prefix_set = set()
    for num in arr1:
        while num > 0:
            if num not in prefix_set:
                prefix_set.add(num)
            num //= 10

    result = float('-inf')

    for num in arr2:
        size = len(str(num))
        while num > 0:
            if num in prefix_set:
                break
            num //= 10
            size -= 1
        result = max(result, size)
    return result


def longest_cont_subarray_with_abs_diff(nums, limit):
    # https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/solutions/609771/java-c-python-deques-o-n
    pass


def non_overlapping_intervals(intervals):
    overlap, result = float('-inf'), 0
    intervals = sorted(intervals, key=lambda x: x[0])
    for start, end in intervals:
        if start >= overlap:
            overlap = end
        else:
            result += 1
    return result


def biggest_3_rhombus_sums_in_grid(grid):
    pass



def k_diff_pairs_in_array(nums, k):
    # https://leetcode.com/problems/k-diff-pairs-in-an-array/solutions/100135/java-python-easy-understood-solution
    result, counter = 0, collections.Counter(nums)

    for key in counter:
        if (k == 0 and counter[key] > 1) or (k > 0 and key + k in counter):
            result += 1
    return result


def counting_alternating_subarrays(nums):
    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        if nums[i] != nums[i - 1]:
            dp[i] = dp[i - 1] + 1

    return sum(dp)


def four_divisors(nums):
    """
    A number with exactly 4 divisors must be:
        - The cube of a prime (p³ → divisors = {1, p, p², p³}).  # This is not exactly necessary as the below condition covers.
        - The product of two distinct primes (p * q → divisors = {1, p, q, pq}).
    """

    def is_prime(num):
        if num < 2 or num % 2 == 0:
            return False
        for i in range(3, int(num ** 0.5) + 1, 2):
            if num % i == 0:
                return False
        return True

    def is_prime2(num):
        if num < 2 or num % 2 == 0 or num % 3 == 0:
            return False
        if num in (2, 3):
            return True
        r = int(num ** 0.5)
        for i in range(5, r + 1, 6):
            if num % i == 0 or num % (i + 2) == 0:
                return False
        return True


    result = 0

    for num in nums:
        cube_root = round(num ** (1/3))

        # Verify exactly: cube_root ** 3 == num (avoids float error) and is_prime(cube_root).
        if cube_root ** 3 == num and is_prime(cube_root):
            result += 1 + cube_root + cube_root ** 2 + cube_root ** 3
        else:
            divs = set()
            for d in range(1, int(num ** 0.5) + 1):
                if num % d == 0:
                    divs.add(d)
                    divs.add(num // d)
                if len(divs) > 4:
                    break
            if len(divs) == 4:
                result += sum(divs)
    return result