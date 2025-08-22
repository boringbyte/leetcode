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

    while top <= bottom:  # This is for inverting the rows along the horizontal axis
        matrix[top], matrix[bottom] = matrix[bottom], matrix[top]
        top += 1
        bottom -= 1

    for row in range(n):  # This is for transposing along the diagonal
        for col in range(row + 1, n):
            matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]

    return matrix


def meeting_rooms_2(intervals):
    # https://walkccc.me/LeetCode/problems/253/#__tabbed_1_3
    # https://medium.com/@edward.zhou/leetcode-253-meeting-rooms-ii-explained-python3-solution-3f8869612df
    if not intervals:
        return 0

    heap = []
    intervals = sorted(intervals, key=lambda x: x[0])
    heapq.heappush(heap, intervals[0][1])

    for start, end in intervals[1:]:
        if start >= heap[0]:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

    return len(heap)


def word_search(board, word):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
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
            if dfs(i, j, 0):
                return True
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

    for current_amount in range(1, amount + 1):
        for coin in coins:
            remaining_amount = current_amount - coin
            if remaining_amount >= 0:
                dp[current_amount] = min(dp[current_amount], 1 + dp[remaining_amount])

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


def longest_cont_subarray_with_abs_diff_le_limit(nums, limit):
    # https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/solutions/609771/java-c-python-deques-o-n
    max_deque = collections.deque()
    min_deque = collections.deque()
    i = 0
    for num in nums:
        while max_deque and num > max_deque[-1]:
            max_deque.pop()
        while min_deque and num < min_deque[-1]:
            min_deque.pop()
        max_deque.append(num)
        min_deque.append(num)
        if max_deque[0] - min_deque[0] > limit:
            if max_deque[0] == nums[i]:
                max_deque.popleft()
            if min_deque[0] == nums[i]:
                min_deque.popleft()
            i += 1
    return len(nums) - i


def non_overlapping_intervals(intervals):
    # https://leetcode.com/problems/non-overlapping-intervals/submissions/1741423184
    intervals = sorted(intervals, key=lambda x: x[1])
    final_end, result = intervals[0][1], 0
    for start, end in intervals[1:]:
        if start >= final_end:
            final_end = end
        else:
            result += 1
    return result


def merge(intervals):
    intervals, result = sorted(intervals, key=lambda x: x[0]), []
    for start, end in intervals:
        if not result or start > result[-1][-1]:
            result.append([start, end])
        else:
            result[-1][-1] = max(result[-1][-1], end)
    return result


class Bank:

    def __init__(self, balance: list[int]):
        self.balance = balance

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if self.withdraw(account1, money):
            if self.deposit(account2, money):
                return True
            self.deposit(account1, money)
        return False

    def deposit(self, account: int, money: int) -> bool:
        if 1 <= account <= len(self.balance):
            self.balance[account - 1] += money
            return True
        return False

    def withdraw(self, account: int, money: int) -> bool:
        if 1 <= account <= len(self.balance) and self.balance[account - 1] >= money:
            self.balance[account - 1] -= money
            return True
        return False


def candy_crush(board: list[list[int]]) -> list[list[int]]:
    # https://algo.monster/liteproblems/723
    # Dimensions of the board
    num_rows, num_cols = len(board), len(board[0])

    # Flag to indicate if we should continue crushing candies
    should_crush = True

    # Keep crushing candies until no more moves can be made
    while should_crush:
        should_crush = False  # Reset the flag for each iteration

        # Mark the candies to be crushed horizontally
        for row in range(num_rows):
            for col in range(num_cols - 2):
                # Check if three consecutive candies have the same value
                if abs(board[row][col]) != 0 and abs(board[row][col]) == abs(board[row][col + 1]) == abs(board[row][col + 2]):
                    should_crush = True  # We will need another pass after crushing
                    # Mark the candies for crushing by negating their value
                    board[row][col] = board[row][col + 1] = board[row][col + 2] = -abs(board[row][col])

        # Mark the candies to be crushed vertically
        for col in range(num_cols):
            for row in range(num_rows - 2):
                # Check if three consecutive candies vertically have the same value
                if abs(board[row][col]) != 0 and abs(board[row][col]) == abs(board[row + 1][col]) == abs(board[row + 2][col]):
                    should_crush = True  # We will need another pass after crushing
                    # Mark the candies for crushing by negating their value
                    board[row][col] = board[row + 1][col] = board[row + 2][col] = -abs(board[row][col])

        # Drop the candies to fill the empty spaces caused by crushing
        if should_crush:
            for col in range(num_cols):
                # Pointer to where the next non-crushed candy will fall
                write_row = num_rows - 1
                for row in range(num_rows - 1, -1, -1):
                    # If the candy is not marked for crushing, bring it down
                    if board[row][col] > 0:
                        board[write_row][col] = board[row][col]
                        write_row -= 1

                # Fill the remaining spaces at the top with zeros
                while write_row >= 0:
                    board[write_row][col] = 0
                    write_row -= 1

    # Return the modified board after all possible crushes are completed
    return board


class FileSystem:
    # https://leetcode.ca/all/1166.html

    def __init__(self):
        # Store path → value mapping
        self.paths = {}

    def create_path(self, path: str, value: int) -> bool:
        # If path already exists, can't create
        if path in self.paths:
            return False

        # Extract parent path
        parent = path[:path.rfind("/")]
        # If parent is not root and not already created → invalid
        if parent != "" and parent not in self.paths:
            return False

        # Otherwise, create the new path
        self.paths[path] = value
        return True

    def get(self, path: str) -> int:
        return self.paths.get(path, -1)


def rotating_the_box(box_grid):
    # # -> Stone, . -> Empty Space, # * -> Obstacle
    m, n = len(box_grid), len(box_grid[0])

    # Step 1: Apply gravity to each row
    for row in box_grid:
        # Start from the rightmost position
        empty = n - 1
        for col in range(n - 1, -1, -1):
            if row[col] == '*':  # Obstacle: reset empty position
                empty = col - 1
            elif row[col] == '#':  # Stone: move it to the farthest right possible
                row[col], row[empty] = '.', '#'
                empty -= 1
        # '.' remains as is

    # Step 2: Rotate the matrix 90° clockwise
    rotated = [[None] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            rotated[j][m - 1 - i] = box_grid[i][j]

    return rotated


def minimum_operations_to_write_letter_y_on_grid(grid):
    n = len(grid)
    center = n // 2

    # Step 1: Identify Y cells
    y_cells = set()
    for r in range(center + 1):
        y_cells.add((r, r))             # left diagonal
        y_cells.add((r, n - 1 - r))     # right diagonal
    for r in range(center, n):
        y_cells.add((r, center))        # vertical line down

    # Step 2: Count frequencies
    count_y = [0] * 3
    count_other = [0] * 3
    for r in range(n):
        for c in range(n):
            v = grid[r][c]
            if (r, c) in y_cells:
                count_y[v] += 1
            else:
                count_other[v] += 1

    total_y = sum(count_y)
    total_other = sum(count_other)

    # Step 3: Try all valid (a, b)
    result = float("inf")
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            ops_y = total_y - count_y[a]
            ops_other = total_other - count_other[b]
            result = min(result, ops_y + ops_other)

    return result


def number_of_black_blocks(m, n, coordinates):
    count = collections.defaultdict(int)  # (block_x, block_y) -> black count
    directions = [(-1, -1), (-1, 0), (0, -1), (0, 0)]  # These are different from normal directions

    for x, y in coordinates:
        for dx, dy in directions:
            if 0 <= x + dx < m - 1 and 0 <= y + dy < n - 1:
                count[(x + dx, y + dy)] += 1

    result = [0] * 5
    total_blocks = (m - 1) * (n - 1)

    for v in count.values():
        result[v] += 1

    result[0] = total_blocks - sum(result[1:])
    return result


def biggest_3_rhombus_sums_in_grid(grid):
    l = []  # stores the sums
    n = len(grid)  # number of rows
    m = len(grid[0])  # number of cols

    # iterate over every tile
    for i in range(n):
        for j in range(m):
            # top tile point of rhombus
            ans = grid[i][j]
            l.append(grid[i][j])  # valid rhombus sum (one point)

            # distance var to store distance from j to both ends of rhombus
            # (used to increase size of rhombus)
            distance = 1

            # make sure the tile is within grid bounds
            while i + distance < n and j - distance >= 0 and j + distance < m:
                # iterate over all possible rhombus sizes using the distance var

                a = i + distance  # next row
                b = j + distance  # col to the right
                c = j - distance  # col to the left

                # right tile point of rhombus: grid[a][b]
                # left tile point of rhombus: grid[a][c]
                ans += grid[a][b] + grid[a][c]

                # a dummy variable to store the present sum of the sides
                # (left and right tile point)
                dummy = 0
                while True:
                    # iterate to find the bottom point of rhombus

                    c += 1  # left tile point moves toward the right
                    b -= 1  # right tile point moves toward the left
                    a += 1  # moves to the bottom (next row)

                    if c == m or b == 0 or a == n:
                        break  # reached bounds

                    # left and right cols met at "middle"
                    if c == b:  # found the bottom tile point of rhombus
                        # add bottom tile sum to sides (left and right) sum
                        dummy += grid[a][b]
                        l.append(ans + dummy)  # appending the obtained sum
                        break

                    dummy += grid[a][b] + grid[a][c]  # adding both sides sum to dummy

                distance += 1

    l = list(set(l))  # remove duplicates
    l.sort(reverse=True)
    # return first 3 largest sums
    return l[:3]


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
        if num in (2, 3):
            return True
        if num < 2 or num % 2 == 0 or num % 3 == 0:
            return False
        r = int(num ** 0.5)
        for i in range(5, r + 1, 6):
            if num % i == 0 or num % (i + 2) == 0:
                return False
        return True

    result = 0
    for num in nums:
        # cube_root = round(num ** (1/3))
        #
        # # Verify exactly: cube_root ** 3 == num (avoids float error) and is_prime(cube_root).
        # if cube_root ** 3 == num and is_prime(cube_root):
        #     result += 1 + cube_root + cube_root ** 2 + cube_root ** 3
        # else:
        divs = set()
        for div in range(1, int(num ** 0.5) + 1):
            if num % div == 0:
                divs.add(div)
                divs.add(num // div)
            if len(divs) > 4:
                break
        if len(divs) == 4:
            result += sum(divs)
    return result


def color_the_array(n, queries):
    # https://leetcode.com/problems/number-of-adjacent-elements-with-the-same-color/
    colors = [0] * n
    result = []
    count = 0  # current number of adjacent pairs with same color

    for index, new_color in queries:
        # Decrement count for pairs broken by changing colors[index]
        if index > 0 and colors[index] != 0 and colors[index] == colors[index-1]:
            count -= 1
        if index < n-1 and colors[index] != 0 and colors[index] == colors[index+1]:
            count -= 1

        # Update the color
        colors[index] = new_color

        # Increment count for pairs formed by new color
        if index > 0 and colors[index] == colors[index-1]:
            count += 1
        if index < n-1 and colors[index] == colors[index+1]:
            count += 1

        result.append(count)

    return result


class LRUCache:
    # https://leetcode.com/problems/lru-cache
    def __init__(self, capacity: int):
        self.cache = collections.OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Move key to the end (mark as most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Remove old entry
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item (first item in OrderedDict)
            self.cache.popitem(last=False)
        # Insert new value as most recent
        self.cache[key] = value


def num_of_sub_arrays_that_match_pattern_1(nums, pattern):
    # https://leetcode.com/problems/number-of-subarrays-that-match-a-pattern-i
    m = len(nums)
    n = len(pattern)
    result = 0

    for i in range(m - n):
        match = True
        for k in range(n):
            if pattern[k] == 1 and nums[i + k + 1] <= nums[i + k]:
                match = False
                break
            elif pattern[k] == 0 and nums[i + k + 1] != nums[i + k]:
                match = False
                break
            elif pattern[k] == -1 and nums[i + k + 1] >= nums[i + k]:
                match = False
                break
        if match:
            result += 1

    return result


def odd_event_list(head: ListNode | None) -> ListNode | None:
    # https://leetcode.com/problems/odd-even-linked-list/description/
    if not head or not head.next:
        return head

    odd, even, even_head = head, head.next, head.next

    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = even_head
    return head
