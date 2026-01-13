import random
from collections import deque, Counter, defaultdict


def shortest_distance_from_all_buildings(grid):
    # https://leetcode.com/problems/shortest-distance-from-all-buildings
    # https://algo.monster/liteproblems/317
    """
    Given a 2D grid, finds the shortest distance from an empty land (0) to all buildings (1)
    such that the sum of distances to all buildings is minimized. Returns -1 if impossible.

    1: Building, 0: Land, 2: Obstacle

    [1, 0, 2, 0, 1]
    [0, 0, 0, 0, 0]
    [0, 0, 1, 0, 0]

    Args:
        grid (List[List[int]]): 2D grid of 0s (empty land), 1s (buildings), 2s (obstacles)

    Returns:
        int: Minimum sum distance from an empty land to all buildings, or -1 if not possible.

    Approach:
        1. For each building, perform BFS to compute distances to all reachable empty lands.
        2. Maintain two grids:
            - distance[r][c]: sum of distances from all visited buildings to this cell
            - reach[r][c]: number of buildings that can reach this cell
        3. After BFS from all buildings, iterate through all empty lands (0s)
           and find the minimum distance where reach[r][c] == total number of buildings.

    Time Complexity:
        O(m * n * number_of_buildings) where m, n are grid dimensions

    Space Complexity:
        O(m * n) for distance and reach matrices and BFS queue
    """
    if not grid or not grid[0]:
        return -1

    m, n = len(grid), len(grid[0])
    distance_sum_grid = [[0] * n for _ in range(m)]
    reach_grid = [[0] * n for _ in range(m)]

    buildings = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                buildings.append((i, j))

    if len(buildings) == 0:
        return -1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def bfs(row, col):
        """BFS from a building to all reachable empty lands"""
        visited = [[False] * n  for _ in range(m)]
        queue = deque([(row, col, 0)])
        visited[row][col] = True

        while queue:
            x, y, distance = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and grid[nx][ny] == 0:
                    visited[nx][ny] = True
                    distance_sum_grid[nx][ny] += (distance + 1)
                    reach_grid[nx][ny] += 1
                    queue.append((nx, ny, distance + 1))

    # Run BFS from each building
    for x, y in buildings:
        bfs(x, y)

    min_distance  = float('inf')
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0 and reach_grid[i][j] == len(buildings):  # all building should reach this land
                min_distance = min(min_distance, distance_sum_grid[i][j])

    return min_distance if min_distance != float('inf') else -1


def nested_lst_weight_sum(nested_list):
    # https://leetcode.com/problems/nested-list-weight-sum
    # https://jzleetcode.github.io/posts/leet-0339-lint-0551-nested-list-weight-sum/
    result, stack = 0, [(nested_list, 1)]

    while stack:
        elements, level = stack.pop()
        for element in elements:
            if element.isInteger():
                result += element.getInteger() * level
            else:
                stack.append((element.getList(), level + 1))

    return result


class MovingAverage1:
    # https://www.jointaro.com/interviews/questions/moving-average-from-data-stream/
    # https://algo.monster/liteproblems/346
    """
    Implements a Moving Average calculator over a sliding window of fixed size.

    Each call to `next(value)`:
      - Adds a new value to the stream.
      - Maintains a running sum and count of elements in the window.
      - If the window exceeds the specified size, removes the oldest element.
      - Returns the current moving average.

    This approach avoids recomputing sums repeatedly by updating incrementally.

    Example:
        m = MovingAverage(3)
        m.next(1)  -> 1.0       # [1]
        m.next(10) -> 5.5       # [1, 10]
        m.next(3)  -> 4.67      # [1, 10, 3]
        m.next(5)  -> 6.0       # [10, 3, 5]

    Time Complexity:  O(1) per operation
    Space Complexity: O(size)
    """

    def __init__(self, size):
        self.size = size
        self.window = deque()
        self.current_count = 0
        self.current_total = 0

    def next(self, value):
        self.window.append(value)
        self.current_count += 1
        self.current_total += value

        if self.current_count > self.size:
            self.current_count -= 1
            self.current_total -= self.window.popleft()

        return self.current_total / self.current_count


def top_k_frequent_elements(nums, k):
    # https://leetcode.com/problems/top-k-frequent-elements
    """
    Time: O(n + m)
    Space: O(n + m)
    """
    counts, freq_dict, result = Counter(nums), defaultdict(list), []

    for num, freq in counts.items():
        freq_dict[freq].append(num)

    for freq in reversed(range(len(nums) + 1)):
        result.extend(freq_dict[freq])
        if len(result) >= k:
            return result[:k]
    return result[:k]


def minimum_remove_to_make_valid_parentheses(s):
    # https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses
    """
    Keys points to remember:
        1. Convert `s` from a string to a list since strings are immutable in Python.
        2. Create a stack that stores only the indices of unmatched "(" positions only.
        3. If there are any indices left in the stack then, set those indices to empty
    """
    s = list(s)
    stack = []

    for i, char in enumerate(s):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                stack.pop()
            else:
                s[i] = ""

    while stack:
        i = stack.pop()
        s[i] = ""

    return "".join(s)


class RandomPickWeight:
    # https://leetcode.com/problems/random-pick-with-weight
    def __init__(self, w):
        self.w = w
        self.total = sum(self.w)
        self.n = len(self.w)

        # 1. Normalize the weights. Now self.w contains probabilities that sum up to 1.
        # Example: w = [1, 3, 2] → self.w = [1/6, 3/6, 2/6] = [0.166, 0.5, 0.333].
        for i in range(self.n):
            self.w[i] = self.w[i] / self.total

        # 2. Convert to Cumulative Distribution Function (CDF) → self.w = [0.166, 0.666, 1.0].
        # This means:
        #   Index 0 covers [0, 0.166]
        #   Index 1 covers (0.166, 0.666]
        #   Index 2 covers (0.666, 1.0]
        for i in range(1, self.n):
            self.w[i] += self.w[i - 1]

    def pick_index(self):
        # k = random.random()
        # return bisect.bisect_left(self.w, k)
        # or
        # 3. Pick a random number from uniform distribution between 0, 1
        k = random.uniform(0, 1)  # returns a floating point number
        left, right = 0, self.n
        while left < right:
            mid = left + (right - left) // 2
            if k > self.w[mid]:
                left = mid + 1
            else:
                right = mid
        return left


def valid_word_abbreviation(word, abbr):
    # https://leetcode.com/problems/valid-word-abbreviation/description/
    # https://shandou.medium.com/leetcode-408-valid-word-abbreviation-63f1ed6461de
    # https://neetcode.io/problems/valid-word-abbreviation?list=neetcode250
    """
    A string can be shortened by replacing any number of non-adjacent, non-empty substrings with their lengths (without leading zeros).
    word = "apple", abbr = "a3e"
    word = "abbreviation", abbr = "a2reviation"
    Need to check 3 conditions:
        1. if both chars are same then, increment the indices and skip the rest of the loop
        2. if abbr char is not a digit or it is zero then, return False
        3. Now start marking index of abbr as it might be number and increment till the end of number so that you can skip all the characters in word
        4. Check if you reached the end of the both the word and abbreviation.
    """
    i = j = 0
    m , n = len(word), len(abbr)

    while i < m and j < n:
        if abbr[j] == word[i]:  # Condition 1: Check if both are same characters
            i += 1
            j += 1
            continue  # skip rest of the loop

        if not abbr[j].isdigit() or abbr[j] == '0':  # Condition 2: Check if 1st character is not a digit or if it is a 0
            return False

        start = j
        while j < n and abbr[j].isdigit():  # Condition 3: Check if the chars in abbr are digits and pick that window
            j += 1

        skip = int(abbr[start: j])  # Skip the window of 'word' picked by 'abbr'
        i += skip

    return i == m and j == n  # Make sure we reached the end and return based on that condition.


def add_strings(num1, num2):
    # https://leetcode.com/problems/add-strings
    i, j, carry = len(num1) - 1, len(num2) - 1, 0
    result = []

    while i >= 0 or j >= 0 or carry:
        if i >= 0:
            digit1 = ord(num1[i]) - ord('0')
        else:
            digit1 = 0
        if j >= 0:
            digit2 = ord(num2[j]) - ord('0')
        else:
            digit2 = 0
        total = digit1 + digit2 + carry
        carry, digit = divmod(total, 10)
        result.append(str(digit))
        i, j = i - 1, j - 1
    return "".join(result[::-1])


def diagonal_traversal(mat):
    # https://leetcode.com/problems/diagonal-traverse
    """
    matrix
          0 1 2
        0 1 2 3
        1 4 5 6
        2 7 8 9
    """
    m, n = len(mat), len(mat[0])
    row_col_map = defaultdict(list)

    for i in range(m):
        for j in range(n):
            row_col_map[i + j].append(row_col_map[i][j])

    result = []
    for key in sorted(row_col_map.keys()):
        if key % 2 == 0:
            result.extend(row_col_map[key][::-1])
        else:
            result.extend(row_col_map[key])

    return result
