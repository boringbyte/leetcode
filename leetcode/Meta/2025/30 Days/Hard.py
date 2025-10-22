import heapq
from collections import deque, defaultdict, Counter
from leetcode.CapitalOne.relevant_leetcode.Easy import ListNode

def vertical_traversal_of_a_binary_tree(root):
    # https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree
    """
    Performs a vertical order traversal of a binary tree.

    In vertical order traversal, nodes are grouped by their vertical column index.
    - The root node is at column 0.
    - For each node, its left child is in column -1 and its right child is in column +1.
    - Within each column, nodes are ordered first by their level (top to bottom)
      and then by their value if multiple nodes share the same level.

    Approach:
    1. Perform a level-order traversal (BFS) using a queue.
    2. Keep track of both the `level` (depth) and `column` index for each node.
    3. Store (level, value) pairs in a dictionary keyed by `column`.
    4. After traversal, sort the columns (from leftmost to rightmost),
       and within each column, sort entries by `level` and then value.
    5. Collect the sorted node values per column.

    Args:
        root (TreeNode): The root of the binary tree.

    Returns:
        List[List[int]]: A list of lists where each inner list contains
                         node values in the same vertical column from top to bottom.

    Example:
        >>> # Binary Tree:
        >>> #       3
        >>> #      / \
        >>> #     9   20
        >>> #         / \
        >>> #        15  7
        >>> vertical_traversal_of_a_binary_tree(root)
        [[9], [3, 15], [20], [7]]

    Time Complexity: O(N log N)
        - N = number of nodes
        - Sorting columns and nodes within columns adds logarithmic overhead.

    Space Complexity: O(N)
        - For the dictionary and BFS queue.
    """
    if not root:
        return []

    d = defaultdict(list)
    queue = deque([(root, 0, 0)])   # (node, level, column)

    while queue:
        current, level, column = queue.popleft()
        d[column].append((level, current.val))
        if current.left:
            queue.append((current.left, level + 1, column - 1))
        if current.right:
            queue.append((current.right, level + 1, column - 1))

    result = []
    for col in sorted(d.keys()):
        col_nodes = sorted(d[col], key=lambda x: (x[0], x[1]))  # sort by level and value
        values = [val for level, val in col_nodes]
        result.append(values)
    return result


def merge_two_sorted_lists(list1, list2):
    """
    Merges two sorted linked lists into a single sorted linked list.

    This function takes two sorted singly linked lists and merges them
    into a single sorted linked list by repeatedly choosing the smaller node
    from the heads of the two lists.

    Args:
        list1 (ListNode): Head of the first sorted linked list.
        list2 (ListNode): Head of the second sorted linked list.

    Returns:
        ListNode: Head of the merged sorted linked list.

    Example:
        Input: list1 = [1,2,4], list2 = [1,3,4]
        Output: [1,1,2,3,4,4]

    Time Complexity: O(m + n)
        - Each node in both lists is visited exactly once.

    Space Complexity: O(1)
        - Merging is done in place using existing nodes (ignoring recursion stack).
    """
    dummy = current = ListNode(0)

    while list1 and list2:
        if list1.val < list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    current.next = list1 or list2
    return dummy.next


def merge_k_sorted_lists(lists):
    # https://leetcode.com/problems/merge-k-sorted-lists
    """
    Merges K sorted linked lists into a single sorted linked list using divide and conquer.

    The algorithm works as follows:
    1. Recursively split the list of linked lists into halves.
    2. Merge each half using the `merge_two_sorted_lists()` function.
    3. Continue merging until only one sorted list remains.

    This approach reduces the problem from O(kN) to O(N log k)
    by balancing the merge operations across levels of recursion.

    Args:
        lists (List[ListNode]): A list containing the heads of k sorted linked lists.

    Returns:
        ListNode: Head of the merged sorted linked list, or None if the input list is empty.

    Example:
        Input: lists = [[1,4,5],[1,3,4],[2,6]]
        Output: [1,1,2,3,4,4,5,6]

    Time Complexity: O(N log k)
        - N is the total number of nodes across all lists.
        - log k levels of merging, each processing all N nodes once.

    Space Complexity: O(log k)
        - Due to recursion depth in divide and conquer.

    References:
        - https://leetcode.com/problems/merge-k-sorted-lists/
    """

    if not lists:
        return
    if len(lists) == 1:
        return lists[0]

    mid = len(lists) // 2
    left = merge_k_sorted_lists(lists[: mid])
    right = merge_k_sorted_lists(lists[mid: ])

    return merge_two_sorted_lists(left, right)


def making_a_large_island_1(grid):
    """
    https://leetcode.com/problems/making-a-large-island/

    Given an n x n binary grid where 1 represents land and 0 represents water,
    you can change at most one 0 to 1. Return the size of the largest island
    possible after making at most one change.

    This version uses BFS instead of DFS to label islands and calculate their sizes.

    Args:
        grid (List[List[int]]): n x n binary matrix.

    Returns:
        int: The largest possible island size after flipping at most one 0.

    Time Complexity:
        O(n^2) – Each cell is processed at most twice (once during labeling, once during checking).
    Space Complexity:
        O(n^2) – For BFS queue, visited cells, and island area storage.
    """
    n = len(grid)
    area = {}  # Map: island_id → area
    island_id = 2  # Unique IDs starting from 2 (since 0 = water, 1 = unvisited land)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # --- Step 1: BFS to label islands and calculate their areas ---
    def bfs(r, c, island_id):
        queue = deque([(r, c)])
        grid[r][c] = island_id
        size = 1

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
                    grid[nx][ny] = island_id
                    size += 1
                    queue.append((nx, ny))
        return size

    # Label all islands
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                area[island_id] = bfs(i, j, island_id)
                island_id += 1

    # If all are 1s → return the whole grid area
    if not area:
        return 1
    max_area = max(area.values())

    # --- Step 2: Try flipping each 0 and compute possible island size ---
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0:
                seen = set()
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] > 1:
                        seen.add(grid[ni][nj])
                new_area = 1  # The flipped cell
                for id_ in seen:
                    new_area += area[id_]
                max_area = max(max_area, new_area)

    return max_area


def making_a_large_island_2(grid):
    """
    https://leetcode.com/problems/making-a-large-island/

    Given an n x n binary grid representing water (0) and land (1),
    you can change at most one 0 to 1 and must return the size of the
    largest possible island (connected 1's).

    The solution labels each island with a unique ID using DFS and
    stores their areas. Then, for each 0, it computes the potential
    island size if that cell were turned to 1.

    Args:
        grid (List[List[int]]): n x n grid of 0s and 1s.

    Returns:
        int: The largest possible island size after flipping one 0.

    Time Complexity:
        O(n^2)
        - Each cell is visited at most twice (once for labeling and once for checking neighbors).

    Space Complexity:
        O(n^2)
        - For recursion stack and the area dictionary.

    Example:
        Input:
        [[1, 0],
         [0, 1]]
        Output: 3
        Explanation:
        Change one 0 to 1 to connect the two islands -> size 3.
    """
    n = len(grid)
    area = {}
    island_id = 2  # Start from 2 to distinguish from 0 and 1

    # DFS to label island and return its area
    def dfs(r, c, id_):
        if r < 0 or c < 0 or r >= n or c >= n or grid[r][c] != 1:
            return 0
        grid[r][c] = id_
        return 1 + dfs(r + 1, c, id_) + dfs(r - 1, c, id_) + dfs(r, c + 1, id_) + dfs(r, c - 1, id_)

    # Step 1: Label all islands and record their area
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                area[island_id] = dfs(r, c, island_id)
                island_id += 1

    max_area = max(area.values(), default=0)

    # Step 2: Try converting each 0 to 1
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 0:
                seen = set()
                new_area = 1  # This flipped 0
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] > 1:
                        seen.add(grid[nr][nc])
                for id_ in seen:
                    new_area += area[id_]
                max_area = max(max_area, new_area)

    return max_area


def minimum_window_substring(s: str, t: str) -> str:
    """
    https://leetcode.com/problems/minimum-window-substring/

    Finds the smallest substring of s that contains all characters of t.

    Args:
        s (str): The main string.
        t (str): The target string with required characters.

    Returns:
        str: The minimum window substring, or "" if no valid window exists.

    Time Complexity:
        O(|s| + |t|) — Each character is visited at most twice (expand and contract).
    Space Complexity:
        O(|s| + |t|) — For frequency counters.
    """

    if not s or not t:
        return ""

    t_count = Counter(t)        # Frequency of required chars
    required = len(t_count)     # Number of unique required chars

    window_count = {}           # Frequency in current window
    formed = 0                  # How many required chars are currently satisfied

    l = 0
    res = (float('inf'), 0, 0)  # (window length, left, right)

    # Expand window
    for r, ch in enumerate(s):
        window_count[ch] = window_count.get(ch, 0) + 1

        # Check if current char satisfies target count
        if ch in t_count and window_count[ch] == t_count[ch]:
            formed += 1

        # Contract window while valid
        while l <= r and formed == required:
            if (r - l + 1) < res[0]:
                res = (r - l + 1, l, r)

            # Remove from window
            left_ch = s[l]
            window_count[left_ch] -= 1
            if left_ch in t_count and window_count[left_ch] < t_count[left_ch]:
                formed -= 1
            l += 1

    return "" if res[0] == float('inf') else s[res[1]:res[2] + 1]


def valid_number(s):
    # https://leetcode.com/problems/valid-number
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
        elif char in '0123456789':
            met_digit = True
        else:
            return False
    return met_digit


def sliding_window_median(nums, k):
    balance = 0

    def add_num(num):
        """Add number to one of the heaps and balance."""
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)
        balance_heaps()

    def balance_heaps():
        """Ensure heaps differ in size by at most one."""
        if len(max_heap) >= len(min_heap) + 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))

    def remove_num(num):
        """Mark num for lazy removal and adjust heap sizes."""
        delayed[num] += 1
        nonlocal balance
        if num <= -max_heap[0]:
            balance -= 1  # one less in max_heap
        else:
            balance += 1  # one less in min_heap
        prune(max_heap)
        prune(min_heap)

    def prune(heap):
        """Pop elements that are marked for deletion."""
        while heap and delayed[abs(heap[0])] > 0:
            num = -heapq.heappop(heap) if heap is max_heap else heapq.heappop(heap)
            delayed[num] -= 1

    def get_median():
        """Compute the current median."""
        if k % 2 == 1:
            return float(-max_heap[0])
        else:
            return (-max_heap[0] + min_heap[0]) / 2.0


    # Initialization
    min_heap, max_heap = [], []
    delayed = defaultdict(int)
    medians = []


    # First window
    for i in range(k):
        add_num(nums[i])
    medians.append(get_median())

    # Sliding the window
    for i in range(k, len(nums)):
        add_num(nums[i])          # Add new number
        remove_num(nums[i - k])   # Remove old number
        balance_heaps()           # Rebalance heaps
        medians.append(get_median())

    return medians


def valid_palindrome_iii(s, k):
    # https://algo.monster/liteproblems/1216
    """
    Determines whether a given string `s` can be transformed into a palindrome
    by deleting at most `k` characters.

    This is based on finding the Longest Palindromic Subsequence (LPS) of `s`.
    The minimum number of deletions required to make `s` a palindrome is:
        minDeletions = len(s) - LPS

    If minDeletions <= k, then it's possible to make `s` a palindrome
    by removing at most `k` characters.

    Approach:
        - Compute the LPS using Dynamic Programming.
        - LPS is equivalent to the Longest Common Subsequence (LCS) between
          the string and its reverse.

    Args:
        s (str): The input string.
        k (int): The maximum number of characters allowed to be deleted.

    Returns:
        bool: True if `s` can be made palindrome with ≤ k deletions, else False.

    Example:
        >>> valid_palindrome_iii("abcdeca", 2)
        True
        # Explanation: Delete 'b' and 'd' → "aceca", which is a palindrome.

    Time Complexity:
        O(n²), where n is the length of s.

    Space Complexity:
        O(n²), due to the 2D DP table.
    """

    n = len(s)
    rev = s[::-1]

    # dp[i][j] stores LCS between s[:i] and rev[:j]
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == rev[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lps = dp[n][n]  # Longest Palindromic Subsequence
    return (n - lps) <= k


def robot_room_cleaner(robot):
    # https://algo.monster/liteproblems/489
    """
    Cleans all reachable cells in a room using the given Robot API.
    The robot starts from an unknown location and can move, turn, and clean.

    The room is modeled as an unknown grid of open and blocked cells.
    The robot can only discover the environment through move() calls.

    Approach:
        - Use DFS (Depth-First Search) with backtracking.
        - Keep track of visited cells to avoid re-cleaning or infinite loops.
        - Maintain the robot's orientation (up, right, down, left) manually.
        - After exploring a direction, backtrack to the previous position
          and restore orientation.

    Args:
        robot (Robot): The control interface provided by the system.

    Returns:
        None. Cleans the entire reachable area.

    Directions Mapping:
        0: up    → (-1, 0)
        1: right → (0, 1)
        2: down  → (1, 0)
        3: left  → (0, -1)

    Time Complexity:
        O(N - M), where N is the total number of cells and M is the number of blocked cells.

    Space Complexity:
        O(N - M), for recursion stack and visited set.
    """

    # Set of cleaned/visited cells
    visited = set()

    # Direction deltas: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def backtrack():
        """Moves the robot one step backward and restores the original direction."""
        robot.turnRight()
        robot.turnRight()
        robot.move()
        robot.turnRight()
        robot.turnRight()

    def dfs(x, y, d):
        """
        DFS recursive exploration:
        - Clean current cell.
        - Try all 4 directions (up, right, down, left).
        - Backtrack after exploring each new cell.
        """
        visited.add((x, y))
        robot.clean()

        # Try 4 directions
        for i in range(4):
            new_d = (d + i) % 4
            dx, dy = directions[new_d]
            nx, ny = x + dx, y + dy

            # Move if possible and unvisited
            if (nx, ny) not in visited and robot.move():
                dfs(nx, ny, new_d)
                backtrack()

            # Turn robot to face the next direction
            robot.turnRight()

    dfs(0, 0, 0)


class MedianFinder:
    # https://leetcode.com/problems/find-median-from-data-stream/description/
    """
    A data structure that efficiently supports adding numbers and finding the median
    of all numbers seen so far.

    This is the standard solution for the LeetCode problem:
    https://leetcode.com/problems/find-median-from-data-stream/

    Approach:
        - Maintain two heaps (priority queues):
            1. max_heap: stores the *smaller half* of the numbers (as negative values for max behavior)
            2. min_heap: stores the *larger half* of the numbers

        - The heaps are balanced such that:
            - Either both heaps have the same number of elements, or
            - max_heap has one more element than min_heap

        - The median is:
            - Top of max_heap if it has more elements (odd total count)
            - Average of tops of both heaps if total count is even

    Example:
        >>> mf = MedianFinder()
        >>> mf.add_num(1)
        >>> mf.add_num(2)
        >>> mf.find_median()
        1.5
        >>> mf.add_num(3)
        >>> mf.find_median()
        2.0

    Time Complexity:
        - add_num(): O(log n)
        - find_median(): O(1)

    Space Complexity:
        - O(n), where n is the number of inserted elements.
    """
    def __init__(self):
        # min_heap → holds larger half
        # max_heap → holds smaller half (stored as negatives)
        self.min_heap, self.max_heap = [], []

    def add_num(self, num: int) -> None:
        """
        Add a new number into the data structure.
        Balances both heaps to maintain median property.
        """
        # Push into max_heap (as negative to simulate max-heap)
        heapq.heappush(self.max_heap, -num)

        # Ensure every element in max_heap <= every element in min_heap
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Balance: max_heap can have at most 1 more element than min_heap
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self) -> float:
        """
        Return the median of all numbers added so far.
        """
        # Odd number of elements → top of max_heap
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]

        # Even number of elements → average of both heap tops
        return (-self.max_heap[0] + self.min_heap[0]) / 2


def word_ladder(begin_word, end_word, word_list):
    # https://leetcode.com/problems/word-ladder
    """
    Finds the shortest transformation sequence length from begin_word to end_word.

    Each step can change only one letter, and the resulting word must be
    in the provided word_list.

    Example:
        begin_word = "hit"
        end_word = "cog"
        word_list = ["hot","dot","dog","lot","log","cog"]
        → Output: 5
        Explanation: "hit" → "hot" → "dot" → "dog" → "cog"

    Approach:
        - BFS (Breadth-First Search) to ensure the shortest path.
        - Preprocess all words into intermediate patterns like "h*t" or "ho*"
          to quickly find neighboring words differing by one letter.
        - Use a queue to track words and their distance levels.

    Args:
        begin_word (str): Starting word.
        end_word (str): Target word to reach.
        word_list (List[str]): List of allowed transformations.

    Returns:
        int: The length of the shortest transformation sequence, or 0 if none exists.
    """
    if end_word not in word_list or not begin_word or not end_word or not word_list or len(begin_word) != len(end_word):
        return 0

    n, hashmap = len(begin_word), defaultdict(list)
    for word in word_list:
        for i in range(n):
            intermediate_word = word[:i] + '*' + word[i + 1:]
            hashmap[intermediate_word].append(word)

    queue, visited = deque([(begin_word, 1)]), {begin_word}  # This is a set

    while queue:
        current_word, level = queue.popleft()
        for i in range(n):
            intermediate_word = current_word[:i] + '*' + current_word[i + 1:]
            for word in hashmap[intermediate_word]:
                if word == end_word:
                    return level + 1
                if word not in visited:
                    visited.add(word)
                    queue.append((word, level + 1))
    return 0
