import random
from collections import deque, defaultdict
from leetcode.utils import DLLNode, TreeNode


class LRUCache1:
    """This solution is by using dictionary and in python 3.6 and above, dict is ordered by default."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return -1

    def put(self, key, value):
        if key in self.cache:       # For put, check if the key is there
            self.cache.pop(key)

        if len(self.cache) >= self.capacity:
            self.cache.pop(next(iter(self.cache)))  # For ordered dict use self.cache.popitem(last=False)
            # oldest_key = next(iter(self.cache))
        self.cache[key] = value


class LRUCache2:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = []  # list of (key, value) pairs

    def get(self, key: int) -> int:
        # Search linearly (O(n))
        for i, (k, v) in enumerate(self.cache):
            if k == key:
                # Move to most recently used (end of list)
                self.cache.append(self.cache.pop(i))
                return v
        return -1  # not found

    def put(self, key: int, value: int) -> None:
        # Check if key already exists
        for i, (k, v) in enumerate(self.cache):
            if k == key:
                # Update value and move to MRU
                self.cache.pop(i)
                self.cache.append((key, value))
                return

        # If at capacity, evict least recently used (front of list)
        if len(self.cache) == self.capacity:
            self.cache.pop(0)   # As we are using append for adding keys, key at position 0 becomes least recently used

        # Insert new key-value at MRU position
        self.cache.append((key, value))


class LRUCache3:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}                         # key -> node
        self.head = DLLNode(0, 0)       # dummy head
        self.tail = DLLNode(0, 0)       # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev, nxt = node.prev, node.next        # As we are removing, we get prev and next from the node
        prev.next, nxt.prev = nxt, prev

    def _add(self, node):
        # We need to add new node right before the last node at the end of the tail as the last node is the dummy node
        prev, nxt = self.tail.prev, self.tail   # As we are adding, we get prev and next from the tail
        prev.next = nxt.prev = node
        node.prev, node.next = prev, nxt

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])

        node = DLLNode(key, value)
        self._add(node)
        self.cache[key] = node

        if len(self.cache) >= self.capacity:
            lru = self.head.next  # Remove node from the next to the head as head node is a dummy node.
            self._remove(lru)
            del self.cache[lru.key]


def shortest_path_in_binary_matrix(grid):
    # https://leetcode.com/problems/shortest-path-in-binary-matrix
    n = len(grid)

    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1

    directions = [(-1, 0), (-1, 1), (1, 1), (0, 1), (1, -1), (-1, 0), (-1, -1)]
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    visited = {(0, 0)}          # This is a set with (row, col) tuple

    while queue:
        x, y, distance = queue.popleft()
        if x == n - 1 and y == n - 1:
            return distance

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, distance + 1))

    return -1


def valid_palindrome_iii(s: str, k: int):
    # https://algo.monster/liteproblems/1216
    """
    Palace Guard Formation Analogy:
    Imagine you're the royal commander with soldiers (letters) lined up.
    Each soldier has a letter on their shield, forming the word 's'.

    GOAL: Check if you can rearrange/remove at most 'k' soldiers to form a symmetric formation (palindrome).

    The Challenge: You can only REMOVE soldiers (delete letters), not rearrange them!

    Secret Strategy: Find the LONGEST PALINDROMIC SUBSEQUENCE (LPS)
    Why? Because if you keep the longest palindrome that's already in the formation, you only need to remove the rest!

    Formula:
        Minimum removals needed = Total soldiers - LPS length
        If this ≤ k → Can form a palindrome by removing ≤ k soldiers!

    How to Find LPS:
    ----------------
    1. The Formation is your string 's' (n soldiers)
    2. Use Dynamic Programming to find LPS length

    DP Approach (Royal Table):
    -------------------------
    dp[i][j] = length of LPS in substring s[i..j]

    Base cases:
      - Single soldier: dp[i][i] = 1 (a soldier alone is symmetric)
      - Empty formation: dp[i][j] = 0 when i > j

    Recurrence:
      - If soldiers at positions i and j have same letter:
            dp[i][j] = 2 + dp[i+1][j-1]  # Include both, check middle
      - Otherwise, pick the better of two options:
            dp[i][j] = max(dp[i+1][j], dp[i][j-1])  # Skip left or right

    Fill the table from smaller formations to larger ones!

    Example: "abcdeca", k = 2
    -------------------------
    Soldiers: a b c d e c a
    LPS: a c d c a (length 5)
    Total soldiers: 7
    Removals needed: 7 - 5 = 2 ≤ 2 → TRUE!

    Why This Works:
    ---------------
    The LPS is the largest palindrome you can KEEP by only removing soldiers.
    All other soldiers outside the LPS must be removed.
    So the minimum removals = n - LPS length.

    Time Complexity: O(n²) - filling an n×n table
    Space Complexity: O(n²) - can be optimized to O(n) if needed
    """

    n = len(s)

    # Royal Table to track LPS lengths
    dp = [[0] * n for _ in range(n)]

    # Base case: Single soldiers are symmetric
    for i in range(n):
        dp[i][i] = 1  # updates along the diagonal

    # Build from smaller formations to larger ones
    # 'length' represents formation size
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j]:
                if length == 2:
                    # Two soldiers with same letter form a pair
                    dp[i][j] = 2
                else:
                    # Include both soldiers, plus what's in between
                    dp[i][j] = 2 + dp[i + 1][j - 1]
            else:
                # Take the better of removing left or right soldier
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    # LPS length is dp[0][n-1] (entire formation)
    lps_length = dp[0][n - 1]

    # Minimum removals needed
    min_removals = n - lps_length

    return min_removals <= k


def find_peak_element(nums):
    # https://leetcode.com/problems/find-peak-element
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2
        if nums[mid + 1] >= nums[mid]:
            left = mid + 1
        else:
            right = mid
    return left


class BinarySearchIterator:
    # https://leetcode.com/problems/binary-search-tree-iterator/
    """
    You need to put left and right side plates in a cylinder stack with an opening at the top.
    Think like a blind person with left arm cut off, and you have stacks of plates on left and right side.
    - First you need to put plates from left side in the cylinder using your mechanical arm (which is _push_left method)
    - When ever someone asks plate from the cylinder stack,
        - You take the plate first
        - You put all your right side plates using your left mechanical arm
        - Then give them the plate
    """

    def __init__(self, root):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self):
        if self.has_next():
            node = self.stack.pop()
            self._push_left(node.right)
            return node.val

    def has_next(self):
        return len(self.stack) > 0


def binary_tree_right_side_view(root: TreeNode):
    # https://leetcode.com/problems/binary-tree-right-side-view
    # Always remember for binary trees.
    # If there is a level wise operation, use 2 loops:
    #   - 1st loop is with "while"
    #   - 2nd loop is with "for"
    result = []

    if not root:
        return result

    queue = deque([root])

    while queue:
        for i in range(len(queue)):
            current = queue.popleft()
            if i == 0:
                result.append(current.val)

            if current.right:
                queue.append(current.right)
            if current.left:
                queue.append(current.left)

    return result


def number_of_islands(grid):
    # https://leetcode.com/problems/number-of-islands
    m, n = len(grid), len(grid[0])
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def dfs(x, y):
        if 0 <= x < m and 0 <= y < n and grid[x][y] == "1": # Find a valid piece of land "1"
            grid[x][y] = 0                                  # Set it to water "0" so that we don't recount it again
            for dx, dy in directions:
                nx, ny = x + dx, y + dx
                dfs(nx, ny)

    result = 0

    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1":
                dfs(i, j)
                result += 1

    return result


def course_schedule(num_courses, pre_requisites):
    # https://leetcode.com/problems/course-schedule
    """
    Solves the Course Schedule problem using topological sorting (Kahn's Algorithm).

    - Each course is a node labeled from 0 to num_courses - 1.
    - A prerequisite pair (curr, prev) means:
        prev → curr (you must take `prev` before `curr`).

    We build:
    - A graph (adjacency list) mapping each course to the courses that depend on it.
    - An in_degree array where in_degree[i] is the number of prerequisites for course i.

    Strategy:
    - Start with all courses that have in_degree 0 (no prerequisites).
    - Repeatedly take such courses, and reduce the in_degree of their neighbors.
    - If all courses can be taken, the schedule is possible.
    """
    graph = defaultdict(list)
    in_degree = [0] * num_courses

    for curr, prev in pre_requisites:
        graph[prev].append(curr)
        in_degree[curr] += 1

    queue = deque([course_id for course_id in range(num_courses) if in_degree[course_id] == 0])
    taken = 0

    while queue:
        curr_course = queue.popleft()
        taken += 1
        for next_course in graph[curr_course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return taken == num_courses


def kth_largest_element_simplified(nums, k):
    # https://leetcode.com/problems/kth-largest-element-in-an-array
    """
    Treasure Hunter's QuickSelect (Simplified):

    1. Choose random guide pile (better for avoiding worst case)
    2. Partition into smaller piles (left) and larger piles (right)
    3. Recursively search only the relevant section

    Memory Hook: "Pick, Partition, Prune"
    """

    def quick_select(arr, k_smallest):
        """Find kth smallest element (0-based)"""
        if len(arr) == 1:
            return arr[0]

        # Pick random guide
        guide_idx = random.randint(0, len(arr) - 1)
        guide = arr[guide_idx]

        # Partition into three piles:
        smaller = [x for x in arr if x < guide]
        equal = [x for x in arr if x == guide]
        larger = [x for x in arr if x > guide]

        # Prune search space:
        if k_smallest < len(smaller):
            return quick_select(smaller, k_smallest)
        elif k_smallest < len(smaller) + len(equal):
            return guide  # Guide is the kth smallest
        else:
            return quick_select(larger, k_smallest - len(smaller) - len(equal))

    # Convert kth largest to kth smallest
    k = len(nums) - k
    return quick_select(nums, k)


def contains_duplicate_ii(nums, k):
    # https://leetcode.com/problems/contains-duplicate-ii
    """
    For each number, the only index that can possibly form a valid pair with the current index is the most recent one.
    This is because of the condition <= k
    Checking anything other than the last index is wasted effort.
    """
    seen_dict = dict()              # {num: i}

    for i, num in enumerate(nums):
        if num in seen_dict and abs(i - seen_dict[num]) <= k:
            return True
        seen_dict[num] = i          # This is because we only need the last index

    return False
