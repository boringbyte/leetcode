import heapq
from collections import defaultdict, deque, Counter


def basic_calculator_ii(s):
    # https://leetcode.com/problems/basic-calculator-ii
    # https://leetcode.com/problems/basic-calculator-ii/discuss/658480/Python-Basic-Calculator-I-II-III-easy-solution-detailed-explanation
    def update_stack(op, value):
        if op == "+": stack.append(value)
        if op == "-": stack.append(-value)
        if op == "*": stack.append(stack.pop() * value)
        if op == "/": stack.append(int(stack.pop()) / value)

    s = s.strip()
    i, num, stack, sign = 0, 0, [], "+"

    while i < len(s):
        if s[i].isdigit():
            num = num * 10 + int(s[i])
        elif s[i] in "+-*/":
            update_stack(sign, num)
            num, sign = 0, s[i]
        elif s[i] == "(":
            num, j = basic_calculator_ii(s[i + 1:])
            i = i + j
        elif s[i] == ")":
            update_stack(sign, num)
            return sum(stack), i + 1
        i += 1

    update_stack(sign, num)
    return sum(stack)


def lowest_common_ancestor_of_a_binary_search_tree(root, p, q):
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree
    # Below solution can be applied to this problem but not be the other way around
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root


def lowest_common_ancestor_of_a_binary_tree(root, p, q):
    # https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree
    """
     Following are the constraints:
         - All node.val are unique
         - p != q
         - p and q will exist in the tree
     Create a child parent dictionary which tracks the relationship in the reverser order of a tree
     Track the path of `p` node from it to the root node and save this path to set.
     Track the parth of `q` node from it till it doesn't encounter node in the set
     """
    child_parent_dict, p_ancestors = dict(), set()

    def dfs(child, parent):
        if child:
            child_parent_dict[child] = parent
            dfs(child.left, child)
            dfs(child.right, child)

    dfs(root, None)

    while p:
        p_ancestors.add(p)
        p = child_parent_dict[p]

    while q not in p_ancestors:
        q = child_parent_dict[q]

    return q


def product_of_array_except_self(nums):
    # https://leetcode.com/problems/product-of-array-except-self
    n = len(nums)
    prefix, suffix = [1] * n, [1] * n

    for i in range(1, n):
        prefix[i] = prefix[i - 1] * nums[i - 1]

    for i in range(n - 2, -1, -1):
        suffix[i] = suffix[i + 1] * nums[i + 1]

    return [p * s for p, s in zip(prefix, suffix)]


def meeting_rooms_ii(intervals):
    # https://leetcode.com/problems/meeting-rooms-ii
    # https://neetcode.io/problems/meeting-schedule-ii/question
    """
    Airport Gate Scheduling Analogy:

    Imagine each meeting is a flight that needs a gate (room).
    Flights have scheduled arrival (start) and departure (end) times.

    GOAL: Find the minimum number of gates needed so no flight waits for a gate.

    How the Airport Controller Works:
    --------------------------------
    1. Sort all flights by arrival time (when they need a gate).
    2. Use a "Gate Availability Board" (min-heap) that tracks:
       - When each currently occupied gate will be free (departure times)
       - The top of the heap = gate that becomes free the EARLIEST

    3. For each incoming flight:
       a. Check: Can the earliest-freeing gate accommodate this flight?
          (Is the flight's arrival time ≥ gate's free time?)
       b. If YES: Reuse that gate (pop from heap, push new departure time)
       c. If NO: Assign a NEW gate (just push new departure time)

    4. The answer = maximum number of gates used simultaneously during the day.
    """
    if not intervals:
        return 0

    intervals = sorted(intervals, key=lambda x: x[0])
    min_heap = []

    heapq.heappush(min_heap, intervals[0][1])

    for i in range(1, len(intervals)):
        start, end = intervals[i]
        if start >= min_heap[0]:
            heapq.heappop(min_heap)
        heapq.heappush(min_heap, end)
    return len(min_heap)


def alien_dictionary(words):
    # https://leetcode.com/problems/alien-dictionary
    # https://medium.com/@timothyhuang514/graph-alien-dictionary-d2b104c36d8e
    if not words:
        return ""

    # Step 1: Collect all unique letters
    all_unique_letters = set()
    for word in words:
        all_unique_letters.update(word)

    # Step 2: Build graph from adjacent word comparisons
    graph = defaultdict(set)            # letter -> set of letters that come after it
    in_degree = Counter()               # letter -> how many letters come before it

    # Initialize in_degree for all letters
    for letter in all_unique_letters:
        in_degree[letter] = 0

    # Compare adjacent words to find ordering clues
    for word1, word2 in zip(words, words[1:]):
        for char1, char2 in zip(word1, word2):
            if char1 != char2:
                if char2 not in graph[char1]:  # graph[char1] returns a set
                    graph[char1].add(char2)
                    in_degree[char2] += 1
                break
        else:   # Check that second word isn't a prefix of first word.
            if len(word1) > len(word2):
                return ""

    # Step 3: Kahn's Algorithm for topological sort
    queue = deque()

    # Add all letters with in_degree 0 (no prerequisites)
    for letter in in_degree:
        if in_degree[letter] == 0:
            queue.append(letter)

    result = []

    while queue:
        current = queue.popleft()
        result.append(current)

        # Decrease in_degree of neighbors
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Step 4: Check for cycle (if result doesn't contain all letters)
    if len(result) != len(all_unique_letters):
        return ""  # Cycle detected or insufficient clues

    return "".join(result)


def closest_binary_search_tree_value(root, target):
    # https://leetcode.com/problems/closest-binary-search-tree-value
    # https://algo.monster/liteproblems/270
    # https://www.geeksforgeeks.org/dsa/find-closest-element-binary-search-tree/
    """
        Example:
        >>> # BST:      4    # compare abs(4 - target) < abs(closest - target)
        >>> #          / \
        >>> #         2   5
        >>> #        / \
        >>> #       1   3
        >>> closest_binary_search_tree_value(root, 3.714)
        4
    """
    closest = float("inf")
    while root:
        if abs(root.val - target) < abs(closest - target):
            closest = root.val
        if target < root.val:
            root = root.left
        elif target > root.val:
            root = root.right
        else:
            break

    return closest


def expression_add_operators(num, target):
    # https://leetcode.com/problems/expression-add-operators/description/
    result, n = [], len(num)

    def backtrack(sofar, total, prev, k):
        if k == n:
            if total == target:  # If we're exactly at the end and our total equals the target, we found a valid expression and add it to result.
                result.append(sofar)
            return
        for i in range(k, n):
            # Skip numbers with leading zeros (except the number 0 itself)
            # Example: "105" -> can form "1+05" is invalid, "10+5" is valid
            if i > k and num[k] == '0':
                break
            chosen = int(num[k: i + 1])
            if k == 0:  # Special case: first number has no operator before it
                backtrack(sofar + str(chosen), total + chosen, chosen, i + 1)
            else:
                backtrack(sofar + '+' + str(chosen), total + chosen, chosen, i + 1)
                backtrack(sofar + '-' + str(chosen), total - chosen, -chosen, i + 1)
                backtrack(sofar + '*' + str(chosen), total - prev + prev * chosen, prev * chosen, i + 1)

    backtrack(sofar='', total=0, prev=0, k=0)
    return result


def move_zeros(nums):
    # https://leetcode.com/problems/binary-tree-vertical-order-traversal
    snow_ball_size = 0  # How much snow (zeros) collected so far

    for i in range(len(nums)):
        if nums[i] == 0:
            # Found snow - just increase plow size
            snow_ball_size += 1
        else:
            # Found house - swap with first piece of snow in plow
            # First snow is at position i - snowball_size
            nums[i - snow_ball_size], nums[i] = nums[i], nums[i - snow_ball_size]


def binary_tree_vertical_order_traversal_1(root):
    # https://algo.monster/liteproblems/314
    # https://leetcode.com/problems/binary-tree-vertical-order-traversal
    # In BFS, we automatically follow order with respect to the level. So there is no need to track it using another variable.
    if root is None:
        return
    result = []
    queue, column_dict = deque([(root, 0)]), defaultdict(list)   # deque holds (node, column)
    while queue:
        current, column = queue.popleft()
        column_dict[column].append(current.val)
        if current.left:
            queue.append((current.left, column - 1))
        if current.right:
            queue.append((current.right, column + 1))
    for key in sorted(column_dict.keys()):
        result.append(column_dict[key])


def binary_tree_vertical_order_traversal_2(root):
    # DFS does not automatically follow order with respect to the level. So there is a need to track it using another variable.
    if root is None:
        return
    result = []
    column_dict = defaultdict(list)

    def dfs(node, depth, column):
        if node:
            column_dict[column].append((depth, node.val))  # key: value -> column: [(depth, node.val)]
            dfs(node.left, depth + 1, column - 1)
            dfs(node.right, depth + 1, column + 1)

    dfs(root, 0, 0)

    for column in sorted(column_dict.keys()):
        depth_value_tuples = column_dict[column]
        depth_value_tuples.sort(key=lambda x: x[0])
        result.append([val for depth, val in depth_value_tuples])
    return result
