import heapq


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
