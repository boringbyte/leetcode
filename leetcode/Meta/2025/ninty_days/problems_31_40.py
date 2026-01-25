from functools import cache
from collections import deque, defaultdict
from leetcode.utils import CloneNode, RandomNode


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
    """This is also similar to Kadane's algorithm and maximum subarray sum problem"""
    max_profit = running_profit = 0
    price_diffs = [b - a for a, b in zip(prices, prices[1:])]
    for price_diff in price_diffs:
        running_profit = max(0, running_profit + price_diff)  # Either start fresh or extend the previous subarray.
        max_profit = max(max_profit, running_profit)
    return max_profit


def binary_tree_maximum_path_sum(root):
    # https://leetcode.com/problems/binary-tree-maximum-path-sum
    """
    Spy Network Analogy:
    Imagine each node in the tree is a spy handler who can:
      1. Pass information UP to their boss (parent)
      2. Combine information from their LEFT and RIGHT subordinates

    Rules:
      - Each spy (node) has a value (positive = intel, negative = risk)
      - A spy can choose to ignore risky subordinates (max(0, ...))
      - A path is a chain of spies passing intel upward

    How it works:

    1. At each spy (node):
       - Ask LEFT subordinate: "What's the best intel you can send me?"
         (If negative, we ignore it - set to 0)
       - Ask RIGHT subordinate: Same question

       Example:
         Current spy value: 5
         Left subordinate reports: 3 (intel)  -> We accept
         Right subordinate reports: -2 (risk) -> We ignore (use 0)

    2. This spy could be the CENTER of the best network:
       - Combine left intel + right intel + own value
       - Update global maximum if better

       Example continued:
         Best network with this spy as center = 3 + 0 + 5 = 8
         Global maximum becomes max(previous, 8)

    3. What does this spy report UP to boss?
       - Can only send ONE chain upward (left OR right, not both)
       - Choose: max(left, right) + own value

       Example continued:
         Report upward: max(3, 0) + 5 = 8
         "Boss, the best intel chain through me is worth 8"

    Key Insight:
      - The global best might NOT go through the boss (root)
      - It could be a local network centered at any spy
      - We return the value of the BEST spy network found

    Time: O(n) - visit each spy once
    Space: O(h) - recursion stack for tree height

    Example Tree: [-10,9,20,null,null,15,7]
            -10
           /   \
          9    20   ← Best network: 15+20+7=42
              /  \
             15   7
    """
    result = float('-inf')

    def dfs(node):
        nonlocal result
        if not node:
            return 0

        left, right = max(dfs(node.left), 0),  max(dfs(node.right), 0)
        result = max(result, left + right + node.val)
        return max(left + node.val, right + node.val)

    dfs(root)
    return result


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


def word_ladder(begin_word, end_word, word_list):
    # https://leetcode.com/problems/word-ladder
    """
    mapping dictionary for
    word_list = ['hot', 'dot', 'dog', 'lot', 'log', 'cog']

    {'*ot': ['hot', 'dot', 'lot'],
     'h*t': ['hot'],
     'ho*': ['hot'],
     'd*t': ['dot'],
     'do*': ['dot', 'dog'],
     '*og': ['dog', 'log', 'cog'],
     'd*g': ['dog'],
     'l*t': ['lot'],
     'lo*': ['lot', 'log'],
     'l*g': ['log'],
     'c*g': ['cog'],
     'co*': ['cog']}
    """
    if begin_word not in word_list or end_word not in word_list:
        return 0

    n = len(begin_word)
    mapping = defaultdict(list)     # {"h*t": ["hot", "hit"], "ho*": ["hot"]}

    for word in word_list:
        for i in range(n):
            intermediate_word = word[:i] + "*" + word[i + 1:]
            mapping[intermediate_word].append(word)

    queue, visited = deque([(begin_word, 1)]), {begin_word}

    while queue:
        current_word, level = queue.popleft()
        for i in range(n):
            intermediate_word = current_word[:i] + "*" + current_word[i + 1:]
            for word in mapping[intermediate_word]:
                if word == end_word:
                    return level + 1
                if word not in visited:
                    visited.add(word)
                    queue.append((word, level + 1))
    return 0


def sum_root_to_leaf_numbers(root):
    # https://leetcode.com/problems/sum-root-to-leaf-numbers
    """
    Treasure Map Analogy:
    Imagine each node is a treasure room containing a digit (0-9).
    Each root-to-leaf path is a unique treasure map code.
    Your mission: Sum up ALL treasure codes in the castle!

    How It Works:
    ------------
    1. Start at the entrance (root) with an empty code (0).

    2. At each room (node):
       - Append this room's digit to your current `code`
         code = code * 10 + room.digit
         - Why x10? Each step left in the castle multiplies the code by 10!*

       - Check if this is a dead-end room (leaf):
         If NO exits (no left/right rooms): Add this code to treasure chest

       - If there are exits (left/right rooms):
         Make copies of your current map and send explorers down each path

    3. Use a stack (backpack) to track:
       - Current room
       - Code built so far

    Time Complexity: O(n) - visit each room once
    Space Complexity: O(h) - stack height = tree height

    Recursive Solution:
    if not root:
        return 0

    result = 0

    def dfs(node, val):
        nonlocal result

        val = val * 10 + node.digit

        if not node.left and not node.right:
            result += val
            return

        if node.left:
            dfs(node.left, val)
        if node.right:
            dfs(node.right, val)

    dfs(root, 0)
    return result
    """
    if not root:
        return 0

    stack, result = [(root, 0)], 0

    while stack:
        current, val = stack.pop()
        val = val * 10 + current.val

        if current.left is None and current.right is None:
            result += val

        if current.left:
            stack.append((current.left, val))
        if current.right:
            stack.append((current.right, val))

    return result


def clone_graph(node):
    # https://leetcode.com/problems/clone-graph
    """
    Ghost Army Analogy:
    Imagine you have an army of ghost soldiers (original graph) who:
      1. Have unique ID numbers (node.val)
      2. Know their fellow soldiers (neighbors)

    Your mission: Create an IDENTICAL ghost army (clone) where:
      - Each original soldier gets a clone with same ID
      - Each clone knows the clones of the original's neighbors

    The Challenge: Soldiers are interconnected in complex ways (cycles)!

    How the Ghost General Works:
    ---------------------------
    1. Keep a "Clone Registry" (hash map) to track:
       Original Soldier -> Clone Soldier

    2. Use a "Summoning Queue" (BFS queue) to process soldiers:
       - Start with the first soldier (node)
       - Create their clone, add to registry
       - Queue them up for neighbor processing

    3. For each soldier in the queue:
       - Look at their original neighbor list
       - For each neighbor:
           * If neighbor not cloned yet -> Create clone, register, queue it
           * Connect current clone to neighbor's clone

    4. Return the clone of the first soldier (the army commander)

    Why BFS? Because we need to process soldiers level by level to
    ensure all connections are properly cloned.

    Example Ghost Army:
    -------------------
    Original:        Clone:
      1 --- 2         1'--- 2'
      |     |         |     |
      4 --- 3         4'--- 3'

    Steps:
    1. Start with soldier 1
       Clone 1' (registry: {1: 1'})
       Queue: [1]

    2. Process soldier 1:
       Neighbors: 2, 4
       - Clone 2' (registry: {1: 1', 2: 2'}, queue: [1, 2])
       - Clone 4' (registry: {1: 1', 2: 2', 4: 4'}, queue: [1, 2, 4])
       Connect: 1'.neighbors = [2', 4']

    3. Process soldier 2:
       Neighbors: 1, 3
       - 1 already cloned -> get 1' from registry
       - Clone 3' (registry: {1: 1', 2: 2', 4: 4', 3: 3'}, queue: [1, 2, 4, 3])
       Connect: 2'.neighbors = [1', 3']

    4. Process soldier 4:
       Neighbors: 1, 3
       - Both already cloned (1' and 3')
       Connect: 4'.neighbors = [1', 3']

    5. Process soldier 3:
       Neighbors: 2, 4
       - Both already cloned (2' and 4')
       Connect: 3'.neighbors = [2', 4']

    Done! Return 1' (clone commander)

    Time: O(V + E) - visit each soldier and each connection once
    Space: O(V) - for the clone registry
    """

    if not node:
        return None

    # Step 1: Clone Registry: Original Soldier -> Clone Soldier and Create the first clone (army commander)
    clone_registry = {node: CloneNode(node.val, [])}

    # Step 2: Summoning Queue: Soldiers waiting to have their neighbors processed
    queue = deque([node])

    while queue:
        original = queue.popleft()
        clone = clone_registry[original]

        # Process each neighbor of the original soldier
        for neighbor in original.neighbors:
            if neighbor not in clone_registry:
                # First time seeing this soldier -> create their clone
                clone_registry[neighbor] = CloneNode(neighbor.val, [])      # same as step 1
                queue.append(neighbor)                                               # same as step 2

            # Connect the clone to the neighbor's clone
            clone.neighbors.append(clone_registry[neighbor])

    return clone_registry[node]


def copy_list_with_random_pointer(head):
    # https://leetcode.com/problems/copy-list-with-random-pointer
    """
    How the Agency Clones Nodes:
    ---------------------------
    1. The "Twin Insertion" Phase (Weaving):
       For each original node, create a clone and insert it RIGHT AFTER the original.

       Original: A -> B -> C -> null
       After:    A -> A' -> B -> B' -> C -> C' -> null

    2. The "Secret Connection" Phase:
       For each original node A, set A'.random to:
       - A.random.next (the clone of A's random contact)
       - null if A.random is null

    3. The "Network Split" Phase:
       Unweave the interleaved list into two separate networks:
       Original: A -> B -> C -> null
       Clone:    A' -> B' -> C' -> null

    Visual Example:
    ---------------
    Original:
        A(1) -> B(2) -> C(3) -> null
        |      |      |
        C      A      B   (random pointers)

    Step 1: Twin Insertion
        A -> A' -> B -> B' -> C -> C' -> null

    Step 2: Set Random Pointers
        For A: A'.random = A.random.next = C'
        For B: B'.random = B.random.next = A'
        For C: C'.random = C.random.next = B'

    Step 3: Split Networks
        Original: A -> B -> C -> null
        Clone:    A' -> B' -> C' -> null
        (with correct random pointers in clone)

    Why This Works:
    --------------
    By placing clones next to originals, we create a direct mapping:
      Original -> Clone is just: original.next (in the interleaved list)

    So: clone.random = original.random.next

    Time: O(n) - three passes through the list
    Space: O(1) - no extra dictionary, only modifies the list structure temporarily
    """

    if not head:
        return None

    # Step 1: Twin Insertion (Weave clones into the list)
    current = head
    while current:
        # Create clone
        clone = RandomNode(current.val)

        # Insert clone right after original
        clone.next = current.next
        current.next = clone

        # Move to next original
        current = clone.next

    # Step 2: Set Random Pointers in clones
    current = head
    while current:
        clone = current.next

        # Set clone's random pointer
        if current.random:
            clone.random = current.random.next
        else:
            clone.random = None

        # Move to next original
        current = clone.next

    # Step 3: Split the interleaved list
    current = head
    clone_head = head.next  # Save head of clone list

    while current:
        clone = current.next

        # Restore original's next pointer
        current.next = clone.next

        # Set clone's next pointer
        if clone.next:
            clone.next = clone.next.next
        else:
            clone.next = None

        # Move to next original
        current = current.next

    return clone_head


def word_break(s, word_dict):
    # https://leetcode.com/problems/word-break
    word_set = set(word_dict)  # Convert to set for O(1) lookups
    n = len(s)

    @cache
    def backtrack(start):
        if start == n:
            return True

        for i in range(start, n):
            chosen_piece = s[start:i + 1]
            if chosen_piece in word_set and backtrack(i + 1):
                return True

        return False

    return backtrack(start=0)

    """    
    # BFS solution
    word_set = set(word_dict)
    n =  len(s)
    
    queue = deque([0])
    visited = {0}

    while queue:
        current = queue.popleft()
        for i in range(current + 1, n + 1):
            if i in visited:
                continue
            if s[current: i] in word_set:
                if i == n:
                    return True
                queue.append(i)
                visited.add(i)
    return False
    """
