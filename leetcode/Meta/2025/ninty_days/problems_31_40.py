from functools import cache
from collections import deque, defaultdict


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
    """
    Mountain Trekking Analogy:
    Imagine stock prices as daily mountain altitudes. Each day you're at a new altitude.
    GOAL: Find the best single climb - buy in a valley (low), sell at a later peak (high).
    The Secret: Track your CLIMBING STREAKS, not just valleys and peaks!

    How the Hiker Thinks:
    --------------------
    Instead of memorizing every altitude, the hiker tracks:
    1. Current climbing streak: How much altitude I've gained in my current climb
    2. Best climb ever: The most altitude I've gained in any single climb

    At Each Day's Climb (price difference):
    ---------------------------------------
    The hiker checks: "How does today's climb affect my current streak?"

    Today's climb = Today's altitude - Yesterday's altitude
    - Uphill (positive): Good for my streak!
    - Downhill (negative): Bad for my streak...

    Decision Rule:
    "If today's climb would make my current streak negative, I'm better off starting a fresh climb from here."

    Then: "Is this my best climb ever?"

    Hiker's Logbook (Example Trek):
    -------------------------------
    Altitudes (stock prices): [7, 1, 5, 3, 6, 4]

    Day 1: Altitude 7 → No climb yet (no previous day)
    Day 2: Altitude 1 → Climb = 1-7 = -6 (downhill!)
              Streak: max(0, 0 - 6) = 0 → Abandon climb, stay at base camp
              Best climb ever: 0

    Day 3: Altitude 5 → Climb = 5-1 = +4 (uphill!)
              Streak: max(0, 0 + 4) = 4 → Start new climb, gained 4 units
              Best climb ever: max(0, 4) = 4

    Day 4: Altitude 3 → Climb = 3-5 = -2 (downhill)
              Streak: max(0, 4 - 2) = 2 → Continue climb (now 2 units total)
              Best climb ever: max(4, 2) = 4

    Day 5: Altitude 6 → Climb = 6-3 = +3 (uphill!)
              Streak: max(0, 2 + 3) = 5 → Continue climb (now 5 units total)
              Best climb ever: max(4, 5) = 5

    Day 6: Altitude 4 → Climb = 4-6 = -2 (downhill)
              Streak: max(0, 5 - 2) = 3 → Continue climb (now 3 units total)
              Best climb ever: max(5, 3) = 5

    Best climb found: 5 units altitude gain
    Translation: Buy at altitude 1, sell at altitude 6 = profit of 5

    Why This Works for Stocks:
    --------------------------
    Every profitable buy-sell pair = a continuous uphill climb
    The maximum profit = the longest continuous uphill climb in the mountain
    By tracking climbing streaks, we find the best time to buy (start of climb) and sell (end of climb) automatically!

    Time: O(n) - one trek through the mountain
    Space: O(1) - only a logbook (2 variables)
    """
    max_profit = current_profit = 0
    price_diffs = [b - a for a, b in zip(prices, prices[1:])]
    for price_diff in price_diffs:
        current_profit = max(0, current_profit + price_diff)  # Either start fresh or extend the previous subarray.
        max_profit = max(max_profit, current_profit)
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
    result = [float('-inf')]

    def dfs(node):
        if not node:
            return 0

        left, right = max(dfs(node.left), 0),  max(dfs(node.right), 0)
        result[0] = max(result[0], left + right + node.val)
        return max(left + node.val, right + node.val)

    dfs(root)
    return result[0]


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
    if not begin_word or not end_word or not word_list or len(begin_word) != len(end_word) or begin_word not in word_list or end_word not in word_list:
        return 0

    n = len(begin_word)
    hashmap = defaultdict(list)
    for word in word_list:
        for i in range(n):
            intermediate_word = word[:i] + "*" + word[i + 1:]
            hashmap[intermediate_word].append(word)

    queue, visited = deque([(begin_word, 1)]), {begin_word}
    while queue:
        current_word, level = queue.popleft()
        for i in range(n):
            intermediate_word = current_word[:i] + "*" + current_word[i + 1:]
            for word in hashmap[intermediate_word]:
                if word == end_word:
                    return level + 1
                if word not in visited:
                    visited.add(word)
                    queue.append((word, level + 1))
    return 0


def sum_root_to_leaf_numbers(root):
    # https://leetcode.com/problems/sum-root-to-leaf-numbers
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


def word_break(s, word_dict):
    # https://leetcode.com/problems/word-break/description/
    """
    queue = deque([0])
    visited = {0}
    word_set = set(word_dict)
    n =  len(s)

    while len(queue) > 0:
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
    word_set, n = set(word_dict), len(s)

    @cache
    def backtrack(k):
        if k == n:
            return True
        for i in range(k, n):
            chosen = s[k: i + 1]
            if chosen in word_set and backtrack(i + 1):
                return True
        return False

    return backtrack(k=0)