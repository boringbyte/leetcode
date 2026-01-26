from collections import defaultdict


def continuous_subarray_sum(nums, k):
    # https://leetcode.com/problems/continuous-subarray-sum
    """
    Checks if the array contains a continuous subarray of length ≥ 2 whose sum is divisible by k.

    ANALOGY: THE 24-HOUR DINER CLOCK

    Imagine a diner with a k-hour clock (0 to k-1). Each order moves the clock hand forward
    by the order amount (mod k). If you see the SAME CLOCK TIME twice, the total bills
    between those visits must be a multiple of $k.

    WHY IT WORKS:
    Same remainder at times i and j → (prefix[j] - prefix[i]) % k == 0
    → Subarray nums[i+1:j+1] sums to n×k

    EXAMPLE with k=6 (6-hour clock):
        Orders: [23, 2, 4, 6, 7]
        Time 0: 23 → clock at 5  (23 % 6 = 5)
        Time 1: +2 → clock at 1  (25 % 6 = 1)
        Time 2: +4 → clock at 5  (29 % 6 = 5) ← Same as time 0

        Between times 0 and 2: orders [2, 4] = $6 = 1×$6 ✓

    ALGORITHM:
    1. Track remainders in a map: {remainder: first_index_seen}
    2. Initialize with {0: -1} for subarrays starting at index 0
    3. For each number:
        - Update running total and remainder (mod k)
        - If remainder seen before and distance ≥ 2 → return True
        - Otherwise store first occurrence

    Memory Aid: "Same clock time, different hour → bills between are k's multiplier!"

    Args:
        nums: List[int] - Array of integers
        k: int - Divisor to check against

    Returns:
        bool: True if such subarray exists, False otherwise

    More such problems:
        https://leetcode.com/problems/continuous-subarray-sum/solutions/5276981/prefix-sum-hashmap-patterns-7-problems/
    """
    remainder_index_map = {0: -1}  # Handle case where subarray starts at index 0
    prefix_sum = 0

    for i, num in enumerate(nums):
        prefix_sum += num
        remainder = prefix_sum % k

        if remainder in remainder_index_map:
            if i - remainder_index_map[remainder] >= 2:
                return True
        else:
            remainder_index_map[remainder] = i
    return False


def diameter_of_binary_tree(root):
    # https://leetcode.com/problems/diameter-of-binary-tree
    """
    Returns the diameter (longest path between any two nodes) of a binary tree.

    ANALOGY: THE TREE HOUSE ROPEBRIDGE

    Imagine each node is a treehouse, and we want to build the longest possible
    ropebridge connecting any two treehouses. The bridge can go up/down branches.

    KEY INSIGHT: The longest path through any node is the sum of the deepest paths
    through its left and right children. But we can't just return that sum because
    parent nodes need to know the depth from their perspective.

    HOW IT WORKS:
    1. For each node, we calculate:
       - Left depth: longest path down left subtree
       - Right depth: longest path down right subtree
       - Through current node: left + right (this is a candidate diameter)

    2. But we return to parent: max(left, right) + 1
       Why? Because from parent's perspective, the best path THROUGH parent
       would use the DEEPER of its child's paths plus the edge to that child.

    3. We track the maximum diameter seen globally using a mutable variable.

    EXAMPLE:
           1
          / \
         2   3
        / \
       4   5

    Node 4: left=0, right=0 → diameter=0 → returns 1
    Node 5: left=0, right=0 → diameter=0 → returns 1
    Node 2: left=1 (from 4), right=1 (from 5) → diameter=2 → returns max(1,1)+1=2
    Node 3: left=0, right=0 → diameter=0 → returns 1
    Node 1: left=2 (from 2), right=1 (from 3) → diameter=3 → returns max(2,1)+1=3

    The diameter is 3 (path [4-2-1-3] or [4-2-5])

    Memory Aid: "Every node asks: 'What's the longest path through me?'
                Then tells parent: 'Here's my depth for YOUR longest path!'"

    Args:
        root: TreeNode - Root of the binary tree

    Returns:
        int - Length of the diameter (number of edges in the longest path)
    """
    result = 0

    def dfs(node):
        nonlocal result

        if not node:
            return 0

        left, right = dfs(node.left), dfs(node.right)
        result = max(result, left + right)
        return max(left, right) + 1

    dfs(root)

    return result


def subarray_sum_equals_k(nums, k):
    # https://leetcode.com/problems/subarray-sum-equals-k
    prefix_sum = 0
    prefix_sum_counts = defaultdict(int)
    result = 0

    # Initialize with 0: 1 for subarrays starting at index 0
    prefix_sum_counts[0] = 1

    for num in nums:
        prefix_sum += num
        diff = prefix_sum - k

        # If we've seen this diff before, each occurrence represents a valid subarray
        if diff in prefix_sum_counts:
            result += prefix_sum_counts[diff]

        # Record the current prefix sum for future checks
        prefix_sum_counts[prefix_sum] += 1

    return result


def exclusive_time_of_functions(n, logs):
    # https://leetcode.com/problems/exclusive-time-of-functions
    # https://leetcode.com/problems/exclusive-time-of-functions/discuss/863039/Python-3-or-Clean-Simple-Stack-or-Explanation

    # Result array: exclusive time of each function (initialized to 0)
    # Stack: holds [function_id, start_time] for functions currently running
    # 'prev_time' is often used for tracking time slices
    result, stack, prev_time = [0] * n, [], 0
    for log in logs:
        # Parse the log entry (format: "id:start/end:timestamp")
        func, status, curr_time = log.split(':')
        func, curr_time = int(func), int(curr_time)

        if status == 'start':
            # If another function was running, give it credit up to (curr_time - 1)
            if stack:
                result[stack[-1]] += curr_time - prev_time
            # Push the new function onto the stack (it starts running now)
            stack.append(func)
            # Update prev_time to this function's start time
            prev_time = curr_time
        else: # status == 'end'
            # Pop the function that just ended
            stack.pop()
            # Add its running time from prev_time up to curr_time (inclusive)
            result[func] += (curr_time - prev_time + 1)
            # The next function (if any) resumes at curr_time + 1
            prev_time = curr_time + 1
    return result


def palindromic_substring(s):
    # https://leetcode.com/problems/palindromic-substrings
    n = len(s)
    result = 0

    def count_palindrome(left, right):
        nonlocal result
        while left >= 0 and right < n and s[left] == s[right]:
            result += 1
            left -= 1
            right += 1

    for i in range(n):
        count_palindrome(i, i)
        count_palindrome(i, i + 1)

    return result


def find_k_closest_elements(arr, k, x):
    # https://leetcode.com/problems/find-k-closest-elements
    """
    Finds the k closest integers to x in the sorted array arr.
    Returns the result sorted in ascending order.

    ANALOGY: THE MOUNTAIN RESCUE MISSION

    Imagine x is an injured hiker on a mountain trail (sorted number line).
    We need to send the k closest rescue teams (array elements) to the hiker.

    STRATEGY: THE "TWO BOUNDARY" APPROACH
    Instead of finding each team individually, we locate the best starting point
    for a rescue squad of size k that's closest to the hiker.

    WHY BINARY SEARCH WORKS:
    1. The array is sorted, so any k consecutive elements form a continuous rescue team
    2. We want the k-team where the farthest member minimizes distance to x
    3. If we know where to start the team, we can slide it left or right to optimize

    BINARY SEARCH LOGIC:
    We search for the optimal START index (0 ≤ start ≤ n-k):
    - Compare arr[mid] and arr[mid+k] (the elements just outside current team)
    - If x is closer to arr[mid] than arr[mid+k], move LEFT (right = mid)
    - If x is closer to arr[mid+k] than arr[mid], move RIGHT (left = mid + 1)
    - When distances are equal, choose the smaller element (move left)

    WHY COMPARE ONLY TWO ELEMENTS?
    Because the array is sorted, if x is much closer to arr[mid] than arr[mid+k],
    then ANY team starting to the right of mid would have its leftmost element
    even farther from x than arr[mid+k], so they can't be better.

    EXAMPLE with arr = [1,2,3,4,5], k=4, x=3:
    Initially consider team [1,2,3,4] (start=0):
        - Compare 1 (left boundary) and 5 (right boundary of next team)
        - |3-1|=2, |5-3|=2 → equal, but prefer smaller → keep left
    Actually, we compare arr[start] vs arr[start+k]:
        start=0: arr[0]=1, arr[4]=5 → |3-1|=2, |5-3|=2 → not greater, so right=0

    The algorithm finds start=0 → result [1,2,3,4]

    Memory Aid: "Find where to park your k-car train closest to station x!"

    Args:
        arr: List[int] - Sorted array of integers
        k: int - Number of closest elements to find
        x: int - Target value

    Returns:
        List[int] - k closest elements to x, sorted in ascending order
    """
    n = len(arr)
    left, right = 0, n - k  # Possible start indices for k-length subarray

    while left < right:
        mid = (left + right) // 2

        # Compare distances from x to the two boundaries
        # If x is closer to arr[mid+k] than arr[mid], move start to the right
        # Note: when equal, we prefer the smaller element (left side)
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid

    # Return the k elements starting from the optimal index
    return arr[left: left + k]


def maximum_swap(num):
    # https://leetcode.com/problems/maximum-swap
    """
    To get the maximum number from swapping two digits at most once, you want to swap the leftmost smaller digit
    with the rightmost largest possible digit that comes after it.
    Input:  9832547610
    Output: 9872543610
    Swap:   3 ↔ 7
    """
    s = list(str(num))  # num = 2736, result = 7236
    n = len(s)

    # Track the rightmost max digit and positions to swap
    max_idx = n - 1
    smallest, largest = -1, -1  # smallest tracks the smallest and largest tracks the largest digits

    # Traverse from right to left
    # 🔧 Example: 2736
    # Start: max_idx = 3 (digit 6)
    # i=2 → compare 3 vs 6 → 3 < 6 → smallest needs to be updated with i and largest with max_idx
    # i=1 → compare 7 vs 6 → 7 > 6 → update max_idx = 1
    # i=0 → compare 2 vs 7 → 2 < 7 → smallest needs to be updated with i and largest with max_idx
    # So the last (smallest, largest) = (0, 1) → swap 2 and 7 → 7236.
    for i in range(n - 2, -1, -1):
        if s[i] > s[max_idx]:
            max_idx = i
        elif s[i] < s[max_idx]:
            smallest, largest = i, max_idx

    if smallest == -1:  # already largest
        return num

    s[smallest], s[largest] = s[largest], s[smallest]
    return int("".join(s))


def valid_palindrome_ii(s):
    # https://leetcode.com/problems/valid-palindrome-ii
    def check_palindrome(i, j):
        while i < j:
            if s[i] == s[j]:
                i += 1
                j -= 1
            else:
                return False
        return True

    left, right = 0, len(s) - 1

    while left < right:
        if s[left] == s[right]:
            left += 1
            right -= 1
        else:
            return check_palindrome(left + 1, right) or check_palindrome(left, right - 1)
    return True


class SparseVector1:
    # https://leetcode.com/problems/dot-product-of-two-sparse-vectors
    def __init__(self, nums):
        self.linked_list = [[i, num] for i, num in enumerate(nums)]

    def dot_product(self, vec):
        if len(vec.linked_list) < len(self.linked_list):
            self, vec = vec, self

        result = i = j = 0
        n1, n2 = len(self.linked_list), len(vec.linked_list)

        while i < n1 and j < n2:
            if self.linked_list[i][0] == vec.linked_list[j][0]:
                result += self.linked_list[i][1] * vec.linked_list[j][1]
                i += 1
                j += 1
            elif self.linked_list[i][0] < vec.linked_list[j][0]:
                i += 1
            else:
                j += 1
        return result
