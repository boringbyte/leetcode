import heapq
import string
from collections import deque, Counter
from itertools import zip_longest
from leetcode.CapitalOne.relevant_leetcode.Easy import ListNode


def diameter_of_binary_tree(root):
    # https://leetcode.com/problems/diameter-of-binary-tree/
    result = [0]
    def dfs(node):
        if not node:
            return 0
        left, right = dfs(node.left), dfs(node.right)
        result[0] = max(result[0], left + right)
        return 1 + max(left, right)

    dfs(root)
    return result[0]


def merge_sorted_array(nums1, m, nums2, n):
    # https://leetcode.com/problems/merge-sorted-array/
    """
    Arrange from backwards of the nums1 array as it contains zeros.
    Also, if the non-zero elements are more in nums2 then simply copy them at the end
    """
    while m > 0 and n > 0:
        if nums1[m - 1] >= nums2[n - 1]:
            nums1[m + n - 1] = nums1[m - 1]
            m -= 1
        else:
            nums1[m + n - 1] = nums2[n - 1]
            n -= 1

    if n > 0:
        nums1[:n] = nums2[:n]


def valid_palindrome(s):
    # https://leetcode.com/problems/valid-palindrome/
    """
    Think of normal palindrome validation of two pointer solution.
    Use negative conditions:
      1. Check if a character on left and right pointers are not an alphanumeric or not.
      2. Another is normal palindrome validation check
    """
    l, r = 0, len(s)
    while l < r:
        if not s[l].isalnum():
            l += 1
        elif not s[r].isalnum():
            r -= 1
        else:
            if s[l].lower() != s[r].lower():
                return False
            l, r = l + 1, r - 1
    return True


def valid_palindrome_ii(s):
    # https://leetcode.com/problems/valid-palindrome-ii
    """
    1. Write a normal two pointer function for checking if a string is palindrome or not.
    2. Use Two-pointer palindrome check — allow one mismatch by skipping either left or right character once.
    """
    def check_palindrome(l, r):
        while l < r:
            if s[l] != s[r]:
                return False
            l, r = l + 1, r - 1
        return True

    i, j = 0, len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return check_palindrome(i + 1, j) or check_palindrome(i, j - 1)
        i, j = i + 1, j -  1
    return True


def valid_word_abbreviation(word, abbr):
    # https://leetcode.com/problems/valid-word-abbreviation/description/
    # https://shandou.medium.com/leetcode-408-valid-word-abbreviation-63f1ed6461de
    """
    A string can be shortened by replacing any number of non-adjacent, non-empty substrings with their lengths (without leading zeros).
    word = "apple", abbr = "a3e"
    word = "abbreviation", abbr = "a2reviation"
    """
    i = j = 0
    m , n = len(word), len(abbr)

    while i < m and j < n:
        if word[i] == abbr[j]:  # Condition 1: Check if both are same characters
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


def range_sum_of_bst(root, low, high):
    # https://leetcode.com/problems/range-sum-of-bst
    result = [0]

    def dfs(node):
        if node:
            if low <= node.val <= high:
                result[0] += node.val
            if low <= node.val:
                dfs(node.left)
            if node.val <= high:
                dfs(node.right)

    dfs(root)
    return result[0]


def kth_missing_positive_number_1(arr, k):
    # https://leetcode.com/problems/kth-missing-positive-number
    # https://leetcode.com/problems/kth-missing-positive-number/solutions/1004535/python-two-solutions-o-n-and-o-log-n-explained/
    s = set(arr)
    n = k + len(arr) + 1
    for i in range(1, n):
        if i not in s:
            k -= 1
        if k == 0:
            return i


def kth_missing_positive_number_2(arr, k):
    start, end = 0, len(arr)
    while start < end:
        mid = (start + end) // 2
        if arr[mid] - mid - 1 < k:
            start = mid + 1
        else:
            end = mid
    return end + k


class MovingAverage:
    # https://www.jointaro.com/interviews/questions/moving-average-from-data-stream/
    # https://algo.monster/liteproblems/346

    def __init__(self, size):
        self.size = size
        self.data = deque()
        self.current_total = 0

    def next(self, value):
        self.data.append(value)
        self.current_total += value

        if len(self.data) > self.size:
            self.current_total -= self.data.popleft()

        current_size = min(len(self.data), self.size)
        return self.current_total / current_size


def best_time_to_buy_and_sell_stock(prices):
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
    profit, buy_price = 0, prices[0]
    for price in prices[1:]:
        buy_price = min(buy_price, price)
        profit = max(profit, price - buy_price)
    return profit


def closest_binary_search_tree_value(root, target):
    # https://algo.monster/liteproblems/270
    # https://www.geeksforgeeks.org/dsa/find-closest-element-binary-search-tree/
    difference = result = float('inf')
    while root:
        new_difference = abs(root.val - target)
        if new_difference < difference:
            difference = new_difference
            result = root.val
        if root.val < target:
            root = root.right
        elif target < root.val:
            root = root.left
        else:
            break

    return result


def add_strings(num1, num2):
    # https://leetcode.com/problems/add-strings

    i, j = len(num1) - 1, len(num2) - 1
    carry, result = 0, []
    while i >= 0 or j >= 0 or carry:
        digit1 = digit2 = 0
        if i >= 0:
            digit1 = int(num1[i])
        if j >= 0:
            digit2 = int(num2[j])
        carry, digit = divmod(digit1 + digit2 + carry, 10)
        result.append(str(digit))
        i, j = i + 1, j - 1
    return ''.join(result[::-1])


def two_sum(nums, target):
    nums_dict = {num: i for i, num in enumerate(nums)}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in nums_dict and i != nums[diff]:
            return [i, nums_dict[diff]]


def valid_parenthesis(s):
    if len(s) % 2:
        return False

    stack, hashmap = [], {'(': ')', '[': ']', '{': '}'}

    for char in s:
        if char in hashmap:
            stack.append(char)
        else:
            if stack and hashmap[stack[-1]] == char:
                stack.pop()
            else:
                return False
    return len(stack) == 0


def to_goat_latin(sentence):
    words = sentence.split()
    result = []
    for i, word in enumerate(words):
        if word[0].lower() in 'aeiou':
            new_word = word + 'ma'
        else:
            new_word = word[1:] + word[0] + 'ma'
        new_word = new_word + ('a' * (i + 1))
        result.append(new_word)
    return ' '.join(result)


def longest_common_prefix(strs):
    # https://leetcode.com/problems/longest-common-prefix/description/
    if not str:
        return ''
    shortest_str = min(strs, key=len)
    for i, char in enumerate(shortest_str):
        for other_str in strs:
            if other_str[i] != char:
                return shortest_str[:i]
    return shortest_str


def strobogrammatic_number(num):
    # https://algo.monster/liteproblems/246
    number_map = {('0', '0'), ('1', '1'), ('6', '9'), ('8', '8'), ('9', '6')}
    i, j = 0, len(num) - 1
    while i <= j:
        if (num[i], num[j]) not in number_map:
            return False
        i, j = i + 1, j - 1
    return True


def remove_duplicates_from_sorted_array(nums):
    # https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/
    # https://leetcode.com/problems/remove-duplicates-from-sorted-array/solutions/2107606/py-all-4-methods-intuitions-walk-through-wrong-answer-explanations-for-beginners-python/
    if not nums:
        return 0
    slow, fast, n = 0, 1, len(nums)
    while fast < n:
        if nums[slow] == nums[fast]:
            fast += 1
        else:
            nums[slow + 1] = nums[fast]
            slow += 1
            fast += 1
    return slow + 1


def middle_of_the_linked_list(head):
    # https://leetcode.com/problems/middle-of-the-linked-list/description/
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def move_zeros(nums):
    # https://leetcode.com/problems/move-zeroes/description/
    snow_ball_size, n = 0, len(nums)

    for i in range(n):
        if nums[i] == 0:
            snow_ball_size += 1
        elif snow_ball_size > 0:
            t = nums[i]
            nums[i] = 0
            nums[i - snow_ball_size] = t


def maximum_difference_by_remapping_a_digit(num):
    # https://leetcode.com/problems/maximum-difference-by-remapping-a-digit/description/
    s = str(num)

    replace_for_max = ''
    for char in s:
        if char != '9':
            replace_for_max = char
            break
    max_str = ''.join(['9' if char == replace_for_max else char for char in s])

    replace_for_min = s[0]
    min_str = ''.join(['0' if char == replace_for_min else char for char in s])

    return int(max_str) - int(min_str)


def climbing_stairs(n):
    # https://leetcode.com/problems/climbing-stairs
    if n <= 3:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]


def pascals_triangle(num_rows):
    # https://leetcode.com/problems/pascals-triangle/description/
    result = [[1]]
    for i in range(2, num_rows + 1):
        l = [0] + result[-1][:]
        r = result[-1][:] + [0]
        result.append([x + y for x, y in zip(l, r)])
    return result


def roman_to_integer(s):
    symbols = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result, n = 0, len(s)

    for i in range(n):
        if i + 1 < n and symbols[s[i]] < symbols[s[i + 1]]:
            result -= symbols[s[i]]
        else:
            result += symbols[s[i]]
    return result


def check_if_two_string_arrays_are_equivalent(word1, word2):
    # https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/solutions/1007878/python-understanding-generators-and-yield-statement/

    def generate(word_list):
        for word in word_list:
            for char in word:
                yield char
        yield None

    for c1, c2 in zip_longest(generate(word1), generate(word2)):
        if c1 != c2:
            return False
        return True


def find_peaks(mountain):
    # https://leetcode.com/problems/find-the-peaks/
    n, result = len(mountain), []

    for i in range(1, n - 1):
        if mountain[i - 1] < mountain[i] > mountain[i + 1]:
            result.append(i)
    return result


def average_selling_price():
    # https://leetcode.com/problems/find-the-peaks/description/
    pass


class RecentCounter:
    # https://leetcode.com/problems/number-of-recent-calls
    def __init__(self):
        self.queue = deque()

    def pint(self, t):
        self.queue.append(t)
        while self.queue[-1] - self.queue[0] > 3000:
            self.queue.popleft()
        return len(self.queue)


def find_all_k_distant_indices_in_an_array(nums, key, k):
    # https://leetcode.com/problems/find-all-k-distant-indices-in-an-array
    pass


def linked_list_cycle(head):
    # https://leetcode.com/problems/linked-list-cycle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def reverse_vowels_of_a_string(s):
    # https://leetcode.com/problems/reverse-vowels-of-a-string/
    vowels = "aeiou"
    s = list(s)
    l, r = 0, len(s) - 1

    while l < r:
        while l < r and s[l].lower() not in vowels:  # Loop through all consonants until next vowel
            l += 1
        while l < r and s[r].lower() not in vowels:  # Loop through all consonants until next vowel
            r -= 1
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1
    s = ''.join(s)
    return s


def single_number(nums):
    #https://leetcode.com/problems/single-number
    result = 0
    for num in nums:
        result ^= num
    return result


def palindrome_number(x):
    # https://leetcode.com/problems/palindrome-number
    if x < 0:
        return False

    div = 1
    while x // div >= 10:
        div *= 10

    while x:
        left, rem = divmod(x, div)
        right = x % 10

        if left != right:
            return False

        x = rem % 10
        div //= 100

    return True


def binary_search(nums, target):
    # https://leetcode.com/problems/binary-search
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if target > nums[mid]:
            l = mid + 1
        else:
            r = mid
    return -1 if nums[l] != target else l


def find_subsequence_of_length_k_with_largest_sum(nums, k):
    # https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum.
    # https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum/solutions/1705383/python-simple-solution-100-time/
    heap = []
    for i, num in enumerate(nums):
        if len(heap) == k:
            heapq.heappushpop(heap, (num, i))
        else:
            heapq.heappush(heap, (num, i))

    heap.sort(key=lambda x: x[1])  # Sorting by index puts them back in original order, so the subsequence condition is respected.
    return [num for num, _ in heap]


def merge_two_sorted_lists(list1, list2):
    # https://leetcode.com/problems/merge-two-sorted-lists/description/

    if list1 and list2:
        head = current = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        if list1 or list2:
            current.next = list1 or list2
        return head.next
    else:
        return list1 or list2


def minimum_depth_of_binary_tree_1(root):
    # https://leetcode.com/problems/minimum-depth-of-binary-tree/solutions/1087559/python-solutions-bfs-and-dfs/

    def dfs(node):
        if not node:
            return 0

        left, right = dfs(node.left), dfs(node.right)
        if left == 0:
            return right + 1
        if right == 0:
            return left + 1
        return 1 + min(left, right)

    return dfs(root)


def minimum_depth_of_binary_tree_2(root):
    if not root:
        return 0

    queue = deque([(root, 1)])
    while queue:
        node, level = queue.popleft()
        if node:
            if not node.left and not node.right:
                return level
            else:
                queue.append((node.left, level + 1))
                queue.append((node.right, level + 1))


def find_the_kth_character_in_string_game_i(k):
    # https://leetcode.com/problems/find-the-k-th-character-in-string-game-i/description/
    # https://leetcode.com/problems/find-the-k-th-character-in-string-game-i/solutions/6911890/beginner-freindly-java-c-python-js/
    word = ['a']
    letters = string.ascii_lowercase
    char_int_map = {char: i for i, char in enumerate(string.ascii_lowercase)}
    next_int_char_map = {i + 1: letters[(i + 1) % 26] for i in range(26)}
    while len(word) < k:
        for i in range(len(word)):
            # Convert character to number (0 for 'a', 1 for 'b', ..., 25 for 'z')
            offset = char_int_map[word[i]]
            # Get the next character cyclically (so 'z' -> 'a')
            # next_char = chr(ord('a') + (offset + 1) % 26)
            next_char = next_int_char_map[offset % 26]
            word.append(next_char)
            if len(word) >= k:
                break
    return word[k - 1]


def contains_duplicate_ii(nums, k):
    # https://leetcode.com/problems/contains-duplicate-ii
    seen = dict()
    for i, num in enumerate(nums):
        if num in seen and abs(i - seen[num]) <= k:
            return True
        seen[num] = i
    return False


def valid_anagram_1(s, t):
    # https://leetcode.com/problems/valid-anagram
    s_counter = Counter(s)
    t_counter = Counter(t)
    return s_counter == t_counter


def valid_anagram_2(s, t):
    # https://leetcode.com/problems/valid-anagram
    l, m = [0] * 26, [0] * 26
    for char in s:
        l[ord(char) - ord('a')] += 1

    for char in t:
        m[ord(char) - ord('a')] += 1

    return l == m


def find_missing_and_repeated_values(grid):
    # https://leetcode.com/problems/find-missing-and-repeated-values
    # https://leetcode.com/problems/find-missing-and-repeated-values/solutions/6503987/counting-math-python-c-java-js-c-go-swift/
    # flat_list = [item for row in grid for item in row]
    n = len(grid)
    size = n * n
    count = [0] * (size + 1)
    for i in range(n):
        for j in range(n):
            count[grid[i][j]] += 1

    a, b = -1, -1
    for num in range(1, size + 1):
        if count[num] == 2:
            a = num
        elif count[num] == 0:
            b = num
    return [a, b]


def find_the_index_of_first_occurrence_in_a_string_1(haystack, needle):
    # https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/solutions/665448/ac-simply-readable-python-kmp-rabin-karp/
    # haystack.find(needle)

    """
    # Robin Karp
    n, h = len(haystack), len(needle)
    needle_hash = hash(needle)
    for i in range(h - n + 1):
        if hash(haystack[i: i + n]) == needle_hash:
            return i
    return -1
    """
    if len(needle) == 0:
        return 0
    elif needle not in haystack:
        return -1
    else:
        return len(haystack.split(needle)[0])


class RangeSumQueryImmutable:
    # https://leetcode.com/problems/range-sum-query-immutable/

    def __init__(self, nums):
        self.nums = nums
        self.running_sum = nums[:]
        for i in range(1, len(self.nums)):
            self.running_sum[i] = self.running_sum[i - 1] + self.nums[i]

    def sum_range(self, left, right):
        if left == 0:
            return self.running_sum[right]
        else:
            return self.running_sum[right] - self.running_sum[left - 1]


def reverse_bits_1(n):
    binary = bin(n)[2:]
    binary = binary.zfill(32) # binary.rjust(32, '0'), for suffix zeros ljust can be used
    return int(binary[::-1], 2)


def reverse_bits_2(n):
    # https://leetcode.com/problems/reverse-bits/solutions/5228734/simple-solution-with-comments-and-explaination/
    result = 0
    for i in range(32):
        least_significant_bit = n & 1  # Get the rightmost bit of 'n'.
        n = n >> 1  # Right shift to remove the least significant bit
        result = result << 1  # Left shift to make room for the least significant bit
        result = result | least_significant_bit  # Add the rightmost bit of 'n' to the result
    return result
