import random

from numpy.ma.core import equal


def buildings_with_an_ocean_view(heights):
    # https://leetcode.com/problems/buildings-with-an-ocean-view
    # https://goodtecher.com/leetcode-1762-buildings-with-an-ocean-view/
    """
    Ocean is on the right side of the building
     | |
    ||||| -> ocean
    """
    stack = []
    n = len(heights)

    for i in range(n - 1, -1, -1):
        if not stack:
            stack.append(i)
        if heights[i] > heights[stack[-1]]:
            stack.append(i)

    return stack[::-1]


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


def minimum_add_to_make_parentheses_valid(s):
    # https://leetcode.com/problems/minimum-add-to-make-parentheses-valid
    left = right = 0

    for char in s:
        if char == "(":
            left += 1
        else:
            if left:
                left -= 1
            else:
                right += 1

    return left + right


def range_sum_bst(root, low, high):
    # https://leetcode.com/problems/range-sum-of-bst
    result = 0

    def dfs(node):
        nonlocal result

        if node:
            if low <= node.val <= high:
                result += node.val
            if low <= node.val:
                dfs(node.left)
            if node.val <= high:
                dfs(node.right)

    dfs(root)
    return result


def product_of_two_run_length_encoded_arrays(encoded1, encoded2):
    # https://leetcode.com/problems/product-of-two-run-length-encoded-arrays
    # https://algo.monster/liteproblems/1868
    """
    Input:
    encoded1 = [[1,3],[2,3]]
    encoded2 = [[6,3],[3,3]]

    Decoded forms:
    encoded1 → [1,1,1,2,2,2]
    encoded2 → [6,6,6,3,3,3]

    Products → [6,6,6,6,6,6] = [[6,6]]
    """
    result = []
    i = j = 0

    while i < len(encoded1) and j < len(encoded2):
        val1, freq1 = encoded1[i]
        val2, freq2 = encoded2[j]

        product = val1 * val2
        overlap = min(freq1, freq2)

        if not result:
            result.append([product, overlap])
        else:
            if result[-1][0] == product:
                result[-1][1] += overlap
            else:
                result.append([product, overlap])

        # Decrement used frequencies
        encoded1[i][1] -= overlap
        encoded2[j][1] -= overlap

        if encoded1[i][1] == 0:
            i += 1

        if encoded2[j][1] == 0:
            j += 1

    return result


def k_closest_points_to_origin(points, k):
    # https://leetcode.com/problems/k-closest-points-to-origin

    def euclidean_distance(i):
        x, y = points[i]
        return x * x + y * y

    def swap(i, j):
        points[i], points[j] = points[j], points[i]

    def partition(left, right, pivot_index):
        pivot_dist = euclidean_distance(pivot_index)

        # Move pivot to end
        swap(pivot_index, right)

        store_index = left
        for i in range(left, right):
            if euclidean_distance(i) <= pivot_dist:
                swap(i, store_index)
                store_index += 1

        # Move pivot to its final place
        swap(store_index, right)
        return store_index

    def quickselect(left, right):
        if left >= right:
            return

        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)

        if pivot_index == k:
            return
        elif k < pivot_index:
            quickselect(left, pivot_index - 1)
        else:
            quickselect(pivot_index + 1, right)

    quickselect(0, len(points) - 1)
    return points[:k]


def squares_of_sorted_array(nums):
    # https://leetcode.com/problems/squares-of-a-sorted-array
    n = len(nums)
    left, right = 0, n - 1
    k = n - 1
    result = [None] * n

    while left <= right:
        a, b = nums[left] ** 2, nums[right] ** 2
        if a >= b:
            left += 1
            result[k] = a
        else:
            right -= 1
            result[k] = b
        k -= 1
    return result


def minimum_window_substring(s, t):
    # https://leetcode.com/problems/minimum-window-substring
    pass


def goat_latin(sentence):
    # https://leetcode.com/problems/goat-latin
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


def decode_string(s):
    # https://leetcode.com/problems/decode-string
    stack = [""]
    num = 0

    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == "[":
            stack.append(num)
            num = 0
            stack.append("")
        elif char.isalpha():
            stack[-1] += char
        else:   # for "[" condition
            chars = stack.pop()
            times = stack.pop()
            current = chars * times
            prev = stack.pop()
            stack.append(prev + current)
    return "".join(stack)
