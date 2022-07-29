import collections
import heapq
import string


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return True
            node = node.children[char]
        return node.is_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


def reverse_string(s):
    l, r = 0, len(s) - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1
    return s


def permute(nums):
    n, result = len(nums), []
    visited = [0] * n

    def backtrack(sofar):
        if len(sofar) == n:
            result.append(sofar[:])
        else:
            for i in range(n):
                if visited[i] != 1:
                    chosen, visited[i] = nums[i], 1
                    backtrack(sofar+[chosen])
                    visited[i] = 0
    backtrack(sofar=[])
    return result


def delete_node(node):
    current = slow = node
    while current.next:
        current.val = current.next.val
        slow = current
        current = current.next
    slow.next = None


def max_depth_recursive(root):
    if root is None:
        return 0
    left = max_depth_recursive(root.left)
    right = max_depth_recursive(root.right)
    return max(left, right) + 1


def max_depth_level(root):
    if root is None:
        return 0
    level, queue = 0, collections.deque([root])
    while queue:
        level, size = level + 1, len(queue)
        for _ in range(size):
            current = queue.popleft()
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
    return level


def max_depth_dfs(root):
    if root is None:
        return 0
    stack, max_depth = [(root, 1)], 0
    while stack:
        current, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if current.left:
            stack.append((current.left, depth + 1))
        if current.right:
            stack.append((current.right, depth + 1))
    return max_depth


def depth_binary_tree(root):
    if root is None:
        return 0
    l, r = depth_binary_tree(root.left), depth_binary_tree(root.right)
    return max(l, r)+1


def diameter_binary_tree(root):
    diameter = [0]

    def recursive(node):
        if node is None:
            return 0
        l, r = recursive(node.left), recursive(node.right)
        diameter[0] = max(diameter[0], l+r)  # There is no l+r+1 because, 4 nodes form 3 edges and
        # we are working with edges in this problem
        return max(l, r)+1
    recursive(root)
    return diameter[0]


def max_path_sum_binary_tree(root):
    max_sum = [float('-inf')]

    def recursive(node):
        if node is None:
            return 0
        l, r = recursive(node.left), recursive(node.right)
        max_sum[0] = max(max_sum[0], l+r+node.val)
        return max(l+node.val, r+node.val, 0)
    recursive(root)
    return max_sum[0]


def subsets(nums):
    result, n = [], len(nums)

    def backtrack(sofar, start):
        result.append(sofar[:])
        for i in range(start, n):
            chosen = nums[i]
            sofar.append(chosen)
            backtrack(sofar, i+1)
            sofar.pop()

    backtrack(sofar=[], start=0)
    return result


def inorder_traversal_recursive(root):
    result = []

    def recursive(root):
        if root:
            recursive(root.left)
            result.append(root.val)
            recursive(root.right)

    recursive(root)
    return result


def inorder_traversal_stack(root):
    stack, result, current = [], [], root
    while True and root:
        if current:
            stack.append(current)
            current = current.left
        elif stack:
            current = stack.pop()
            result.append(current.val)
            current = current.right
        else:
            break
    return result


def reverse_list(head):
    prev = None
    while head:
        current = head
        head = head.next
        current.next = prev
        prev = current
    return prev


def generate_parenthesis(n):
    result = []

    def backtrack(sofar, left, right):
        if len(sofar) == 2*n:
            result.append(sofar)
        else:
            if left < n:
                backtrack(sofar+'(', left+1, right)
            if right < left:
                backtrack(sofar+')', left, right+1)

    backtrack('', 0, 0)
    return result


def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result


def kth_smallest_BST(root, k):
    stack, current = [], root
    while True:
        if current:
            stack.append(current)
            current = current.left
        elif stack:
            current = current.pop()
            k -= 1
            if k == 0:
                return current.val
            current = current.right


def rotate_image_TR(matrix):
    def transpose(matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(i+1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    def reflect(matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(n//2):
                matrix[i][j], matrix[i][~j] = matrix[i][~j], matrix[i][j]

    transpose(matrix)
    reflect(matrix)


def sorted_array_BST(nums):
    n = len(nums) - 1

    def convert(left, right):
        if left < right:
            mid = (left + right) // 2
            node = TreeNode(nums[mid])
            node.left = convert(left, mid - 1)
            node.right = convert(mid + 1, right)
            return node

    convert(left=0, right=n-1)


def pascal_triangle(num_rows):
    result = [[1]]
    if num_rows > 1:
        for _ in range(2, num_rows+1):
            l = [0] + result[-1][:]
            r = result[-1][:] + [0]
            result.append([x+y for x, y in zip(l, r)])
    return result


def rotate(matrix):
    n = len(matrix)
    for i in range(n//2 + n%2):
        for j in range(n//2):
            matrix[i][j], matrix[~j][i], matrix[~i][~j], matrix[j][~i] = \
                matrix[~j][i], matrix[~i][~j], matrix[j][~i], matrix[i][j]


def top_k_frequent_elements_with_heap(nums, k):
    result, count, heap = [], collections.Counter(nums), []
    for value, priority in count.items():
        heapq.heappush(heap, (-priority, value))

    for _ in range(k):
        _, value = heapq.heappop(heap)
        result.append(value)
    return result


def top_k_frequent_elements_with_dictionary(nums, k):
    result, count, freq_dict = [], collections.Counter(nums), collections.defaultdict(list)
    for num, freq in count.items():
        freq_dict[freq].append(num)

    for key in sorted(freq_dict.keys(), reverse=True):
        result.extend(freq_dict[key])
        if len(result) >= k:
            return result[:k]
    return result[:k]


def kth_largest_element_array(nums, k):
    # This is not providing good results. This is in correct
    n = len(nums)
    if not nums or k > n:
        return -1

    def partition(left, right):
        p_index = left + (right - left) // 2
        i, pivot = left, nums[p_index]
        nums[right], nums[p_index] = nums[p_index], nums[right]
        for j in range(left, right+1):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        return i-1

    L, R, pivot_index, k = 0, n-1, n, n-k
    while pivot_index != k:
        pivot_index = partition(L, R)
        if pivot_index < k:
            L = pivot_index + 1
        else:
            R = pivot_index - 1
    return nums[k]


def group_anagrams(strs):
    hashmap = collections.defaultdict(list)
    for word in strs:
        key = [0] * 26
        for char in word:
            key[ord(char) - ord('a')] += 1
        hashmap[tuple(key)].append(word)
    return hashmap.values()


def product_except_self(nums):
    n = len(nums)
    prefix_mul, suffix_mul = [1] * n, [1] * n
    for i in range(1, n):
        prefix_mul[i] = prefix_mul[i-1] * nums[i-1]

    for i in range(n-2, -1, -1):
        suffix_mul[i] = suffix_mul[i+1] * nums[i+1]

    for i in range(n):
        suffix_mul[i] = prefix_mul[i] * suffix_mul[i]
    return suffix_mul


def majority_element(nums):
    result, count = None, 0
    for num in nums:
        if count == 0:
            result = num

        if result == num:
            count += 1
        else:
            count -= 1
    return result


def hamming_weight(n):
    result = 0
    while n:
        result += 1
        n &= (n-1)
    return result


def is_anagram(s, t):
    s_map, t_map = [0]*26, [0]*26
    for char in s:
        s_map[ord(char) - ord('a')] += 1

    for char in t:
        t_map[ord(char) - ord('a')] += 1

    return s_map == t_map


def level_order_traversal(root):
    if not root:
        return
    result, queue = [], collections.deque([root])
    while queue:
        size = len(queue)
        while size > 0:
            current = queue.popleft()
            result.append(current.val)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
            size -= 1
    return result


def contains_duplicate(nums):
    return len(nums) > len(set(nums))


def move_zeros(nums):
    snow_ball_size, n = 0, len(nums)
    for i in range(n):
        if nums[i] == 0:
            snow_ball_size += 1
        elif snow_ball_size > 0:
            t = nums[i]
            nums[i] = 0
            nums[i - snow_ball_size] = t


def excel_sheet_title_to_number(column_title):
    hashmap, result = {char: i+1 for i, char in enumerate(string.ascii_uppercase)}, 0
    for i, char in enumerate(column_title[::-1]):
        result += hashmap[char] * (26 ** i)
    return result


def merge_sorted_linked_lists(list1, list2):
    current = dummy = ListNode()
    if not list1 or not list2:
        return list1 or list2

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

    return dummy.next


def unique_paths_grid_dp1(m, n):
    grid = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] = grid[i-1][j] + grid[i][j-1]
    return grid[-1][-1]


def unique_paths_grid_dp2(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[-1]


def odd_even_linked_list(head):
    if not head or not head.next:
        return head
    odd, even, even_head = head, head.next, head.next

    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = even_head
    return head


def find_duplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow


def bisect_left(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = lo + (hi-lo) // 2
        if nums[mid] < target:
            lo = mid+1
        else:
            hi = mid
    return lo


def bisect_right(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = lo + (hi-lo) // 2
        if nums[mid] <= target:
            lo = mid+1
        else:
            hi = mid
    return lo


if __name__ == '__main__':
    print(reverse_string(["h", "e", "l", "l", "o"]))
    print(permute([1, 2, 3]))
    head = ListNode(4)
    head.next = ListNode(5)
    head.next.next = ListNode(1)
    head.next.next.next = ListNode(9)
    print(subsets([1, 2, 3]))
    print(single_number([1, 1, 2, 3, 3, 4, 5, 5, 4]))
    print(group_anagrams(strs=["eat", "tea", "tan", "ate", "nat", "bat"]))
