from LCMetaPractice import TreeNode


class MaxSubArraySum:
    def __init__(self):
        self.arr = [-1, 2, 4, -3, 5, 2, -5, 2]

    def algorithm1(self):
        best, n = 0, len(self.arr)
        for i in range(n):
            for j in range(i, n):
                total = 0
                for k in range(i, j + 1):
                    total += self.arr[k]
                best = max(best, total)
        return best

    def algorithm2(self):
        best, n = 0, len(self.arr)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += self.arr[j]
                best = max(best, total)
        return best

    def algorithm3(self):
        best, total, n = 0, 0, len(self.arr)
        for k in range(n):
            total = max(self.arr[k], total + self.arr[k])
            best = max(best, total)
        return best


class BinarySearch:

    def __init__(self):
        self.arr = [-1, 0, 8, 11, 15, 19, 20, 21, 22, 30, 31]

    def algorithm1(self, target):
        left, right = 0, len(self.arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if self.arr[mid] == target:
                return mid
            elif self.arr[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1

    def algorithm2(self, target):
        left, right = 0, len(self.arr) - 1
        while left < right:
            mid = left + (right - left) // 2
            if target > self.arr[mid]:
                left = mid + 1
            else:
                right = mid
        return left if self.arr[left] == target else -1

    def algorithm3(self, target):
        left, n = 0, len(self.arr)
        mid = n // 2
        while mid > 0:
            while left + mid < n and self.arr[left + mid] <= target:
                left += mid
            mid //= 2
        return left if self.arr[left] == target else -1


class Codec:
    @staticmethod
    def serialize(root):
        values = []

        def dfs(node):
            if node:
                values.append(str(node.val))
                dfs(node.left)
                dfs(node.right)
            else:
                values.append('#')
        dfs(root)
        return ' '.join(values)

    @staticmethod
    def deserialize(data):

        def dfs():
            value = next(values)
            if value == '#':
                return None
            node = TreeNode(int(value))
            node.left = dfs()
            node.right = dfs()
            return node

        values = iter(data.split())
        return dfs()


def window():
    nums = [1, 2, 3, 4, 5]
    n = 3
    i, j = 0, n
    while j <= len(nums):
        print(nums[i: j])
        i += 1
        j += 1


if __name__ == '__main__':
    problem1 = MaxSubArraySum()
    print(problem1.algorithm1())
    print(problem1.algorithm2())
    print(problem1.algorithm3())

    binary_search = BinarySearch()
    print(binary_search.algorithm1(19))
    print(binary_search.algorithm1(23))
    print(binary_search.algorithm2(19))
    print(binary_search.algorithm2(23))
    print(binary_search.algorithm3(19))
    print(binary_search.algorithm3(-1))
    print(binary_search.algorithm3(31))
    print(binary_search.algorithm3(23))
    print(binary_search.algorithm3(-8))
    print(binary_search.algorithm3(33))
