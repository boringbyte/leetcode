def shifting_letters(s, shifts):
    n, result = len(s), ''
    for i in range(n - 2, -1, -1):
        shifts[i] = (shifts[i] + shifts[i + 1]) % 26

    for i, char in enumerate(s):
        idx = (ord(char) - ord('a') + shifts[i]) % 26
        result += chr(idx + ord('a'))
    return result


# https://www.geeksforgeeks.org/distance-nearest-cell-1-binary-matrix/
# https://www.geeksforgeeks.org/subset-sum-problem-dp-25/
def is_subset_sum(nums, n, target):
    if target == 0:
        return True
    if n == 0:
        return False
    if nums[n - 1] > target:
        pass
# Implement array flat method. check if words are anagrams
# Write a program to help a robot find the exit to a maze.
# Simple linked list related problem with a small twist
