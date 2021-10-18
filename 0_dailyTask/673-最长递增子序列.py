# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/20 20:42
# @FileName : 673-最长递增子序列.py
# @SoftWare : PyCharm



class Solution:
    def findNumberOfLIS(self, nums):
        max_sub_len = 0
        max_sub_cnt = 0




if __name__ == '__main__':
    nums = [1,3,5,4,7]
    # inst = Solution()
    # print(inst.is_increase_sequence(nums))
    import heapq
    print(heapq.heapify())

    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ext = len(arr) % 6
    base = len(arr) // 6
    print(ext, base)
    nums = [base + 1 if i + 1 <= ext else base for i in range(6)]
    print(nums)
