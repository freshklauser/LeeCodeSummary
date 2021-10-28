# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/25 21:31
# @FileName : demo.py
# @SoftWare : PyCharm


import bisect


def is_valid(s):
    cnt = 0
    for char in s:
        if s == '(':
            cnt += 1
        if s == ')':
            cnt -= 1
            # 右括号多则直接无效
            if cnt < 0:
                return False
    # 合理的字符串，左右括号相等，cnt==0
    return cnt == 0


def check_power(num):
    return bin(num).count('1') == 1


print(list(filter(check_power, [2, 4, 6, 8, 10, 16])))


class Solution:
    def permute(self, nums):
        res = []
        self.dfs(nums, [], set(), res)
        return res

    def dfs(self, nums, path, visited, res):
        """
        index: 下一个要访问的元素索引
        """
        if len(path) == len(nums):
            res.append(path)
            return
        for i in range(len(nums)):
            if nums[i] in visited:
                continue
            path.append(nums[i])
            visited.add(nums[i])
            self.dfs(nums, path[:], visited, res)
            path.pop()
            visited.remove(nums[i])


if __name__ == '__main__':
    nums = [2, 4, 6, 8, 0]
    inst = Solution()
    inst.permute(nums)