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


print(is_valid("(a)((())((((()"))