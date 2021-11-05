# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/11/5 21:20
# @FileName : 1218_最长定差子序列.py
# @SoftWare : PyCharm


def run(arr, difference):
    d = {}
    for a in arr:
        d[a] = d.get(a - difference, 0) + 1
        print(a, '||', a - difference, '||', d[a], '--->', d)
    return max(d.values())


if __name__ == '__main__':
    arr = [1,5,7,8,5,3,4,2,1]
    difference = 2
    run(arr, difference)
