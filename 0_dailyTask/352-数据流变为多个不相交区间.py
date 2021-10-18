# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/9 22:33
# @FileName : 352-数据流变为多个不相交区间.py
# @SoftWare : PyCharm


import bisect


class SummaryRanges:
    def __init__(self):
        self.res = [-float('inf'), float('inf')]
        self.visited = set()

    def addNum(self, val):
        if val in self.visited:
            return
        self.visited.add(val)
        ind = bisect.bisect_right(self.res, val)
        if val == self.res[ind - 1] + 1 and val == self.res[ind] - 1:
            self.res = self.res[:ind - 1] + self.res[ind + 1:]
        elif val == self.res[ind - 1] + 1:
            self.res[ind - 1] = val
        elif val == self.res[ind] - 1:
            self.res[ind] = val
        else:
            self.res = self.res[:ind] + [val, val] + self.res[ind:]

    def getIntervals(self):
        return [[self.res[i], self.res[i + 1]] for i in range(1, len(self.res) - 2, 2)]


class SummaryRanges:
    def __init__(self) -> None:
        self.bcj = [i for i in range(10001)]
        self.res = {}

    def find(self, i):
        if self.bcj[i] != i:
            self.bcj[i] = self.find(self.bcj[i])
        return self.bcj[i]

    def union(self, i, j):
        fi, fj = self.find(i), self.find(j)
        if fi != fj:
            self.bcj[fj] = fi
            self.res[fi] += self.res[fj]
            del self.res[fj]

    def addNum(self, val: int) -> None:
        if self.find(val) in self.res:  # 如果已经存在直接返回
            return
        self.res[val] = 1  # 首先把他记为1 然后看前一位后一位是否存在
        if val > 0 and self.find(val - 1) in self.res:
            self.union(val - 1, val)
        if val < 10001 and self.find(val + 1) in self.res:
            self.union(val, val + 1)

    def getIntervals(self) -> List[List[int]]:
        return sorted([[k, k + v - 1] for k, v in self.res.items()])