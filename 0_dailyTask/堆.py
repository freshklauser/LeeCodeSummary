# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/12 20:31
# @FileName : 堆.py
# @SoftWare : PyCharm

import heapq

seqs = [5, 2, 4, 7, 11, 3, 12, 65, 34]
heapq.heapify(seqs)
print(seqs)     # [2, 5, 3, 7, 11, 4, 12, 65, 34]
heapq.heapreplace(seqs, 200)
print(seqs)     # [3, 5, 4, 7, 11, 200, 12, 65, 34]
# # seqs = [2, 3, 5, 1, 54, 23, 132]
# heap = []
# for v in seqs:
#     heapq.heappush(heap, v)
# print(heap)     # [2, 5, 3, 7, 11, 4, 12, 65, 34]
#
# heapq.heapify(seqs)
# print(seqs)     # [2, 5, 3, 7, 11, 4, 12, 65, 34]
#
# print([heapq.heappop(seqs) for _ in range(len(seqs))])
# # out：[2, 3, 4, 5, 7, 11, 12, 34, 65]


print(seqs)

