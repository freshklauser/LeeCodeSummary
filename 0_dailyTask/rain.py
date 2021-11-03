# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/11/3 21:20
# @FileName : rain.py
# @SoftWare : PyCharm
import heapq


class Solution:

    def trapRainWater(self, heightMap):
        m = len(heightMap)
        n = len(heightMap[0])
        visited = [[0] * n for _ in range(m)]
        src_points = []  # (i, j) 元祖作为元素
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                rds = (heightMap[i - 1][j], heightMap[i + 1][j],
                       heightMap[i][j - 1], heightMap[i][j + 1])
                print(rds, '-->', heightMap[i][j])
                if heightMap[i][j] < min(rds):
                    print(i, j)
                    src_points.append((i, j))
                    visited[i][j] = 1
        print(src_points)
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

        def dfs(heightMap, point, visited):
            i, j = point
            queue = []

    def check_neighbour(self, heightMap, ci, cj, directions):
        if i == 0 or j == 0:
            return False
        # neighbours = (
        #     heightMap[i - 1][j], heightMap[i + 1][j],
        #     heightMap[i][j - 1], heightMap[i][j + 1]
        # )
        neighbours = [(ci + dx, cj + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        for i, j in neighbours:
            pass

if __name__ == '__main__':
    nums = [
        [11, 4, 13, 9, 5, 12],
        [3, 2, 1, 13, 2, 4],
        [3, 1, 3, 2, 6, 1],
        [2, 5, 3, 2, 3, 1]
    ]
    inst = Solution()
    # inst.trapRainWater(nums)
    priority_queue = [(11,2,12), (1, 3, 23), (1, 1, 1), (42, 0, 12), (1, 1, 43)]
    heapq.heapify(priority_queue)
    # print(a, type(a))
    # print(heapq.heappop(a))
    # print(heapq.nsmallest(2, a, key=lambda x: x[2]))
    # print(heapq.heappop(heapq.nsmallest(1, a, key=lambda x: x[2])))
    # print(a)

    # priority_queue = []
    # for item in a:
    #     heapq.heappush(priority_queue, (item[0], item[1], item[2]))
    print(priority_queue)
    print(heapq.heappop(priority_queue))
    print(priority_queue)
    print(heapq.heappop(priority_queue))
    print(priority_queue)
    print(heapq.heappop(priority_queue))
    print(priority_queue)
# import heapq
#
#
# class PriorityQueue:
#
#     def __init__(self):
#         self._queue = []
#         self._index = 0
#
#     def push(self, item, priority):
#         # 传入两个参数，一个是存放元素的数组，另一个是要存储的元素，这里是一个元组。
#         # 由于heap内部默认有小到大排，所以对priority取负数
#         heapq.heappush(self._queue, (-priority, self._index, item))
#         self._index += 1
#
#     def pop(self):
#         return heapq.heappop(self._queue)
#
#
# if __name__ == '__main__':
#     q = PriorityQueue()
#
#     q.push('lenovo', 1)
#     q.push('Mac', 5)
#     q.push('ThinkPad', 2)
#     q.push('Surface', 3)
#     print(q.pop())