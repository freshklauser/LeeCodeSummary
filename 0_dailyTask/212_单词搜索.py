# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/16 21:51
# @FileName : 212_单词搜索.py
# @SoftWare : PyCharm



# class Solution:
#     def findWords(self, board, words):
#         m = len(board)
#         n = len(board[0])
#         visited = [[False] * n for _ in range(m)]
#         direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#         res = []
#
#         def dfs(i, j, idw, visited):
#             if idx == len(word):
#                 res.append(word_)
#                 return
#             if idx > len(word):
#                 return
#             for x, y in direction:
#                 ti = i + x
#                 tj = j + y
#                 print(idw, 'dddddddddd', word, '')
#                 if 0 <= ti < m and 0 <= tj < n \
#                         and board[ti][tj] == word[idw] \
#                         and (ti, tj) not in visited:
#                     visited[ti][tj] = True
#                     idw += 1
#                     dfs(ti, tj, idw, visited)
#                     idw -= 1
#                     visited[ti][tj] = False
#
#         for word_ in words:
#             word = list(word_)
#             for idx in range(m):
#                 for idy in range(n):
#                     if word[0] == board[idx][idy]:
#                         visited[idx][idy] = True
#                         dfs(idx, idy, 1, visited)
#                         visited[idx][idy] = False
#         print(res)
#         return res

from collections import Counter


class Solution:
    def findWords(self, board, words):
        rows = len(board)
        cols = len(board[0])
        visited = [[False] * cols for _ in range(rows)]
        direction = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        res = []

        def dfs(i, j, visited, word, word_idx):
            if word_idx == len(word):
                return True
            if i < 0 or i >= rows or j < 0 or j >= cols \
                    or word[word_idx] in visited:
                return False
            if board[i][j] == word[word_idx]:
                word_idx += 1
                visited[i][j] = True
                for dx, dy in direction:
                    ti = i + dx
                    tj = j + dy
                    if dfs(ti, tj, visited, word, word_idx):
                        return True
                visited[i][j] = False
            return False

        board_cnts = dict()
        for r in range(rows):
            for c in range(cols):
                if board[r][c] not in board_cnts.keys():
                    board_cnts[board[r][c]] = 0
                board_cnts[board[r][c]] += 1

        for word in words:
            word_cnts = Counter(word)
            is_valid = True
            for k in word_cnts:
                if k not in board_cnts:
                    is_valid = False
                    break
                if word_cnts[k] > board_cnts[k]:
                    is_valid = False
                    break
            if not is_valid:
                continue

            matched = False
            for r in range(rows):
                for c in range(cols):
                    if dfs(r, c, visited, word, 0):
                        res.append(word)
                        matched = True
                        break
                if matched:
                    break
        return res


if __name__ == '__main__':


    board = [["o", "a", "a", "n"], ["e", "t", "a", "e"], ["i", "h", "k", "r"]]
    words = ["oath", "pea", "eat", "rain"]
    board = [["b"], ["a"], ["b"], ["b"], ["a"]]
    words = ["baa", "abba", "baab", "aba"]
    inst = Solution()
    print(inst.findWords(board, words))
