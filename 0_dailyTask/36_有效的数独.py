# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/17 21:26
# @FileName : 36_有效的数独.py
# @SoftWare : PyCharm


from collections import Counter

from pprint import pprint


class Solution:
    def isValidSudoku(self, board):
        rows = [[0] * 9 for _ in range(9)]
        cols = [[0] * 9 for _ in range(9)]
        boxes = [[[0] * 9 for _ in range(3)] for _ in range(3)]

        for i in range(9):
            for j in range(9):
                val = board[i][j]
                if val == '.':
                    continue
                val = int(val)
                rows[i][val - 1] += 1
                cols[j][val - 1] += 1
                boxes[i//3][j//3][val - 1] += 1
                if rows[i][val - 1] > 1 or cols[j][val - 1] > 1 \
                        or boxes[i//3][j//3][val - 1] > 1:
                    return False
        return True


if __name__ == '__main__':
    board = [
        ["5", "3", ".", "1", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
    inst = Solution()
    seq = board[1]
    # print(inst.is_valid_unit(seq))
    print(inst.isValidSudoku(board))