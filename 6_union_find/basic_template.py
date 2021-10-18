# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/5/24 21:50
# @FileName : basic_template.py
# @SoftWare : PyCharm

"""
    使用一个数组 parent 存储每个变量的连通分量信息，其中的每个元素表示当前变量所在的
连通分量的父节点信息，如果父节点是自身，说明该变量为所在的连通分量的根节点。一开
始所有变量的父节点都是它们自身
    对于合并操作，我们将第一个变量的根节点的父节点指向第二个变量的根节点；
    对于查找操作，我们沿着当前变量的父节点一路向上查找，直到找到根节点。

sample: https://leetcode-cn.com/leetbook/read/disjoint-set/ow9p9t/
"""


class UnionFind(object):

    def __init__(self):
        self.parents = list(range(26))

    def find(self, index):
        if index == self.parents[index]:
            return index
        self.parents[index] = self.find(self.parents[index])
        return self.parents[index]

    def union(self, index1, index2):
        self.parents[self.find(index1)] = self.find(index2)


def equationPossible(equations):
    uf = UnionFind()
    for st in equations:
        if st[1] == '=':
            # 等式合并
            index1 = ord(st[0]) - ord('a')
            index2 = ord(st[3]) - ord('a')
            uf.union(index1, index2)
    for st in equations:
        if st[1] == '!':
            index1 = ord(st[0]) - ord('a')
            index2 = ord(st[3]) - ord('a')
            print(uf.find(index1))
            print(uf.find(index2))
            if uf.find(index1) == uf.find(index2):
                return False
    return True


if __name__ == '__main__':
    uf = UnionFind()
    print(uf.parents)
    print(uf.union(2, 5))
    print(uf.find(2))
    print()
    equationPossible(["a==b", "b!=c", "c==a"])
