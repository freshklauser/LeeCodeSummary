# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/17 20:30
# @FileName : binary_tree_init.py
# @SoftWare : PyCharm


class BinaryTree:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def get(self):
        return self.data

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def setLeft(self, node):
        self.left = node

    def setRight(self, node):
        self.right = node


def init_bst():
    binaryTree = BinaryTree(0)
    binaryTree.setLeft(BinaryTree(11))
    binaryTree.setRight(BinaryTree(22))
    binaryTree.getLeft().setLeft(BinaryTree(23))
    binaryTree.getLeft().setRight(BinaryTree(4))
    binaryTree.getRight().setLeft(BinaryTree(15))
    binaryTree.getRight().setRight(BinaryTree(6))
    return binaryTree


if __name__ == '__main__':
    print(init_bst().getLeft().get())