# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/24 22:29
# @FileName : 430_扁平化多级双向链表.py
# @SoftWare : PyCharm


"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

class Solution:
    def flatten(self, head):

        if not head: return head
