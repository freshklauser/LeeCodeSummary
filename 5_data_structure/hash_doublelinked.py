# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/11/9 22:24
# @FileName : hash_doublelinked.py
# @SoftWare : PyCharm


class DoubleLinkedList:
    def __init__(self, key, val, prev, next):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next


class LRUCache:

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = DoubleLinkedList()

    def get(self, key):
        pass

    def put(self, key, value):
        pass





# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)