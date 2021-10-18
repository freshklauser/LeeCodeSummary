# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/8 21:45
# @FileName : 187-重复的DNA.py
# @SoftWare : PyCharm


class Solution:
    def findRepeatedDnaSequences(self, s):
        n = len(s)
        res = set()
        for i in range(n - 9):
            target_s = s[i: i + 10]
            for j in range(i + 1, n - 9):
                # print(s[i], s[j], 'xxxxxxxx')
                if s[i] != s[j]:
                    continue
                sub_s = s[j: j + 10]
                if sub_s == target_s:
                    res.add(target_s)
                    break
        return res


if __name__ == '__main__':
    s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
    # s = "AAAAAAAAAAA"
    res = Solution().findRepeatedDnaSequences(s)
    print(res)
    from collections import defaultdict
    import collections
    from pprint import pprint
    print([item for item in dir(collections) if not item.startswith('_')])
    collections.Counter
    ['AsyncGenerator', 'AsyncIterable', 'AsyncIterator', 'Awaitable', 'ByteString', 'Callable', 'ChainMap',
     'Collection', 'Container', 'Coroutine', 'Counter', 'Generator', 'Hashable', 'ItemsView', 'Iterable', 'Iterator',
     'KeysView', 'Mapping', 'MappingView', 'MutableMapping', 'MutableSequence', 'MutableSet', 'OrderedDict',
     'Reversible', 'Sequence', 'Set', 'Sized', 'UserDict', 'UserList', 'UserString', 'ValuesView', 'abc', 'defaultdict',
     'deque', 'namedtuple']
