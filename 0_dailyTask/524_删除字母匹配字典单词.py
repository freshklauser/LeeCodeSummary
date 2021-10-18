# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/9/14 21:45
# @FileName : 524_删除字母匹配字典单词.py
# @SoftWare : PyCharm


import collections


class Solution:

    def findLongestWord(self, s, dictionary):
        dict_sorted = sorted(dictionary, key=lambda x: (-len(x), x))
        print(dict_sorted)
        for i, word in enumerate(dict_sorted):
            if (len(s) < len(word)) or (len(s) == len(word) and s != word):
                continue
            status = self.is_valid_match(word, s)
            if status:
                return word
        return ''

    @staticmethod
    def is_valid_match(word, org_str):
        elements = collections.deque(word)
        for i, s in enumerate(org_str):
            if s == elements[0]:
                elements.popleft()
                if not elements:
                    return True
        return False if elements else True


if __name__ == '__main__':
    s = "abpcplea"
    dictionary = ["ale","apple","monkey","plea", "apple", "bcpla"]
    inst = Solution()
    res = inst.findLongestWord(s, dictionary)
    print(res)
    #
    # print(inst.is_valid_match(dictionary[2], s))

# if word in s:
#     # 如果当前匹配到的字符长度不低于当前的最大长度，则添加到结果列表中
#     # 如果当前匹配的比已匹配的最长长度小，则不添加结果，继续下一轮循环
#     if len(word) < max_len:
#         continue
#     max_len = len(word)
#     res.update({word: (len(word), dict_map[word])})