# -*- coding: utf-8 -*-
# @Author   : Administrator
# @DateTime : 2021/10/17 20:29
# @FileName : tranversal.py
# @SoftWare : PyCharm

from collections import deque

from binary_tree_init import init_bst


class Traversal:
    def __init__(self):
        self.bst_values = []

    @staticmethod
    def layer_order_traversal(root):
        """
        BFS层序遍历, 使用队列
        :param root:
        :return:
        """
        result = []
        dq = deque()
        dq.append(root)
        while dq:
            cur_length = len(dq)
            level = []
            for _ in range(cur_length):
                cur = dq.popleft()
                if not cur:
                    continue
                dq.append(cur.left)
                dq.append(cur.right)
                level.append(cur.data)
            if level:
                result.append(level)
        return result

    def pre_order_traversal(self, root, method=None):
        if method == 'recurse':
            return self.pre_order_traversal__recurse(root)
        if method == 'dfs':
            return self.pre_order_traversal__dfs(root)
        return self.pre_order_traversal__stack(root)

    def mid_order_traversal(self, root, method=None):
        if method == 'recurse':
            return self.mid_order_traversal__recurse(root)
        if method == 'dfs':
            return self.mid_order_traversal__dfs(root)
        return self.mid_order_traversal__stack(root)

    def post_order_traversal(self, root, method=None):
        if method == 'recurse':
            return self.post_order_traversal__recurse(root)
        if method == 'dfs':
            return self.post_order_traversal__dfs(root)
        return self.post_order_traversal__stack(root)

    def pre_order_traversal__recurse(self, root):
        """
        前序遍历， 递归实现, 二叉树遍历最易理解和实现的版本
        三部曲：
        1)  遍历根节点值
        2） 左子节点递归调用
        3） 右子节点递归调用
        :param root: 二叉搜索树
        :return:
        """
        if not root:
            return []
        return [root.data] \
               + self.pre_order_traversal__recurse(root.left) \
               + self.pre_order_traversal__recurse(root.right)

    @staticmethod
    def pre_order_traversal__dfs(root):
        """ 通用模板：dfs 递归实现前序遍历 """

        def dfs(cur_node):
            if not cur_node:
                return
            result.append(cur_node.data)
            dfs(cur_node.left)
            dfs(cur_node.right)

        result = []
        dfs(root)
        return result

    @staticmethod
    def pre_order_traversal__stack(root):
        """ 最常用模板: 栈实现，非递归，性能最优 """
        if not root:
            return []
        result = []
        stack = []
        cur_node = root
        while stack or cur_node:
            while cur_node:
                # 添加根节点到结果表的同时当前节点遍历入栈，并指针移动到当前节点的左节点
                result.append(cur_node.data)
                stack.append(cur_node)
                cur_node = cur_node.left
            cur_node = stack.pop()
            cur_node = cur_node.right
        return result

    @staticmethod
    def mid_order_traversal__stack(root):
        if not root:
            return []
        result = []
        stack = []
        cur_node = root
        while stack or cur_node:
            while cur_node:
                # 当前节点遍历入栈，同时指针移动到当前节点的左节点
                stack.append(cur_node)
                cur_node = cur_node.left
            cur_node = stack.pop()
            result.append(cur_node.data)
            cur_node = cur_node.right
        return result

    @staticmethod
    def post_order_traversal__stack(root):
        if not root:
            return []
        result = []
        stack = []
        cur_node = root
        while stack or cur_node:
            while cur_node:
                result.append(cur_node.data)
                stack.append(cur_node)
                cur_node = cur_node.right
            cur_node = stack.pop()
            cur_node = cur_node.left
        return result[::-1]

    def mid_order_traversal__recurse(self, root):
        if not root:
            return []
        return self.mid_order_traversal__recurse(root.left) \
               + [root.data] \
               + self.mid_order_traversal__recurse(root.right)

    def post_order_traversal__recurse(self, root):
        if not root:
            return []
        return self.post_order_traversal__recurse(root.left) \
               + self.post_order_traversal__recurse(root.right) + [root.data]

    @staticmethod
    def mid_order_traversal__dfs(root):

        def dfs(cur_node):
            if not cur_node:
                return
            dfs(cur_node.left)
            result.append(cur_node.data)
            dfs(cur_node.right)

        result = []
        dfs(root)
        return result

    @staticmethod
    def post_order_traversal__dfs(root):

        def dfs(cur_node):
            if not cur_node:
                return
            dfs(cur_node.left)
            dfs(cur_node.right)
            result.append(cur_node.data)

        result = []
        dfs(root)
        return result


if __name__ == '__main__':
    b_tree = init_bst()
    print('前序：', Traversal().pre_order_traversal__stack(b_tree))
    print('中序：', Traversal().mid_order_traversal__recurse(b_tree))
    print('中序stack：', Traversal().mid_order_traversal__stack(b_tree))
    print('后序：', Traversal().post_order_traversal__dfs(b_tree))
    print('后序rec：', Traversal().post_order_traversal__recurse(b_tree))
    print('后序stack：', Traversal().post_order_traversal__stack(b_tree))
    print('---'*50)
    print(Traversal().pre_order_traversal(b_tree))
    print(Traversal().layer_order_traversal(b_tree))
    # print(pre_order(b_tree))