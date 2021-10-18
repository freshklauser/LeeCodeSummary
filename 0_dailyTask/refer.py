class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        def dfs(node):
            if not node:
                return
            if not node.child and not node.next:
                return node 
            elif node.child:
                last = dfs(node.child)
                if last:
                    last.next = node.next
                if node.next:
                    node.next.prev = last
                node.next = node.child
                node.child.prev = node
                node.child = None
                return dfs(last)
            else:
                return dfs(node.next)
        dfs(head)
        return head    

作者：himymBen
链接：https://leetcode-cn.com/problems/flatten-a-multilevel-doubly-linked-list/solution/pythonjava-dfsdi-gui-by-himymben-9098/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。