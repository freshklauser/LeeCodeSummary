# 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：head = [1,2,3,4,5], n = 2
# 输出：[1,2,3,5]
#  
# 
#  示例 2： 
# 
#  
# 输入：head = [1], n = 1
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：head = [1,2], n = 1
# 输出：[1]
#  
# 
#  
# 
#  提示： 
# 
#  
#  链表中结点的数目为 sz 
#  1 <= sz <= 30 
#  0 <= Node.val <= 100 
#  1 <= n <= sz 
#  
# 
#  
# 
#  进阶：你能尝试使用一趟扫描实现吗？ 
#  Related Topics 链表 双指针 
#  👍 1734 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        """
        快慢指针，虚拟头结点即哑节点
        :param head:
        :param n:
        :return:
        """
        dummy_head = ListNode(next=head)
        fast = dummy_head
        slow = dummy_head
        while n:
            fast = fast.next
            n -= 1
        while fast.next:
            fast = fast.next
            slow = slow.next
        # fast 走到结尾后，slow的下一个节点为倒数第N个节点
        slow.next = slow.next.next

        return dummy_head.next

        """
        先遍历获取size, 再遍历删除对应节点
        """
        # if head.next is None:
        #     return head.next
        #
        # # 链表size
        # cur = head
        # dsize = 1
        # while cur.next:
        #     dsize += 1
        #     cur = cur.next
        #
        # # 如果删除头结点，则直接返回 head.next (如果不判断的话需要加上 None 的哑节点作为head)
        # if dsize == n:
        #     return head.next
        #
        # # 删除除了头结点之外的节点
        # cur = head
        # for i in range(1, dsize, 1):
        #     pre = cur
        #     cur = cur.next
        #     if i == dsize - n:
        #         pre.next = cur.next
        # return head

# leetcode submit region end(Prohibit modification and deletion)
