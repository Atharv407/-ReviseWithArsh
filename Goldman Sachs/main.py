#Maximum Sum BST
class Solution:
    def maxSumBST(self, root: Optional[TreeNode]) -> int:
        def sbst(node) :
            s = node.val
            minval = s
            maxval = s
            chk = True
            if node.left :
                l = sbst(node.left)
                if not (l and l[2] < node.val) :
                    chk = False
                else :
                    s += l[0]
                    minval = l[1]
            if node.right :
                r = sbst(node.right)
                if not (r and r[1] > node.val) :
                    chk = False
                else :
                    s += r[0]
                    maxval = r[2]
            if chk :
                self.res = max(self.res, s)
                # print(node.val, s, minval, maxval)
                return [s, minval, maxval]

        self.res = 0
        sbst(root)
        return self.res
            
            