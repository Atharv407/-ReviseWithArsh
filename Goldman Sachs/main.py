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
            

#Employee Priority System
class Solution:
    def findHighAccessEmployees(self, ac: List[List[str]]) -> List[str]:
        dic = {}
        res = set()
        ac.sort(key = lambda x : x[1])
        for i in ac :
            # print(dic)
            if not i[0] in dic :
                dic[i[0]] = [int(i[1])]
            elif int(i[1]) - dic[i[0]][-1]  > 99 :
                dic[i[0]] = [int(i[1])]
            elif int(i[1]) - dic[i[0]][0]  > 99 :
                dic[i[0]] = [dic[i[0]][-1], int(i[1])]
            elif len(dic[i[0]]) == 2 :
                res.add(i[0])
            else :
                dic[i[0]].append(int(i[1]))
        return res
            

#Combination Sum
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def func(p, t, n) :
            if t < p+1 : return []
            mv = 45 - (9-n)*(10-n)/2
            if mv < t : return []
            if n == 1  : return [{t}]
            res = []
            for i in range(p+1, 11 - n) :
                nres = func(i, t-i, n-1) 
                if not nres : continue
                for j in nres :
                    j.add(i)
                res += nres
            return res
        return [list(i) for i in func(0, n, k)]


#Kth Smallest Element Query
class Solution:
    def smallestTrimmedNumbers(self, nums: List[str], q: List[List[int]]) -> List[int]:
        nums = list(enumerate(nums))
        l = len(nums[0][1])
        dp = {}
        for i in range(len(q)) :
            k, trim = q[i]
            if not trim in dp :
                dp[trim] = list(range(len(nums)))
                dp[trim].sort(key = lambda x : [nums[x][1][l-trim:], nums[x][0]])
                #  = nnums
            q[i] = dp[trim][k-1]
        # print(dp)
        return q
    
#Minimize the Maximum of Two Arrays
class Solution:
    def minimizeSet(self, divisor1: int, divisor2: int, uniqueCnt1: int, uniqueCnt2: int) -> int:
        def lcm(a, b):
            return abs(a*b) // math.gcd(a, b)

        h = int(uniqueCnt1*(divisor1)/(divisor1-1) + uniqueCnt2*(divisor2)/(divisor2-1))
        l = uniqueCnt1 + uniqueCnt2
        while l < h : 
            m = (l+h)//2
            nd1, nd2 = int(m/divisor1), int(m/divisor2)               
            dd = lcm(divisor2, divisor1)
            nd = int(m/dd)

            neutrals = m - nd1 - nd2 + nd
            a = nd2 - nd 
            b = nd1 - nd 

            if b + neutrals >= uniqueCnt2 and neutrals - max(0, uniqueCnt2 - b) + a >= uniqueCnt1 :
                h = m
            else : 
                l = m + 1

        return h

