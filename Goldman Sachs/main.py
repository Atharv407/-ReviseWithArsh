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


#Flip Matrix
class Solution:

    def __init__(self, m: int, n: int):
        self.mat = defaultdict(list)
        self.m = m
        self.n = n
        self.mweights = [n]*m
        self.r = list(range(self.m))
        self.curr = [0, 0]
        

    def flip(self) -> List[int]:
        i = random.choices(self.r, weights = self.mweights, k = 1)[0]
        jst = self.curr[1] if i == self.curr[0] else 0
        j = random.randint(jst, self.n-1)
        
        res = [i, j] if not (i, j) in self.mat else self.mat[(i, j)]

        burned = [self.curr[0], self.curr[1]] 
        if (self.curr[0], self.curr[1]) in self.mat :
            burned =  self.mat[(self.curr[0], self.curr[1])]

        self.mat[(i, j)] = burned[:]
        self.mweights[self.curr[0]] -= 1

        if self.curr[1] == self.n-1 :
            self.curr = [self.curr[0] + 1, 0]
        else :
            self.curr[1] += 1

        return res
    
    def reset(self) -> None:
        self.mat = defaultdict(list)
        self.mweights = [self.n for i in range(self.m)]
        self.curr = [0, 0]


#Combinations in a Phone Number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        let = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l'], ['m', 'n', 'o'], ['p', 'q', 'r', 's'], ['t', 'u', 'v'], ['w', 'x', 'y', 'z']] 
        res = ['']
        if not digits : return []
        for i in digits :
            nres = []
            for l in let[int(i)-2] :
                for w in res :
                    nres.append(w + l)

            res = nres

        return res


#Find Consecutive Integers from a Data Stream
class DataStream:

    def __init__(self, value: int, k: int):
        self.value, self.k, self.ck = value, k, 0
        

    def consec(self, num: int) -> bool:
        if num == self.value : 
            self.ck += 1
            return self.ck >= self.k 
        else :
            self.ck = 0
        return False
        
#K - divisible Elements Subarrays
class Solution:
    def countDistinct(self, nums: List[int], k: int, p: int) -> int:
        res = set()
        for i in range(len(nums)) :
            curr = ''
            ki = 0
            for j in range(i, len(nums)) :
                if nums[j] %p == 0 :
                    if ki == k :
                        break
                    else :
                        ki += 1
                        curr += ' ' + str(nums[j])
                        res.add(curr)
                else :
                    curr += ' ' + str(nums[j])
                    res.add(curr)
        # print(res)
        return len(res)

#Number of People Aware of a Secret
class Solution:
    def peopleAwareOfSecret(self, n: int, delay: int, forget: int) -> int:
        dn = [0 for i in range(delay + 1)]
        fn = [0 for i in range(forget )]
        dn[-1] = 1
        fn[-1] = 1
        mod = (10**9 + 7)
        for i in range(n-1) :
            dn[0] -= fn[0]
            fn[0] = 0
            fn = fn[1:] + fn[:1]
            fn[-1] = (dn[0] + dn[1])%mod
            dn = dn[1:] + dn[:1]
            dn[-1] += dn[0]%mod
            dn[0] = dn[-1]%mod
        s = 0 
        for i in dn : 
            s = (s + i)%mod
        return s
            
        
#Map of Highest Peak
class Solution:
    def highestPeak(self, height: List[List[int]]) -> List[List[int]]:
        # height = isWater
        q = deque()
        for i in range(len(height)) :
            for j in range(len(height[0])) :
                if height[i][j] == 1 :
                    q.append([i,j,0])
                    height[i][j] = 0
                else :
                    height[i][j] = None
        res = 0
        while q :
            # print(q)
            i, j, h = q.popleft()
            if i -1 > -1 and height[i-1][j] == None :
                height[i-1][j] = h + 1
                q.append([i-1, j, h+1])
            if j+1 < len(height[0]) and height[i][j+1] == None :
                height[i][j+1] = h + 1
                q.append([i, j+1, h+1])
            if j-1 > -1 and height[i][j-1] == None :
                height[i][j-1] = h + 1
                q.append([i, j-1, h+1])
            if i +1 < len(height) and height[i+1][j] == None :
                height[i+1][j] = h + 1
                q.append([i+1, j, h+1])
        return height
        