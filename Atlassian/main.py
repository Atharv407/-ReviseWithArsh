#Assign Cookies
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        i, j = 0, 0
        res = 0
        for i in range(len(g)) :
            while j < len(s) and s[j]  < g[i] :
                j +=1 
            if j == len(s) : break
            res += 1
            j += 1
        return  res
    
#Kth Largest Element in a Stream
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.stream = nums
        self.stream.sort()
        self.k = k 
        

    def add(self, val: int) -> int:
        point = bisect.bisect(self.stream, val)
        self.stream.insert(point, val)
        # print(self.stream)
        return self.stream[-self.k]

#Find the Distance Value Between Two Arrays
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        res = 0
        for i in arr1 :
            for j in arr2 :
                if abs(i-j) <= d : 
                    res -= 1
                    break
            res += 1
        return res
                
#LRU Cache
class LRUCache:

    def __init__(self, capacity: int):
        self.cache = {}
        self.cap = capacity
        self.s = 0

    def get(self, key: int) -> int:
        if key in self.cache :
            val = self.cache.pop(key)
            self.cache[key] = val
            return val
        else :
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache :
            val = self.cache.pop(key)
            self.cache[key] = value
        else :
            if self.s == self.cap :
                self.cache.pop(next(iter(self.cache)))
                self.s -= 1
            self.cache[key] = value
            self.s += 1
        # print(self.cache)
        
#The Number of Good Subsets
class Solution:
    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        def val(n) :
            if n in vals : return vals[n]
            res = ''
            for i in valid : 
                if i <= n and n%i == 0 : res = '1' + res
                else : res = '0' + res
            vals[n] = int(res, 2)
            return vals[n]

        good = {2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30}
        valid = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
        mod = (10**9 + 7)
        n = Counter(nums)
        ones = n[1]
        vals = {}
        dp = Counter()

        for i in n : 
            if not i in good : continue
            vi = val(i)
            for j in list(dp.keys()):
                if dp[j] > 0 and j & vi == 0 :
                    dp[j|vi] = (dp[j|vi] + dp[j]*n[i])%mod     
            dp[vi] += n[i]

        res = sum(dp.values())

        if ones > 0 : res = res*(pow(2, ones)%mod)
        return int(res)%mod
    
#Find Beautiful Indices in the Given Array I
class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        ah, bh, sah, sbh = 0, 0, 0, 0
        la, lb = len(a), len(b)
        mod = 10**9 + 7
        if len(a) > len(s) or len(b) > len(s) : return []

        for i in range(len(a)) :
            ah = (ah*29)%mod + ord(a[i]) - ord('a') + 1
            sah = (sah*29)%mod + ord(s[i]) - ord('a') + 1
        for i in range(len(b)) :
            bh = (bh*29)%mod + ord(b[i]) - ord('a') + 1
            sbh = (sbh*29)%mod + ord(s[i]) - ord('a') + 1

        a = []
        b = []
        if sah == ah : a.append(0)
        if sbh == bh : b.append(0)
        for i in range(1, len(s)-la+1) :
            sah -= ((ord(s[i-1]) - ord('a') + 1)*pow(29, (la-1)))%mod
            sah = (sah*29)%mod + (ord(s[i+la-1]) - ord('a') + 1)
            if sah == ah : a.append(i)

        for i in range(1, len(s)-lb+1) :
            sbh -= ((ord(s[i-1]) - ord('a') + 1)*pow(29, (lb-1)))%mod
            sbh = (sbh*29)%mod + (ord(s[i+lb-1]) - ord('a') + 1)
            if sbh == bh : b.append(i)
        
        res = []
        ai, bi = 0, 0
        while ai < len(a) and bi < len(b) :
            if a[ai] - b[bi] > k : bi += 1
            elif b[bi] - a[ai]  > k : ai += 1
            else : 
                res.append(a[ai])
                ai += 1

        return res

#Count Words Obtained After Adding a Letter
class Solution:
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        def chk(i, t) :
            if not i in dic : return False
            for w in dic[i] :
                if t | w == t : return True
            return False

        dic = {}
        mod = 10**9 + 7
        chart = {}
        for i in range(26) :
            chart[chr(i + ord('a'))] = pow(2, (i+1))%mod 
        for i in startWords :
            l = len(i)
            bn = 0
            for j in i :
                bn += chart[j]
            if l not in dic : dic[l] = set()
            dic[l].add(bn)
    
        res = 0
        for i in targetWords :
            lt = len(i)
            if lt-1 not in dic : continue
            bn = 0
            for j in i :
                bn += chart[j]
            for j in i :
                if bn - chart[j] in dic[lt-1] :
                    res += 1
                    break
            
        return res
    
