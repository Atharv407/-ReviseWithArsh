# Who is the winner ? 
class Solution :
    def findTheWinner(self, n: int, k: int) -> int:
        if n==1: 
            return 1
        elif k==1:
            return n
        elif n>=k:
            next_n = n - n//k
            z = self.findTheWinner(next_n, k) - 1
            x = (z-n%k + next_n) % next_n
            return x + x//(k-1) + 1
        else:
            return (k + self.findTheWinner(n-1, k) -1) % n + 1
        
#Envelopes and Dolls
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes = sorted(envelopes,key = lambda x: [x[0], -x[1]])
        print(envelopes)
        dp = [envelopes[0][1]]
        for env in envelopes[1:]:
            print(dp)
            if dp[-1] < env[1]:
                dp.append(env[1])
            else:
                i = bisect_left(dp,env[1])
                dp[i] = env[1]
        return len(dp)

#Image Smoother 
class Solution:
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        m, n = len(img), len(img[0])
        res = [[0]*n for i in img]
        for i in range(m) :
            for j in range(n) :
                curr = img[i][j]
                ct = 1
                if j > 0 :
                    curr += img[i][j-1]
                    ct += 1
                    if i > 0 :
                        curr += img[i-1][j-1]
                        ct += 1
                    if i < m-1 :
                        curr += img[i+1][j-1]
                        ct += 1
                if j < n-1 :
                    curr += img[i][j+1]
                    ct += 1
                    if i > 0 :
                        curr += img[i-1][j+1]
                        ct += 1
                    if i < m-1 :
                        curr += img[i+1][j+1]
                        ct += 1
                if i > 0 :
                    curr += img[i-1][j]
                    ct += 1
                if i < m-1 :
                    curr += img[i+1][j]
                    ct += 1
                res[i][j] = math.floor(curr/ct)
        return res

#Minimum moves to equal array elements
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        nums.sort()
        mid = 0
        if len(nums)%2 == 1 :
            mid = nums[int((len(nums)-1)/2)]
        else :
            mid =  nums[int((len(nums))/2)-1]
        res = 0
        
        for i in nums :
            res += abs(mid - i)
        return res
    
#Counting nice sub arrays
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        res = 0
        i, j, nodd = 0, -1, 0
        l = len(nums)
        for i in range(l) :
            while nodd < k and j < l-1 :
                j += 1
                if nums[j] % 2 == 1 :
                    nodd +=1
            if nodd == k :
                res += 1
                for tj in range(j+1, l) :

                    if nums[tj] % 2 == 1 :
                        break
                    res += 1
            if nums[i] %2 == 1 :
                nodd -= 1
        return res
            
#Repeated DNA Sequences
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        subs = Counter()
        res = set()
        for i in range(len(s) - 9) :
            if s[i:i+10] in res : continue
            subs[s[i:i+10]] += 1
            if subs[s[i:i+10]] > 1 : 
                res.add(s[i:i+10])
        return list(res)
        
#Count the Number of Incremovable Subarrays 1
class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def chk(i, j) :
            for a in range(1, i) : 
                if nums[a] <= nums[a-1] : 
                    return False
            for a in range(j+2, len(nums)) : 
                if nums[a] <= nums[a-1] : 
                    return False
            if i -1 > -1 and j +1 < len(nums) and nums[i-1] >= nums[j+1] : return False
            return True
        res = 0
        for i in range(len(nums)) : 
            for j in range(i, len(nums)) :
                if chk(i, j) : 
                    res += 1
                    # print(i, j)
                # else : print(i, j)
        return res
        
#Max Product of Length of Two Palindromic Sequences
class Solution:
    def maxProduct(self, s: str) -> int:
        
        def disjoint(i, j) : 
            for a in pals[i] : 
                if a in pals[j] : return False
            return True
        pal = [[set()]*len(s) for i in s]

        for i in range(len(s)-1, -1, -1) :
            pal[i][i] = {(i,)}
            for j in range(i+1, len(s)) :
                pal[i][j] = {(i,)}
                if s[i] == s[j] : 
                    pal[i][j].add((i, j))
                    for sub in pal[i+1][j-1] :
                        nsub = (i,) + sub + (j,)
                        pal[i][j].add(nsub)
                pal[i][j].update(pal[i+1][j])
                pal[i][j].update(pal[i][j-1])
                
        pals = list(pal[0][-1])
        pals.sort(key = lambda x : len(x), reverse = True)
        res = 1

        for i in range(len(pals)) : 
            for j in range(i+1, len(pals)) :
                if disjoint(i, j) : res = max(res, len(pals[i])*len(pals[j]))

        return res

#Wiggle Sort
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ln = len(nums)
        nums.sort()
        m = int((ln-1)/2)
        nums[::2], nums[1::2] = nums[m::-1], nums[:m:-1]

#Shopping Offers
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        def cal(needs) : 
            if str(needs) in net : 
                return net[str(needs)]
            res = 0
            for i in range(len(price)) :
                res += price[i]*needs[i]
            for i in special : 
                nneeds = needs[:]
                j = 0
                while j < len(needs) :
                    if nneeds[j] < i[j] : break
                    nneeds[j] -= i[j]
                    j += 1
                if j == len(needs) : res = min(res, i[-1] + cal(nneeds))
            net[str(needs)] = res
            return res

        net = {}
        net[str([0]*len(needs))] = 0
        
        return cal(needs)
        
#Minimum Cost to Convert String 1
class Solution:
    def minimumCost(self, s: str, t: str, o: List[str], c: List[str], cost: List[int]) -> int:
        dist = [[float('inf')]*26 for i in range(26)]

        for i in range(26) : 
            dist[i][i] = 0

        for i in range(len(o)) :
            a = ord(o[i]) - ord('a')
            b = ord(c[i]) - ord('a')
            dist[a][b] = min(dist[a][b], cost[i])
        
        for k in range(26) : 
            for i in range(26) :
                for j in range(26) : 
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        res = 0
        for i in range(len(s)) :
            res += dist[ord(s[i]) - ord('a')][ord(t[i]) - ord('a')]

        if res >= float('inf') : return -1
        return res