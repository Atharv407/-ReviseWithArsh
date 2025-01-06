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
