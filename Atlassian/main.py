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
        
