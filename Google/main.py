#Russian Doll Envelopes
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes = sorted(envelopes,key = lambda x: [x[0], -x[1]])
        # print(envelopes)
        dp = [envelopes[0][1]]
        for env in envelopes[1:]:
            # print(dp)
            if dp[-1] < env[1]:
                dp.append(env[1])
            else:
                i = bisect_left(dp,env[1])
                dp[i] = env[1]
        return len(dp)

#Destroying Asteroids
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids.sort()
        for i in asteroids : 
            if mass < i : return False
            mass += i
        return True

#Find the City With the Smallest Number of Neighbors at a Threshold Distance
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], thres: int) -> int:
        graph = [[float('inf')]*n for i in range(n)]
        for [i, j, d] in edges :
            graph[i][j] = d
            graph[j][i] = d
        for k in range(n) :
            graph[k][k] = 0
            for i in range(n) :
                for j in range(n) :
                    #  min(
                    if graph[i][j] > graph[i][k] + graph[k][j] : graph[i][j] = graph[i][k] + graph[k][j]
        res = math.inf
        ans = 0
        # print(graph)
        for i in range(n) :
            curr = 0
            for j in range(n) : 
                if graph[i][j] <= thres : curr += 1
            if res >= curr : 
                res = curr 
                ans = i
            # print(curr)
        return ans
