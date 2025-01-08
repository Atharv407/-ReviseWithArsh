# First non-repeating character in a stream
class Solution:
    def firstUniqChar(self, s: str) -> int:
        chars = Counter(s)
        for k in chars : 
            if chars[k] == 1 : 
                return s.find(k)
        return -1
            
#Longest Mountain
class Solution:
    def longestMountain(self, arr: List[int]) -> int:
        curr = 1
        res = 0
        incr = True
        if len(arr) < 3 : return 0
        for i in range(1, len(arr)) :
            if arr[i] > arr[i-1] :
                if incr == True : curr += 1
                else : 
                    res = max(res, curr)
                    curr = 2
                    incr = True
            elif arr[i] < arr[i-1] : 
                if incr == False : curr += 1
                else : 
                    if curr > 1 :
                        curr += 1
                        incr = False
                    else : curr = 1
            else : 
                if incr == False :
                    res = max(res, curr)
                incr = True
                curr = 1
        if incr == False : res = max(res, curr)
        return res

#Maximum sum of distinct subarrays with length k
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        subarr = {}
        i = 0
        res = 0
        j = 0
        s = 0
        while i < len(nums)-k+1 : 
            if j < i + k :
                # j += 1
                if nums[j] in subarr :
                    # subarr = set()
                    i = subarr[nums[j]]
                    subarr = {}
                    s = 0
                    j = i - 1
                else : 
                    subarr[nums[j]] = j
                    s += nums[j]
                j += 1
            else : 
                # print(i, j, subarr)
                res = max(res, s)
                subarr.pop(nums[i])
                s -= nums[i]
                i = i + 1
        return res
                
#Delete N nodes after M nodes of a linked list
# your task is to complete this Function
# Function shouldn't return anything

'''
class Node:
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None
'''
class Solution:
    def linkdelete(self, head, n, m):
        curr = head
        while True :
            # curr = head
            for i in range(m-1) :
                curr = curr.next
                if curr == None : return
            # print(curr.data)
            nex = curr.next
            for j in range(n) :
                nex = nex.next
                if nex == None : return
            curr.next = nex
            # print(nex.data)
            curr = nex



#{ 
 # Driver Code Starts
class Node:
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    # Function to initialize head
    def __init__(self):
        self.head = None

    # Function to insert a new node at the beginning
    def push(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node

    # Utility function to print the linked list
    def printList(self):
        temp = self.head
        while temp:
            print(temp.data, end=" ")
            temp = temp.next
        print("")  # Newline after printing the linked list


if __name__ == '__main__':
    t = int(input())  # Number of test cases
    while t > 0:
        llist = LinkedList()
        values = input().strip().split()
        for i in reversed(values):  # Reversed input list to preserve order
            llist.push(i)
        n, m = map(int, input().strip().split())  # n: keep, m: delete
        Solution().linkdelete(llist.head, n,
                              m)  # Call the method to modify the list
        llist.printList()
        t -= 1
        print("~")  # Separator for test cases

# } Driver Code Ends