---
layout: post
title:  "How to Solve A Coding Problem in an Interview When You Are Stuck"
date:   2021-10-31 12:50:00 -0400
categories: default
tags: [Coding, Interview]
---

Sometimes if we have encountered a problem and the solution is not obvious to you, what you will do at this time?

According to my personal interview experience, we can not keep thinking or thinking without talking for a long time. That is a big red flag.

#### My personal thought:

when I am stuck in coming up with a solution immediately, I would start to analyze different test cases to do the inference. Then I can continue talking through the interview.

It could be done with these steps:

**1)** Analyze a small test case with data size 1, data size 2 and try to come up with the solution.

**2)** Analyze data size 3 and more,  and try to find a pattern or solution to that.

**3)** Find the final solution for this problem according to previous two steps' inferences.


I find this is quite helpful, and it is also clear to the interviewer.


#### Example:

Let's give an example: Leetcode problem 1094. Car Pooling.

There is a car with capacity empty seats. The vehicle only drives east (i.e., it cannot turn around and drive west).

You are given the integer capacity and an array trips where trip[i] = [numPassengersi, fromi, toi] indicates that the ith trip has numPassengersi passengers and the locations to pick them up and drop them off are fromi and toi respectively. The locations are given as the number of kilometers due east from the car's initial location.

Return true if it is possible to pick up and drop off all passengers for all the given trips, or false otherwise.


#### Analysis:

**1)** We could start from one data size input.
such as.

trips =[[2,1,5]], capacity = 1;   => 1 < 2, return False

trips = [[[2, 1, 5]]  capacity = 4;   => 4 > 2, return True

trips = [[[2, 1, 5],[3, 3, 7]]  capacity = 4;   => 

[1  &ensp;    5]

&ensp; &nbsp;  [3   &ensp;    7]
     
=> 4 < 2+ 3, return False

Here we might to consider the overlapping and the total passengers accumulated here 


trips = [[[2, 1, 5],[3, 6, 7]]  capacity = 4;   => 


[1  &ensp;    5]
       
&ensp; &ensp; &ensp;  &ensp; [6   &ensp;  7]

for first trip, used  2 passengers, then it ends, we release 2 passengers.  Later for second trip, it uses 3,  the passengers still < 4 all the way,  return True


Here we might to consider the accumulation of passengers when there is no overlapping between two trips and the remainder capacity here 

**2)** use three cases:


trips = [[[2, 1, 5],[2,3, 7], [2, 6, 8]]  capacity = 4;   => 


[1  &ensp;    5]

&ensp; &nbsp;  [3   &ensp;    7]

&ensp; &ensp; &ensp; &nbsp; [ 6 &ensp;    8 ]

For first trip, we use 2 passengers,  then second trip comes, the capacity become 4.  Later the first trip ends, we release 2 passengers. Now the capacity becomes 2 again. Later for third trip, it uses 2,   the capacity become 0 until to the end  return True

Here we can see that, we need to decide when to add the passengers, that is when a trip comes. when a old trip finishes and we decrease the passengers, that is a new trip comes and has no overlapping with the old trip. 

**3)** Therefore, we could use a priority queue (min heap which tracks the end of the trip) to update the trip and decide if the new trip has no overlapping. The alternative way is that, we record the start, end position and use sweeping line algorithm to solve this.

### Code:

(1) Priority queue method

{% highlight python %} 

    import heapq

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        
        trips = sorted(trips, key = lambda e:e[1])
        print ("trips: ", trips)
        hp = []
        
        cur_cap = capacity
        for num_ppl, start, end in trips:
            
            if len(hp) == 0:   # que is empty
                if num_ppl > cur_cap:
                    return False
                heapq.heappush(hp, [end, num_ppl])
                cur_cap -= num_ppl
            else:

                #print ("hp: ", start, hp)
                while (len(hp) and hp[0][0] <= start): # pop
                    tmp_end, tmp_num_ppl = heapq.heappop(hp)
                    cur_cap += tmp_num_ppl
                
                if num_ppl > cur_cap:
                    return False
                #print ("hp11: ", hp)
                heapq.heappush(hp, [end, num_ppl])
                cur_cap -= num_ppl
        return True        

{% endhighlight %}


(2) Sweeping line method:

{% highlight python %} 

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:


        trips = sorted(trips, key = lambda e:e[1])
        
        dict_driving = collections.defaultdict(list)
        
        for num_ppl, start, end in trips:
            dict_driving[start].append([num_ppl, 1])
            dict_driving[end].append([num_ppl, -1])
            
            
        positions = sorted(dict_driving)
        #print ("dict_driving: ", dict_driving)
        cur_cap = capacity
        for pos in positions:
            #print ("pos: ", pos)

            for info in  dict_driving[pos]:
                num_ppl, indicator = info
                if indicator == 1:
                    cur_cap -= num_ppl
                else:
                    cur_cap += num_ppl
                if cur_cap < 0:
                    return False
        return True
{% endhighlight %}
