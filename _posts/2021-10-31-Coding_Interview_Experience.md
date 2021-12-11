---
layout: post
title:  "What to do when you're stuck for a coding problem"
date:   2021-10-31 18:50:00 +0800
categories: default
tags: Coding, Interview
---

Sometimes if we have encountered a problem that is not obvious to the solution, what to do at this time?

According to my personal interview experience, we can not never stop thinking or thinking without talking for a long time. That is a big red flag.

#### My personal suggestion:

when I am stuck in coming up with a solution immediately, I will use a test case to inference.
It could be done with these steps:

1) analyze a small test case with data size 1, data size 2 and try to come up with the solution

2) analyze data size 3 and more,  try to find a pattern or solution to that

3) then find the final solution for this problem according to previous two steps' inferences.


I found this is quite helpful, and it is also clear to the interviewer.


#### Example:
Let's give an example:

For the leetcode problem 1094. Car Pooling.

There is a car with capacity empty seats. The vehicle only drives east (i.e., it cannot turn around and drive west).

You are given the integer capacity and an array trips where trip[i] = [numPassengersi, fromi, toi] indicates that the ith trip has numPassengersi passengers and the locations to pick them up and drop them off are fromi and toi respectively. The locations are given as the number of kilometers due east from the car's initial location.

Return true if it is possible to pick up and drop off all passengers for all the given trips, or false otherwise.

