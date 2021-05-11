* In CS, big O notation is used to describe the scaling of an algorithm or data structure with respect to time or space requirements and the number of elements.

* Some common scalings, in order of least complex to most complex (generally): O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!)

* An O(1) procedure is not necessarily cheap, it just does not scale with the number of elements. 

* Properties:
    - The order O() of a function which is a sum of other subfunctions is determined by the order of the fastest growing subfunction
    - Powers of n within a logarithm do not affect the order, but shift it by a constant factor
    - Logs with different bases have the same order, but exponentials with different bases are not the same order.
    - Product: the order of two functions multiplied together is the result of multiplying the two orders together: f = O(i), g = O(j) -> fg = O(ij) 
    - Sum: The largest order function is the order of the sum of the functions

* The most common application is in informing the selection of an appropriate data structure:
    - Arrays have O(1) access but O(n) insert/delete
    - Linked lists have O(n) access but O(1) insert/delete
    - Arrays, stacks, queues, linked lists all have O(n) search, whereas hash tables have O(1) search
    - Most trees, such as binary search trees, have O(log n) access, search, insert, and delete  

* There are a million sorting algorithms with diverse complexities. Memorizing this would be silly, just go to [bigocheatsheet](https://www.bigocheatsheet.com/)

