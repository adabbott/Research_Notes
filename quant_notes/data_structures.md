# Data Structures Notes

### Hash Tables
Hash tables use hash functions to quickly insert and search through data. Two types:
    * Hash set: a set (no repeated values) with a defined hash function
    * Hash map: key/value pairs (like a python dict)

Hash Table maps 'keys' to 'buckets' using a hash function.  
In the ideal case, the hash function is a one-to-one mapping.
Usually, however, multiple keys may be mapped to the same bucket, known as a _collsion_.
Collision resolution algorithms try to handle these cases, namely, how to organize
values that are in the same bucket, deal with too many values assigned to the same bucket, 
and how to search for a specific value in a certain bucket.


### Linked List


