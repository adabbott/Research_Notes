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
An ordered collection of values, where
* data is **not** stored contiguously in memory
* each unit contains an element and a pointer to the next element
    - doubly-linked list contains pointers to next and previous element
* Linked lists are great (O(1) scaling) for inserting and removing elements anywhere
* element index lookup is slow (O(n)) since you have to jump from element to element to find it
* decent memory efficiency, but not great.
* can have different types in the same structure, as opposed to say, arrays.

### Arrays
Continous storage in memory of constant-type. Can be:
* Dynamic : fixed size, with a buffer region giving some wiggle room to grow. If the array grows beyond the preallocated
buffer space, have to reallocate new memory. Different implementations have different specifications for how this is done, how much space to reserve
* Static : fixed size array. 


### Stacks
Linear data structure where only _one end_ is modifiable, and you can only add/remove to the end or beginning depending on the type of stack.

### Queue
Linear data structure with "first in, first out" property for removing and adding. 
Whereas stacks add/remove on one end of the data structure, queues always remove the oldest element,
and newest added elements are adjacent and ordered according to how recently they were added.

### Binary Tree
Collection of nodes in a tree structure, where every node has two children or less.
The top node is called the _root_, terminal nodes _leaves_, and nodes coming off a particular _parent_ node are denoted as _child_ nodes.

### Binary search tree
This is a binary tree with a sorted structure. 
* The left child of the parent node has keys **less than** the parent node
* The right child of the parent node has keys **greater than** the parent node
* The above two rules are recrusively true for all subtrees in the binary search tree.
    - That is, every child must also be a root of binary search tree that is a subtree of the whole
* Search, min, max, can be done very efficiently by exploiting the sorted structure

Why use a tree?
* When data naturally forms a hierarchy (like file structures), trees make sense.
* Trees also give a decent middle ground for many operations: 
    - access and search (faster than linked lists, but slower than arrays)
    - insertion/deletion (worse than linked lists, but faster than arrays)
Consequently, if your use case needs to do a lot of access/search/insertion/deletion all at once, and roughly the same number of times, trees are likely the way to go.

* Trees make information easy to search with various _tree traversal algorithms_
    - Breadth-first-search: explore nodes in layers/levels
    - Depth-first-search: explore nodes level by level until you hit the leaves

### Binary heaps
A binary tree, except all levels are completely filled, with the except of the last level (the leaves), which is left-biased (all keys as left as possible)
      O
     / \
    /   \
   O     O
  / \   / 
 O   O O   

A binary heap always has a property (invariant) such as: 
    * Min-Heap: value of any node must be smaller than its children
    * Max-Heap: value of any node must be greater than its children
Operations need to retian the invariant, and maintain the left-biased structure
When inserting a new node, operations must be performed to continue to satisfy the invariant 
When deleting a node, a restructuring needs to occur to restore the shape (left-biased)
    * for example, take a leaf and move it to the hole where the element was deleted
then, we need to appropriately swap elements to satisfy the invariant.

### Other trees
Other tree types (B-Tree, Cartesian Tree, Red-Black Tree, Splay Tree...)
offer various changes to the standard binary search tree, most of which have 
O(log n) access, search, insertion and deletion, but have small performance differences between them in certain cases.
More notably, a B-tree generalizes the binary search tree to allow for more than two children for nodes, making it more generally applicable to problems without significant
engineering effort.



