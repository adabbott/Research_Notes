# NumPy Cheat Sheet
(I've used numpy a lot, so only noting things that seem worth reminding of. ) 

* Several ndarrays can share the same data, so that an ndarray can be a **view** to another ndarray.  
ndarrays can also view into memory owned by Python strings or other objects if they implement the buffer and array interfaces.

* Attributes:
    - __all__: If all elements are True, return True
    - __any__: If a single element is True, return True
    - __argmax__, __argmin__, __argsort__ : returns _indices_ which are maximum, minimum, or would sort the array
    - __clip__: Return an array with values limited by a min and/or max
    - __cumsum__, __cumprod__: returns cumultive result of addition or multiplication up to that index, for every value along an axis
    - __sum__, __prod__: straight-forward sum or product along an axis
    - __ptp__ : peak to peak (max - min) along an axis 
    - __sort__: sort an array in-place
    - __std__: return standard deviation 
    - __swapaxes__: returns a view of the array with axes interchanged
    - __squeeze__: remove axes of length 1

* Arithemetic and comparison operations
    * arithemetic `+  -  *  /  //  %  divmod()  ** pow,  <<  >>  &  ^  |  ~
    * Comparisons == < > <= >= != 
    * all of the above characters are equivalent to calling universal functions in NumPy.


* Slicing, indexing:
    * start, stop step --> A[i:j:k]
    * Negative i and j == n+i, n+j , where n is the dim size
    * Negative k means _step from the back_
    * Reverse an array A[::-1] just like python lists/strings
    * Ellipsis: expands number of : objects needed to select all dimensions. Only one ellipsis
    can be used in a slice. Can be read as "every axis to the left" or "everything to the right" of the specified axes
    A[...,0] == A[:,:,0] if A has 3 dimensions


    * "Fancy" (advanced) indexing: A selection object is instead a sequence-like object 
        - Careful: returns a copy!
        - Contrast with basic slicing is always a view of the original array
        - Two types for advanced indexing: integer or bool
        - With boolean masks, if mask is smaller than array that is being masked, equivalent to matching shapes with False in unspecified indices
    
    * Broadcasting
        - All input arrays to a function with a number of dimensions smaller than the largest dimension input array, have 1's prepended to their shapes
        - The size in each dimension of the output shape is the max of all input sizes in that dimension
        - An input can be used in the calculation if its size in a partiuclar dimension either matches the output size of has value of 1
        - If a dimension size is 1, the first data entry in that dimension will be used for all calculations along that dimension.
    * Another way to thinkabout it, for broadcasting, one of the following statements is true:
        1. **All arrays have the same shape**
        2. **The arrays have the sume number of dimensions and length of each dim is either the same or 1**
        3. **The arrays that have too few dimensions can have their shapes prepended witha dimension of length 1 to satisfy 2 above **
    
    * Ufuncs: functions that operate element by element on whole arrays. Methods:
        - accumulate: accumulate result of applying the op to all elements
        - at: performs inplace operation at indices 
        - outer: apply op to all pairs within A and B
        - reduce: reduce an arrays dimension by one by applying the ufunc along one axis
        - reduceat: performs a local reduce with specified slices over a single axis
    * About 60 ufuncs, most are called internally with basic operations, standard API. One may
    want to call a ufunc directly to place outputs in preallocated objects
    
    * Array creation routines:
        - fromfunction: construct an array by executing a defined function over each coordinate (neat)

    


        
        
    
    




