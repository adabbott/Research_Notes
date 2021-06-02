# Pandas Notes 

* Series are one-dimensional arrays, with a label,
that can hold any data type. The axis labels 
are the **index**. Default index labels are just integer indices,
but can be customized with, say, dicitonary data or passing a list of labels.

* Series can be initialized with data that is a dictionary, ndarray,
etc

* Slicing on a Series works as expected, and slices according to values 
but also slices the index (duh)

* Can get an element according to label with the get method
    `series.get("a")` and will return None if missing (or another default value of choice)

* Series operations are vectorized just like in nmupy, and can be passed
to many numpy methods.

* Big difference between Series vs ndarray: operations between more than
one series will automatically align the data based on the labels (index).
This is unexpected compared to NumPy; an operation between two unaligned Series
will have the union of the indices involved. 

* DataFrames: 2-d labeled with columns that can have different types.
Can be initialized with:
    - Dictionary of 1d ndarrays, or lists, dicts, or Series
    - 2-D numpy.ndarray
    - Structured or record ndarray
    - A Series
    - Another DataFrame

* Can pass index (row labels) and columns (column labels). Can use this to filter
out data as you create the Dataframe, e.g. can speicfy only a certain set of row labels
or column labels:  DataFrame(otherdf, index=["specific","indices"], columns=['specific', 'columns'])


# Column selection, addition deletion

A dataframe is just a **dictionary** of like-index Series objects, so manipulating columns is similar to manipulating a dictionary.

```python
# create new column from previous cols
df["c"] = df["a"] * d["b"]
# create bool
df["a > b"] = df["a"] > df["b"]
# delete, pop
del df["name"]
e = df.pop("e")
# Creating a new column with scalar values will automatically fill the column with the value
df["zero_col"] = 0
```


# Indexing, selecting data
While standard Numpy and Python slicing schemes can work, the best performance can be obtained
using pandas 



# Key operations

* DataFrame.describe() gives descriptive statistics of each column

* drop_duplicates() gives a new DF with duplicate rows removed. Can be inplace.

* groupby() groups a dataframe by 1 or more columns,

