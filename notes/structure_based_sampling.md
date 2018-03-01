[This paper](http://aip.scitation.org/doi/abs/10.1063/1.4989536) used a similar idea to that planned for the code,
where one does not just randomly take points from the dataset for the training set, but instead evenly use points on a grid over the configuration space.

That is, nuclear configurations for the trainig set are selected based on the relative distance of the molecule descriptors
based on the euclidean distance (L2 norm), and these are evenly dispersed for the desired training set size.

This naturally shows massive improvement over random sampling, which is common in ML. Since PESs are continuous, and for the most part slowly changing landscapes,
this approach makes far more sense.

The vibrational energy levels of the full 44819 point surface vs surfaces learned through machine learned subsets (50%, 25%, 10%) generally have less than 1 wavenumber differences (!)
