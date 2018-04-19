# Week of April 16th

* Began writing phase II proposal

* Found really nice paper comparing NN vs GP: [Neural networks vs gaussian process regression for representing potential energy surfaces: A comparative study of fit quality and vibrational spectrum accuracy.](https://aip.scitation.org/doi/full/10.1063/1.5003074)

* Learned how to make a very basic ML model in Keras. Combined with data parsing routine to develop a simple, 'hello world' NN model for O<sub>2</sub> stretching.

[comment]: <> (* Update: Improved NN model for O<sub>2</sub> stretching. Wrote energy only and energy + gradient models. energy + gradient model performs slightly better all else equal. Divided dataset 50% test, 25% train, 25% validation, mean absolute error in the test set is on the order of 0.5-1.5%. Cross validation still not attempted. 
)


* Meeting Notes:
    * Get a project name
    * Don't automate Keras models yet, first create a clean way to generate data, convert it to Keras-friendly form, and a way to use/access Keras models afterwards 
    * Design a workflow so that data only needs to be obtained once, and save it to a file
    * During ML model development, take note of trends in the hyperparameters, such as NN structure and number of nodes, that typically perform well, so that when incorporating HyperOpt for hyperparameter search, the search is more limited and refined.
    * Look into Keras/Tensorflow diagnostic plots of model performance  
    * Move driver into the code, allow software to be called from anywhere 
    * Create simple documentation on how to run software for data generation
    * Look into MolSSI quantum chemistry database project, add support in Phase II plan 
    * Clean up repo, add docstrings 
    * Highlight Psi4Numpy publication in PII proposal, record anticipated publications in Phase II plan
       
