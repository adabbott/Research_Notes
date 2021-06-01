***Descriptive statistics*** presents, organizes, and summarizes data, a collection of values/diagnostics which describe the characteristics of a sample or population
***Inferential statistics*** the task of drawing conclusions about a population based on observed data in a sample, infer the results of a sample 

### Descriptive Statistics
Set of metrics for summarizing data (brainless stuff) 
* Central tendency: mean, median, mode
* Measures of variability: range, standard deviation (average variation from the mean), variance (average of square deviation from the mean)
* Skew: symmetry of the distribution of variables
* Kurtosis: peak-ness or flattness of a distribution
* Shape: modality (single mode, bi modal, multi modal)

### Inferential Statistics
* A sample set of data is used to make an inference of a population 
* Use probability to estimate how confident our conclusions are (confidence intervals, margins of error)

* Variance (_s_) of a sample in denominated by _n-1_. Variance of a population (_σ_) is denominated by _n_.

* The **Standard Error** is an estimate of the standard deviation of the **sampling distribution**, where the 
**sampling distribution** is defined as the set of all samples of a particular size that can be taken from a population.
The standard error shows you how good your data is. The standard error for the mean is the standard deviation over the square root of the number of samples.  
$$ \frac{s}{\sqrt{n}} $$  

[Notes from this resource](https://web.mit.edu/~csvoss/Public/usabo/stats_handout.pdf)

# Miscellaneous Topics
* Z-scores: how many standard deviations σ away from the mean μ

* Central limit theorem: given a population, if you take sufficiently large samples with replacement, the distribution of the sample means will be approximately Gaussian distributed.

* T-test (ONE SAMPLE): The purpose of a T-test is to test whether the mean of a normally distributed population is different from a specified value.
The Null Hypothesis states that the population mean is some particular value, and the alternative hypothesis states that the mean is not equal or > or < that value.
The t statistic is a standardized difference between the population mean and sample mean:
$$ t = \frac{\bar{x} - \mu_0}{ s } $$
    - **p-value**: the probability the sample mean was obtained by chance given the population mean.
    - If the p-value is less than the predetermined value for significance, (α, typically 0.05), the null hypothesis is rejected.

    - If the alternate hypothesis is that the mean is != some value, double the absolute value of t-statistic and read off p-value.
    - If the alternate hypothesis is that the mean is > some value, and the t-statistic is positive, read p-value as given
    - If the alternate hypothesis is that the mean is < some value, and the t-statistic is negative, read off absolute value p-value.
    - If the t-statistic is not the expected sign, the p-value is 1-p 

* T-test (TWO SAMPLE)
    
# Linear Regression (TODO)

* Assumptions

* Univariate, multivariate

* MAP, MLE, prior distribution Gaussian yields L2 regularization, prior Laplacian yields L1 regularization.





