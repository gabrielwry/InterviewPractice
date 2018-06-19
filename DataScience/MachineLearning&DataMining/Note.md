# Machine Learning  

## Data:

1. Data is a collection of data objects and their attributes.
2. Categorical Attributes vs. Numeric Attributes:
   1. Categorical is qualitative: 
      1. Nominal (Id, eye color, zip code) vs. Ordinal (rankings, grades, height) 
   2. Numeric is quantitative :
      1. Interval (calendar date, temperatures in Celsius or Fahrenheit) vs. Ratio (temperature in Kelvin, length, time)
3. Distinctness: $= \neq$ Order:$ < >$ Addition: $+ -$ Multiplication: $* /$
   - Nominal attribute: distinctness

- Ordinal attribute: distinctness & order
- Interval attribute: distinctness, order & addition
- Ratio attribute: all 4 properties

1. | Attribute Type | Description                                                  | Examples                                                     | Operations                                                   |
   | :------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
   | Nominal        | The values of a nominal attribute are just different names, i.e., nominal attributes provide only enough information to distinguish one object from another. ($=\neq$) | zip codes, employee ID numbers, eye color, sex: {male, female} | mode, entropy, contingency correlation, $\mathcal{X} ^2$ test |
   | Ordinal        | The values of an ordinal attribute provide enough information to order objects. (<, >) | hardness of minerals, {good, better, best}, grades, street numbers | median, percentiles, rank correlation, run tests, sign tests |
   | Interval       | For interval attributes, the differences between values are meaningful, i.e., a unit of measurement exists. (+, -) | calendar dates, temperature in Celsius or Fahrenheit         | mean, standard deviation, Pearson's correlation, $t$ and $F$ tests |
   | Ratio          | For ratio variables, both differences and ratios are meaningful. (*, /) | temperature in Kelvin, monetary quantities, counts, age, mass, length, electrical current | geometric mean, harmonic mean, percent variation             |

2. Discrete vs. Continuous:

   1. Discrete: Has only a finite or countable infinite set of values, always represented by integer number
   2. Continuous: Real number as attribute value, always represented by floating point number
   3. Nominal and ordinal are always discrete, interval and ratio are always continuous. 

3. Summary Statistics of Data: central tendency vs. variability or dispersion 

4. Computational: 

   1. Distributed measure –can be computed by partitioning the data into smaller subsets.
   2. Algebraic measure –can be computed by applying an algebraic function to one or more distributed measures.
   3. Holistic measure –must be computed on the entire dataset as a whole.

5. Central Tendency: Mean, median mode. How data seem similar, location of data

   1. Skewed vs. Symmetric (mode = median =mean)

6. Dispersion: The degree to which numerical data tend to spread

   1. Range: difference between the largest and smallest values
   2. Percentile: the value of a variable below which a certain percent of data fall
   3. Quartiles: Q1 (25th percentile), Median (50th percentile), Q3 (75th percentile)
   4. Inter-quartile range: IQR = Q3 – Q1
   5. Five number summary: min, Q1, M, Q3, max (Boxplot)
   6. Outlier: usually, a value at least 1.5 x IQR higher/lower than Q3/Q1
   7. Variance and standard deviation

7. Graphic Display of Basic Statistical Description:

   1. Boxplot![1522090608674](C:\Users\gabri\AppData\Local\Temp\1522090608674.png)
   2. Histogram: equal-width vs. equal-depth; A set of rectangles that reflect the counts or frequencies of values at the bucket (bar chart)
   3. Scatter Plot: Displays values for two numerical attributes (bivariate data); can suggest correlations between variables with a certain confidence level: positive (rising), negative (falling), or null (uncorrelated).

1. Data pre-processing:
   1. Quality-Issue: Incomplete (lacking interesting attribute); noisy (errors or outliers); inconsistent (discrepancies); duplicate (containing duplicate record)
   2. Handle Missing Values: delete tuple, or fill in with mean (attribute mean or mean from the same class)
   3. Handel Noisy Data: Noise - random error or variance in a measured variable
      1. Binning and smoothing, sort and partition into bins (equi-width or equi-depth), then smooth by bin mean, bin median, bin boundaries
         1. Equi-width : Divides the range into N intervals of equal size, may dominate by outliers, doesn't handle skewed data very well
         2. Equi-depth: Divides the range into N intervals, each containing approximately same number of samples; tricky when handling categorical attributes
      2. Regression: smooth by fitting the data in a regression function
      3. Cluster: detect and remove outliers fall outside clusters
      4. Manually check
   4. Normalization: scaled to fall within a small, specified range
      1. Min-max normalization: [minA, maxA] to [new_minA, new_maxA]
      2. Z-score normalization (μ: mean, σ: standard deviation) $v' ={ {v-\mu}\over\sigma_A}$
      3. Normalization by decimal scaling
   5. Data Integration: combines data from multiple sources into a unified view; 
   6. Data warehouse: tightly coupled
      1. advantage: high query performance, can operate when source is not available, extra aggregated information from warehouse, not affected by the local processing
      2. disadvantage: data freshness, hard to build when only have access to query interface of local source
   7. Federated Database Systems: use a mediator and a wrapper to access different source
      1. Advantage: No need to copy and store data at mediator, More up-to-date data, Only query interface needed at sources
      2. Disadvantage: Query performance, Source availability
   8. Database Heterogeneity: 
      1. System Heterogeneity: use of different operating system, hardware platforms
      2. Schematic or Structural Heterogeneity: the native model or structure to store data differ in data sources.
      3. Syntactic Heterogeneity: differences in representation format of data
      4. Semantic Heterogeneity: differences in interpretation of the 'meaning' of data
   9. Data Matching: 
      1. Semantic Integration: reconciling semantic heterogeneity; schema-matching (A.cusit_id = B.cust-#), data matching (Bill Clinton = William Clinton)
      2. Schema Matching: rule based vs. learning based; 1-1 matches vs. complex matches
         1. Probabilistic record linkage: Similarity between pairs of attributes, combined scores representing probability of matching, threshold based decision. 
   10. Data Transformation: 
       1. Aggregation: sum/count/average E.g. Daily sales -> monthly sales
       2. Discretization and generalization E.g. age -> youth, middle-aged, senior
          1. Discretization: transform continuous attribute into discrete counterparts (intervals) Supervised vs. unsupervised Split (top-down) vs. merge (bottom-up)
          2. Generalization: generalize/replace categorical attributes from low level concepts (such as age ranges) by higher level concepts (such as young, middle-aged, or senior)
       3. (Statistical) Normalization: scaled to fall within a small, specified range E.g. income vs. age Not to be confused with database normalization and text normalization: 
       4. Attribute construction: construct new attributes from given ones E.g. birthday -> age
   11. Data Reduction
       1. Instance reduction
          1. Sampling (instance selection): obtaining a small representative samples to represent the whole data set N A sample is representative if it has approximately the same property (of interest) as the original set of data
             1. Sampling Method: 
                1. Simple Random Sampling: There is an equal probability of selecting any particular item
                2. Sampling without replacement; As each item is selected, it is removed from the population
                3. Sampling with replacement: Objects are not removed from the population as they are selected for the sample -the same object can be picked up more than once
                4. Stratified sampling Split the data into several partitions (stratum); then draw random samples from each partition
                5. Cluster sampling When "natural" groupings are evident in a statistical population; sample a small number of clusters
          2. Numerocity reduction:Reduce data volume by choosing alternative, smaller forms of data representation
             1. Parametric methods Assume the data fits some model, estimate model parameters, store only the parameters, and discard the data (except possible outliers); Regression
                1. Linear Regression: Line fitting or polynomial fitting; least square fitting
             2. Non-parametric: do not assume models but use cluster and histograms
                1. Histogram: divide data into buckets and store average(sum) for each bucket; Equi-width, equi-depth, and v-optimal
       2. Dimension reduction
          1. Feature selection
             1. Select a subset of features by removing irrelevant, redundant, or correlated features such that mining result is not affected; irrelevant features/ redundant or correlated features; 
                1. Correlation measures the linear relationship between variables, does not necessarily imply causality. 
                2. Correlation Analysis: 
                   1. Numerical Data - correlation coefficient: Pearson's product moment coefficient. $r_{A,B} = {\sum{(A-\bar{A})(B-\bar{B})} \over {(n-1)\sigma_A \sigma_B}}={\sum{(AB)-(n-\bar{A}\bar{B})} \over {(n-1)\sigma_A \sigma_B}}$ where n is the sum of tuples, $\sum{AB}$ is the sum of dot product of $AB$ ; $r_{AB} = 0$ means the two are independent. 
                   2. Categorical Data: chi-square- tests the hypothesis that A and B are independent $\mathcal{X}^2 = \sum{(Observed - Expected)^2 \over {Expected}}$ , the larger the $\mathcal{X}^2$ The more likely these two attributes are correlated. 
             2. Filter approaches: selected independent of data mining algorithm. 
             3. Wrapper approaches: use data mining algorithm as black-box to find best sub-set
             4. Embedded approaches: as part of data mining
          2. Feature extraction: Create new features (attributes) by combining/mapping existing ones Common methods: 
             1. Principle Component Analysis: find an orthogonal transformation with a set of principal components, such that the first principal component has the largest variance: Mathematically Compute the covariance matrix; Find the eigenvectors of the covariance matrix correspond to large eigenvalues
             2. Singular Value Decomposition
             3. Other compression methods (time-frequency analysis)
                1. Fourier transform (e.g. time series), decompose a signal into a finite number of sine waves, where each wave represented by a Fourier coefficient
                2. Discrete Wavelet Transform (e.g. 2D images), represent data in terms of average and difference in wavelet, store only a small fraction of the strongest wavelets. 

## Frequent Item Mining:

1. Frequent pattern: a pattern (a set of items, subsequences, substructures, etc.) that occurs frequently in a data set
   1. Itemset: $X = {x_1, …, x_k}$ (k-itemset) 
   2. Frequent itemset: X with minimum support count 
      1. Support count (absolute support): count of transactions containing X
   3. Association rule: $A\to B$ with minimum support and confidence
      1. Support: probability that a transaction contains $A \cup B$ 
      2. Confidence: conditional probability that a transaction having A also contains B, $c=P(B|A)$
      3. Association rule mining process Find all frequent patterns (more costly) Generate strong association rules
   4. Methods: 
      1. Apriori: Any nonempty subset of a frequent itemset must be frequent
         1. Apriori pruning principle: If there is any itemset which is infrequent, its superset should not be generated/tested!
         2. Level-wise search method: 1)Initially, scan DB once to get frequent 1-itemset 2) Generate length (k+1) candidate itemsets from length k frequent itemsets 3)Test the candidates against DB 4)Terminate when no frequent or candidate set can be generated
         3. Implementation Detail: 
            1. Candidate sets generation: self-joining $L_{k-1}$ , assuming items and itemsets are sorted in order, joinable only if the first $k-2$ items are in common; Pruning infrequent subset
            2. Count support of Candidates: for each transaction $t$ in the database, find all candidates in $C_k$ that are subset of $t$ and increment the count (need to check each subset s in t), this can be huge. Better use a dynamic programming approach, prefix-tree, hash-tree, or hash-table.
               1. DHP (Direct Hashing and Pruning): hash k-itemset into buckets and a k-itemset whose bucket count is below the threshold can not be frequent; This is useful for 2-itemset, the idea is to filter out use a more broader bucket such that the infrequent itemset are filtered out using a hash function.
         4. Improvement on Apriori: Partitioning to reduce scan, and sampling
      2. Frequent pattern growth: 
         1. Basic idea is to grow long-pattern from shorter pattern recursively: “abc” is a frequent pattern; All transactions having “abc”: DB|abc(conditional DB); “d” is a local frequent item in DB|abc, then abcd is a frequent pattern
         2. FPG vs. Apriori: DFS vs. BFS; always faster because it decompose both mining task and DB, and using a pre-fix tree to recursively find shorter path and concatenate with suffix
2. Closed and Maximal patterns:
   1. Closed: An itemset X is closed if $f$ is frequent and there exists no super-pattern with the same support as X.
   2. Max Pattern: If X is frequent, and there exists no super pattern Y that is frequent. 

## Regression:

Regression problem produced a real-valued output; use training set and learning algorithms to produce a hypothesis function $h$ to take the feature vector $x$ and predict the output $y$

1. Linear Regression:  $ h_\theta(x) = \Theta_0+\Theta_1(x) $ , univariable linear regression.

   1. Cost Function: Choose $\theta_0, \theta_1$ so that $h_\theta(x)$ is close to $y$ for our training set. $J_{(\theta_0,\theta_1)} = {1\over {2m}} {\sum_{i=1}^{m}(h_\theta(x^i)-y^i)^2}$  

2. Multivariable Linear Regression: $h_\theta(x) = \theta X^T$ 

3. Polynomial Regression: 

   1. For example, if our hypothesis function is $h_\theta(x) = \theta_0 + \theta_1 x_1$ then we can create additional features based on x_1x1, to get the quadratic function $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2hθ(x)=θ0+θ1x1+θ2x12$ or the cubic function $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3 $ 
   2. note: feature scaling becomes very important if we create new features

4. Gradient Descent: Parameter Learning -> Gradient Descending, to minimize $J_{(\theta_0, \theta_1)}$:

   1. Linear Regression:

      1. Start with some $\theta_0, \theta_1$,keep changing $\theta_0, \theta_1$ and to reduce cost function, and hopefully end up at least local minimum.

      2. repeat until convergence{

         ​    $\theta_j :=\theta_j - \alpha {\delta \over{\delta \theta_j}}J(\theta_0,\theta_1) \text{   (simultaneously for j=0, and j=1) }$  

         } note: compute tmp $\theta_0$ and $\theta_1$ first, then assign them

         as this approach a local minimum, gradient descent will automatically take smaller steps, until it hits local minimum. 

      3. Decompose the Partial-Derivative:

         1. $j=0: {\delta\over\delta\theta_0}J(\theta_0,\theta_1) = {1\over m}\sum^m_{i=1}(h_\theta(x^i)-y^i) $
         2. $j=1: {\delta\over\delta\theta_1}J(\theta_0,\theta_1) = {1\over m}\sum^m_{i=1}(h_\theta(x^i)-y^i) x^i $

      4. Batch Gradient Descend: each step uses all training examples

   2. Multivariable Linear Regression:

      1. repeat until convergence{

         $θ_j:=θ_j−\alpha{1\over m}∑_{i=1}^m=(h_θ(x^i−y^i)⋅x^{(i)}_ j\text{for j := 0...n}  $

         }note: update all features simultaneously

   3. Feature Scaling: Make sure features are on a similar scale, to make the gradient descent converges much faster.

      1. Mean normalization: replace $x_i$ with $x_i - \mu_i$  
      2. etc.

   4. Learning Rate: 

      1. Debugging Gradient Descent: plot no. of iterations against the min of cost function, usually increasing $J(\theta)$ caused by some too large $\alpha$; but a small $\alpha$ may cause it too slow to converge

5. Normal Equation: Solve for $\theta$ analytically , $θ=(X^TX)^{−1}X^Ty$ , can be proved using projection (minimize the error vector ) 

   1. Design matrix: for $m$ examples and $n$ features, make a $m \times (n+1)$ matrix with $x_0 = 1$ 
   2. Vector $y$ is the $m \times 1$ vector 
   3. Normal Equation vs. Gradient Descent:
      1. Gradient Descent: need learning rate, need many iterations, but works well with large $n$
      2. Normal Equation: time complexity $O(N^3)$ , not efficient large $n$ , doesn't work for more complicated algorithms
   4. Non-invertibility: $X^TX$ is non-invertible always caused by too many features (m<n)

## Classification: 

1. Logistic Regression:$h(\theta) = g(\theta ^ Tx)$ where $g(z) = {1\over{1+e^{-z}}}$  , the output  is an estimated probability that $y=1$ on input $x$ 

   1. Decision Boundary: a property of the hypothesis (including the parameter sets) , that separates the output of classification. 

      1. Non-linear Boundary: add higher order parameter 

   2. Cost Function: 

      1. Square-error will not apply, as the sigmoid function caused the cost-function to be an non-convex one (gradient descent will not find the local minimum)

      2. $J(θ)={1\over m}∑_{i=1}^mCost(hθ(x(i)),y(i))$

          $Cost(hθ(x),y)=−log(hθ(x)) \text{           if y = 1}$

         $Cost(hθ(x),y)=−log(1−hθ(x))\text{          if y = 0}  $

      3. The idea is to penalize the algorithm, the higher the wrong probability, the higher the penalty 

      4. Simplified: $Cost(hθ(x),y)=−ylog(1−hθ(x))-(1−y)log(1−hθ(x))$ 

   3. Gradient Descent:

       repeat until convergence{

      $θ_j:=θ_j−\alpha{1\over m}∑_{i=1}^m=(h_θ(x^i−y^i)⋅x^{(i)}_ j\text{for j := 0...n}  $

      }note: update all features simultaneously

   4. Providing ways to compute cost function and the partial derivative of it, there are other optimization algorithms that are available: 

      - Gradient Descent vs. (Conjugate Descent, BFGS, L-BFGS) with a clever inner-loop that picks a different picking learning rate 	
      - No need to pick a learning rate, so converge faster, but are more complex

2. Multi-class Classification: 

   1. One-vs-all: find the max probability from different classifiers (logistic regression or SVM)

   2. Decision Tree

      1. Greedy-algorithm: top-down manner, divide-conquer, split attributes based on heuristic or statistical measure (information gain) Attribute Selection Measures: Information Gain, Gain Ratio, Gini index
         1. Information Gain:
            1. Entropy: A measure of uncertainty associated with a random variable; $H(Y) = -\sum ^m _{i=1} {p_i log(p_i)}$  where $p_i = P(Y=y_i)$ , higher entropy means more uncertainty
            2. Conditional Entropy: $H(Y|X) = \sum_x{p(x)H(Y|X=x)}$ 
            3. Select the attribute with the highest information gain: Let pi be the probability that an arbitrary tuple in D belongs to class Ci, estimated by $|C_i, D|/|D|$,
               1. Expected Information: entropy needed to classify a tuple in D:  $Info(D) = -\sum ^m _{i=1} {p_i log_2(p_i)}$ 
               2. Information needed after using A to split D into v partition to classify D: $Info_A(D) = \sum ^v_{j=1} {D_j \over D} \times Info(D_j)$
               3. $Gain(A) = Info(D) - Info_A(D)$ 
         2. Gain Ratio: Information gain is biased towards attribute with a larger number of values
            1. $\text{SplitInfo}_A(D) = -\sum ^v _{j=1}{{|D_j| \over |D|}\times log_2({{|D_j|}\over |D|}})$
            2. GainRatio(A) = Gain(A)/SplitInfo(A)
         3. Gini Index: largest reduction in impurity is chosen to split the node

   3. Bayes Classifier: A statistical classifier, performs probabilistic prediction

      1. Bayes theorem: $P(H|X) ={ {P(X|H)P(H)} \over{P(X)}}$

         1. Informally, this can be viewed as: posteriori = likelihood x prior/evidence
            1. P(H) (prior probability): the initial probability
            2. P(X): probability that sample data is observed
            3. P(X|H) (likelihood): the probability of observing the sample X, given that the hypothesis holds
         2. Classification is to maximize the posteriori, which is practical difficult as it requires initial knowledge of many probabilities,

      2. Naïve Bayes: assuming all attributes are conditionally independent (the product of individual conditional probabilities)

         1. Avoiding zero-probability: laplacian correction (adding 1 to each case)
         2. Advantage: easy to implement, good result; disadvantage: assumption loss of accuracy, dependencies exist among variables

      3.  Bayesian belief network (probabilistic networks): allows conditional independencies between subsets of variables

         1. Two components: 

            (1) A directed acyclic graph (called a structure) : Represents dependency among the variables Each node has a conditional probability distribution P( X | parents(X) )

            (2) a set of conditional probability tables (CPTs)

         2. Using chain rule: $P(X_1, ..., X_n) = \prod ^n _i{P(X_i|Parents(X))}$

   4. KNN (lazy learners)

      1. Lazy vs. Eager: Lazy learning(e.g., instance-based learning): Simply stores training data (or only minor processing) and waits until it is given a test tuple; Eager learning(the above discussed methods): Given a set of training tuples, constructs a classification model before receiving new (e.g., test) data to classify
      2. KNN: return the average of k-nearest neighbors, distance-weight assigning closer neighbor a higher weight, robust to noisy data by averaging k-nearest 

## Clustering

Finding groups of objects (clusters) Objects similar to one another in the same group Objects different from the objects in other groups;The global optimal is NP hard

1. Heuristic methods: k-means and k-medoids algorithms:

   1. K-mean :Each cluster is represented by the center of the cluster: this is easy to implement but depending on initial centroids, may terminate at a local optimum, and is sensitive to noisy data and outliers

      ​	i. Given k , and randomly choose k initial cluster centers 

      ​	ii. Partition objects into k nonempty subsets by assigning
      	iii. each object to the cluster with the nearest centroid

      ​	iv. Update centroid, i.e. mean point of the cluster

      ​	v. Go back to Step 2, stop when no more new assignment

   2. k-medoids : Each cluster is represented by one of the object in the cluster, Instead of taking the mean value of the object in a cluster as a reference point, medoids can be used, which is the most centrally located object in a cluster

      ​	i. Arbitrarily select k objects as medoid

      ​	ii. Assign each data object in the given data set to most similar medoid.

      ​	iii. Randomly select non-medoid object O’

      ​	iv. Compute total cost, S, of swapping a medoid object to O’ (cost as total sum of absolute  		error)

      ​	v. If S<0, then swap initial medoid with the new one

      ​	vi. Repeat until there is no change in the medoid.

2. A good clustering will produce high quality clusters with Homogeneity - high intra-class similarity Separation - low inter-class similarity

   1. Similarity Measurement: Euclidean / Manhattan / Minkowski distances
   2. Normalize different data type to enable distance calculation
   3. Jaccard distance/similarity of two sets s the size of their intersection divided by the size of their union

3. Evaluation: 


## SVM

Reduce computation complexity 

1. Large Margin Classification:
   1. An Alternative view of logistic regression: 

      The cost function: $-y log{1 \over 1+ e^{-\theta ^T x}}- (1-y)log(1- {1\over {1+e^-\theta^T x}})$, consider this two terms of contribution to error (plug in $y=1$ and $y=0$ )

   2. Rewrite Logistic Regression cost function:

       $Cost(hθ(x),y)=−ylog(1−hθ(x))-(1−y)log(1−hθ(x))$  $\to$ 

      $ min_\theta \sum_{i=1} ^m y^i cost(\theta^T x^i) + (1-y^i) cost(\theta^T x^i) + {\lambda \over 2} \sum _{j=0} ^n \theta_j ^2$ 		

   3. Note: 

      The hypothesis doesn't output a probability, but only the classification;

      The SVM built an additional safety margin factor ;

   4. Proof (Sketch):

      1. Vector Inner Product: $u^Tv= p \bullet{||u||}$ where $p$ is the length of projection of $v$ onto $u$		
      2. the decision boundary: $min_\theta = {1\over 2} \sum_{j=1} ^n \theta_j ^2$  such that the projection of the training example onto the parameter vector  times the length of the parameter vector ( aka. the inner product of the training example and the parameter example) produces the correct result; keep in mind that we want the length of parameter to be small (to avoid overfitting); so what SVM trying to do is to maximize the norm between boundary and training examples.  

2. Kernel

   1. Non-linear Decision Boundary solved by polynomial linear regression, computationally expensive

   2. Define new features a different way: given $x^i$ , compute the new feature depending on proximity to landmarks $l$ 

   3. A kernel is the function to measure similarity, eg. Gaussian Kernel = $exp(-{{||x-l||}^2\over 2\sigma ^2})$ 

      1. if $x \approx l$ the kernel gives $\approx 1$ 
      2. map each sample in the training set $x^i$ to a new vector, with $l^i = x^i$ (so the no. of new features is actually no. of samples and it is a vector of similarities to other samples)
      3. For computational efficiency, rewrite it to $\theta ^T M\theta$ 

   4. Parameters: 

      1. When regularization term (rewrite to $C = {1\over \lambda}$) is large, the SVM becomes sensitive to outliers
      2. When $\sigma ^2$ is large, the bias is higher and the variance is lower, because the features vary smoothly 

   5. Choice of Kernel:

      1. Linear (no-kernel): large no. of features, small no. of samples 

      2. Gaussian Kernel: need to choose $\sigma ^2$ , small no. of feature, large no. of sample.

         Note: need to perform feature scaling if use Gaussian Kernel, otherwise the similarity may be dominated by one of the features. 

      3. Other kernels: need to satisfy "Mercer's Theorem" (basically assure a positive symmetric function converges) ; polynomial kernel, string kernel, chi-square kernel, histogram kernel etc. 

3. Logistic Regression vs. SVM:

   1. when training size is smaller than the size of feature, use Logistic Regression or SVM without kernel 
   2. n is small, m is intermediate (use SVM with Gaussian Kernel) 
   3. n is small, m is too large ( add features and use logistic or SVM without kernel )

## Neural Network

1. Backpropagation
2. RNN
3. Deep Learning

## Recommender System



## Machine Learning System:

1. Debugging Learning Algorithms: What if the hypothesis makes unacceptable large errors ? 

   1. Collect more training examples;  $\to$ fix high variance
   2. Try smaller set of features; $\to$ fix high variance
   3. Try get additional features; $\to$ fix high bias
   4. Try add polynomial features; $\to$ fix high bias
   5. Adjust $\lambda$ (higher $\lambda$ more variance, lower $\lambda$ more bias)

2. Evaluate Hypothesis: 

   1. Cross Validation: Split the dataset to training 60% / cross-validation 20%/ test set 20%

      Help to generalize the error. 

      1. Randomly partition the data into $k$ mutually exclusive subsets, each approximately equal size; At i-th iteration, use Di as test set and others as training set; 
      2. Leave-one-out: k folds where k= # of tuples, for small sized data; 
      3. Stratified cross-validation*: folds are stratified so that class dist. in each fold is approx. the same as that in the initial data

   2. Holdout method, random subsampling

      1. Given data is randomly partitioned into two independent sets Training set (e.g., 2/3) for model construction Test set (e.g., 1/3) for accuracy estimation
      2. Random sampling: a variation of holdout Repeat holdout k times, accuracy = avg. of the accuracies obtained

      

   3. Bootstrap: A resampling technique, works well with small data sets, Samples given data for training tuples uniformly with replacement

   4. Error Analysis: Spot systematic pattern

      Error Metrix:

      |                            | Actual Positive(P) | Actual Negative(N) |
      | -------------------------- | ------------------ | ------------------ |
      | **Predicted Positive(PP)** | true positive(TP)  | false positive(FP) |
      | **Predicted Negative(PN)** | false negative(FN) | true negative(TN)  |

      Recall / Sensitivity = $TP \over P$ 

      Precision = $TP\over TP+FP$ 

      miss = $TN \over P$ 

      F1 = $ 2 \bullet{{2TP \over 2TP+FP+FN}}$ the harmonic mean of precision

   5. Bias vs. Variance: 

      1. plot degree of polynomial against Training Error and CV error. 
         2. Bias underfit, variance overfit. 
         3. With a higher degree of polynomial, it is more likely to have low error on training set but higher error on cross validation error. 

   6. Regularization: Add regularization items to make the parameters smaller

      1. simpler hypothesis , less prone to overfitting 

      2. $min_θ {1\over 2m} ∑_{i=1}^m(hθ(x(i))−y(i))2+λ ∑_{j=1}^nθ_j^2  $ , where $\lambda$ is the regularization parameter. 

      3. Gradient Descent w/ regularization: Repeat{

         $θ_0:=θ_0−α {1\over m} ∑_{i=1}^m(hθ(x^i)−y^i)x^i_0  $

         $ θ_j:=θ_j−α [({1\over m} ∑_{i=1}^m(h_θ(x^i)−y^i)x^i_j)+{λ\over m}θ_j] \text{ j∈{1,2...n}} $

         } the additional term is the partial derivation of the regularization term in the cost function

         $θ_j:=θ_j(1−α{m\over λ})−α{1\over m}∑_{i=1}^m(h_θ(x^i)−y^i)x_j^i  $

         basically, this is the traditional gradient descent plus a shrink of the $\theta$ by a factor less than $1$ every time. 

      4. Normal Equation w/ regularization: 

         $θ=(X^TX+λ⋅L)^{−1}X^Ty \text{     where L is a matrix whose diagonal is filled with 0111..1 } $

      5. A too large $\lambda$ may cause the hypothesis function to underfit, high bias (it penalize too much on the parameters that they don't have an effect on the output)

      6. Choosing $\lambda$ : try different $\lambda$ choices and pick whichever gives the lowest $J_{CV}$ 

   7. Learning Curve: $J_{train} \text{ and } J_{CV}$ against $m$(the training set size) 

      1. High bias: $J_{CV}$ will flatten out soon (hard to fit many training examples), same for $J_{train}$ getting more training data will not help
      2. High variance: $J_{CV}$ is high with even low training set size, and $J_{train}$ keeps low with high training size, getting more training data will help.

   8. Class-imbalanced Data Sets; Typical methods for imbalance data in 2-class classification:

      1. Oversampling: re-sampling of data from positive class
      2. Under-sampling: randomly eliminate tuples from negative class
      3. Threshold-moving: moves the decision threshold, t, so that the rare class tuples are easier to classify, and hence, less chance of costly false negative errors
      4. Ensemble techniques: Ensemble multiple classifiers introduced above

   9. Model Selection: ROC Curve, Receiver Operating Characteristics curves shows the trade off between true positive rate and the false positive rate, the area of ROC curve is a measure of the  accuracy of the model

      1. Accuracy classifier accuracy: predicting class label
      2. Speed: time to construct the model (training time); time to use the model (classification/prediction time)
      3. Robustness: handling noise and missing values
      4. Scalability: efficiency in disk-resident databases
      5. Interpretability understanding and insight provided by the model

   10. Improve Models:

       1. Ensemble methods Use a combination of models to increase accuracy Combine a series of k learned models, M1, M2, …, Mk, with the aim of creating an improved model M*
       2. Bagging: majority vote; Boosting: weighted vector; 
       3. Bagging + decision tree Each classifier in the ensemble is a decision tree classifier During classification, each tree votes and the most popular class is returned

   11. System Design: Build a simple algorithm $\to$ test on cross-validation $\to$ plot learning curve (figure out bias vs. variance) $\to$ Error Analysis