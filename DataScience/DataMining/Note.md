# Data Mining Note
## Data

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

4. | Attribute Type | Description                                                  | Examples                                                     | Operations                                                   |
   | :------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
   | Nominal        | The values of a nominal attribute are just different names, i.e., nominal attributes provide only enough information to distinguish one object from another. ($=\neq$) | zip codes, employee ID numbers, eye color, sex: {male, female} | mode, entropy, contingency correlation, $\mathcal{X} ^2$ test |
   | Ordinal        | The values of an ordinal attribute provide enough information to order objects. (<, >) | hardness of minerals, {good, better, best}, grades, street numbers | median, percentiles, rank correlation, run tests, sign tests |
   | Interval       | For interval attributes, the differences between values are meaningful, i.e., a unit of measurement exists. (+, -) | calendar dates, temperature in Celsius or Fahrenheit         | mean, standard deviation, Pearson's correlation, $t$ and $F$ tests |
   | Ratio          | For ratio variables, both differences and ratios are meaningful. (*, /) | temperature in Kelvin, monetary quantities, counts, age, mass, length, electrical current | geometric mean, harmonic mean, percent variation             |

5. Discrete vs. Continuous:

   1. Discrete: Has only a finite or countable infinite set of values, always represented by integer number
   2. Continuous: Real number as attribute value, always represented by floating point number
   3. Nominal and ordinal are always discrete, interval and ratio are always continuous. 

6. Summary Statistics of Data: central tendency vs. variability or dispersion 

7. Computational: 

   1. Distributed measure –can be computed by partitioning the data into smaller subsets.
   2. Algebraic measure –can be computed by applying an algebraic function to one or more distributed measures.
   3. Holistic measure –must be computed on the entire dataset as a whole.

8. Central Tendency: Mean, median mode. How data seem similar, location of data

   1. Skewed vs. Symmetric (mode = median =mean)

9. Dispersion: The degree to which numerical data tend to spread

   1. Range: difference between the largest and smallest values
   2. Percentile: the value of a variable below which a certain percent of data fall
   3.  Quartiles: Q1 (25th percentile), Median (50th percentile), Q3 (75th percentile)
   4. Inter-quartile range: IQR = Q3 – Q1
   5. Five number summary: min, Q1, M, Q3, max (Boxplot)
   6. Outlier: usually, a value at least 1.5 x IQR higher/lower than Q3/Q1
   7. Variance and standard deviation

10. Graphic Display of Basic Statistical Description:

    1. Boxplot![1522090608674](C:\Users\gabri\AppData\Local\Temp\1522090608674.png)
    2. Histogram: equal-width vs. equal-depth; A set of rectangles that reflect the counts or frequencies of values at the bucket (bar chart)
    3. Scatter Plot: Displays values for two numerical attributes (bivariate data); can suggest correlations between variables with a certain confidence level: positive (rising), negative (falling), or null (uncorrelated).

11. Data pre-processing:

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
    5.  Data Integration: combines data from multiple sources into a unified view; 
       1. Data warehouse: tightly coupled
          1. advantage: high query performance, can operate when source is not available, extra aggregated information from warehouse, not affected by the local processing
          2. disadvantage: data freshness, hard to build when only have access to query interface of local source
       2. Federated Database Systems: use a mediator and a wrapper to access different source
          1. Advantage: No need to copy and store data at mediator, More up-to-date data, Only query interface needed at sources
          2. Disadvantage: Query performance, Source availability
    6. Database Heterogeneity: 
       1. System Heterogeneity: use of different operating system, hardware platforms
       2. Schematic or Structural Heterogeneity: the native model or structure to store data differ in data sources.
       3. Syntactic Heterogeneity: differences in representation format of data
       4. Semantic Heterogeneity: differences in interpretation of the 'meaning' of data
    7. Data Matching: 
       1. Semantic Integration: reconciling semantic heterogeneity; schema-matching (A.cusit_id = B.cust-#), data matching (Bill Clinton = William Clinton)
       2. Schema Matching: rule based vs. learning based; 1-1 matches vs. complex matches
          1. Probabilistic record linkage: Similarity between pairs of attributes, combined scores representing probability of matching, threshold based decision. 
    8. Data Transformation: 
       1. Aggregation: sum/count/average E.g. Daily sales -> monthly sales
       2. Discretization and generalization E.g. age -> youth, middle-aged, senior
          1. Discretization: transform continuous attribute into discrete counterparts (intervals) Supervised vs. unsupervised Split (top-down) vs. merge (bottom-up)
          2. Generalization: generalize/replace categorical attributes from low level concepts (such as age ranges) by higher level concepts (such as young, middle-aged, or senior)
       3. (Statistical) Normalization: scaled to fall within a small, specified range E.g. income vs. age Not to be confused with database normalization and text normalization: 
       4. Attribute construction: construct new attributes from given ones E.g. birthday -> age
    9. Data Reduction
       1. Instance reduction
          1. Sampling (instance selection): obtaining a small representative samples to represent the whole data set N A sample is representative if it has approximately the same property (of interest) as the original set of data
             1. Sampling Method: 
                1. Simple Random Sampling: There is an equal probability of selecting any particular item
                2. Sampling without replacement; As each item is selected, it is removed from the population
                3. Sampling with replacement: Objects are not removed from the population as they are selected for the sample -the same object can be picked up more than once
                4. Stratified sampling Split the data into several partitions (stratum); then draw random samples from each partition
                5.  Cluster sampling When "natural" groupings are evident in a statistical population; sample a small number of clusters
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

    ## Frequent Item Mining

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
       2. Max Pattern: If X is frequent, and there exists no super pattern Y that is frequent. ![1522123316676](C:\Users\gabri\AppData\Local\Temp\1522123316676.png)

    ## Classification

    1. Supervised Learning, the training data are accompanied by labels indicating the class of the observation, and new class is classified based on the training set. 

    2. Classification vs. Numeric Prediction: predicts categorical class labels (discrete or nominal), models continuous-valued functions

    3. Decision Tree Induction: 

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
       2. Over-fitting: too many branches may reflect anomalies and noises. 
          1. Pre-pruning: halt tree construction early; Post-pruning: remove branches from a 'fully grown ' tree 
          2. Random forest (Ensemble)

    4. Bayes Classification: A statistical classifier, performs probabilistic prediction

       1. Bayes theorem: $P(H|X) ={ {P(X|H)P(H)} \over{P(X)}}$
          1. Informally, this can be viewed as: posteriori = likelihood x prior/evidence
             1. P(H) (prior probability): the initial probability
             2. P(X): probability that sample data is observed
             3. P(X|H) (likelihood): the probability of observing the sample X, given that the hypothesis holds
          2. Classification is to maximize the posteriori, which is practical difficult as it requires initial knowledge of many probabilities,
       2. Naïve Bayes: assuming all attributes are conditionally independent (the product of individual conditional probabilities)
          1. Avoiding zero-probability: laplacian correction (adding 1 to each case)
          2. Advantage: easy to implement, good result; disadvantage: assumption loss of accuracy, dependencies exist among variables

    5. Model Evaluation and Selection: 

       1. Evaluation Matrix: 

          | Actual class\Predicted class | C1             | ¬ C1            |      |
          | ---------------------------- | -------------- | --------------- | ---- |
          | C1                           | True Positive  | False Negatives | P    |
          | ¬ C1                         | False Positive | True Negatives  | N    |
          |                              | P'             | N'              | All  |

          1. Classifier Accuracy, or recognition rate: percentage of test set tuples that are correctly classified Accuracy = (TP + TN)/All Error rate:1 –accuracy, or Error rate = (FP + FN)/All
          2. Sensitivity (Recall): True Positive recognition rate Sensitivity = TP/P 
          3. Specificity: True Negative recognition rate Specificity = TN/N
          4. Precision: TP/(TP+FN)
          5. F measure: harmonic mean of precision and recall $F = {{2 \times \text{precision} \times \text{recall}} \over {precision+recall}}$ 

       2. Methods:

          1. Holdout method, random subsampling

             1. Given data is randomly partitioned into two independent sets Training set (e.g., 2/3) for model construction Test set (e.g., 1/3) for accuracy estimation
             2. Random sampling: a variation of holdout Repeat holdout k times, accuracy = avg. of the accuracies obtained

          2. Cross-validation

             Randomly partition the data into $k$ mutually exclusive subsets, each approximately equal size; At i-th iteration, use Di as test set and others as training set; Leave-one-out: k folds where k= # of tuples, for small sized data; *Stratified cross-validation*: folds are stratified so that class dist. in each fold is approx. the same as that in the initial data

          3. Bootstrap: A resampling technique, works well with small data sets, Samples given data for training tuples uniformly with replacement

       3. Class-imbalanced Data Sets; Typical methods for imbalance data in 2-class classification:

          1. Oversampling: re-sampling of data from positive class
          2. Under-sampling: randomly eliminate tuples from negative class
          3. Threshold-moving: moves the decision threshold, t, so that the rare class tuples are easier to classify, and hence, less chance of costly false negative errors
          4. Ensemble techniques: Ensemble multiple classifiers introduced above

       4. Model Selection: ROC Curve, Receiver Operating Characteristics curves shows the trade off between true positive rate and the false positive rate, the area of ROC curve is a measure of the  accuracy of the model

          1. Accuracy classifier accuracy: predicting class label
          2. Speed: time to construct the model (training time); time to use the model (classification/prediction time)
          3. Robustness: handling noise and missing values
          4. Scalability: efficiency in disk-resident databases
          5. Interpretability understanding and insight provided by the model

    6. Improvement: 

       1. Ensemble methods Use a combination of models to increase accuracy Combine a series of k learned models, M1, M2, …, Mk, with the aim of creating an improved model M*
       2. Bagging: majority vote; Boosting: weighted vector; 
       3. Bagging + decision tree Each classifier in the ensemble is a decision tree classifier During classification, each tree votes and the most popular class is returned

    ## Advanced Classification

    1. Bayesian belief network (probabilistic networks): allows conditional independencies between subsets of variables

       1. Two components: 

          (1) A directed acyclic graph (called a structure) : Represents dependency among the variables Each node has a conditional probability distribution P( X | parents(X) )

          (2) a set of conditional probability tables (CPTs)

       2. Using chain rule: $P(X_1, ..., X_n) = \prod ^n _i{P(X_i|Parents(X))}$

    2. Support Vector Machine (SVM)

       1. Large-Margin Linear Classifier: margin is the boundary could be increased by before hitting a data point, robust to outliers and thus strong generalization ability;
          1. Linear function: $g(x) = w^T x+b$ 
          2. $w^T x^+ +b = 1$, $w^T x^- +b = -1$ $M = {(x^+ - x^-) \cdot n ={ (x^+ - x^-) \cdot {w\over||w||}} ={ 2\over ||w||}}$
          3.  Solve the optimization problem
       2. Non-linear SVM: mapped to higher-dimensional feature space where training set is separable, use kernel trick

    3. Neural networks

       1. Use hidden layer to keep adjusting weights for better results; Gradient Descent (all), Stochastic (single), Mini-Batch (small set of tuples)

    4. lazy learners (KNN)

       1. Lazy vs. Eager: Lazy learning(e.g., instance-based learning): Simply stores training data (or only minor processing) and waits until it is given a test tuple; Eager learning(the above discussed methods): Given a set of training tuples, constructs a classification model before receiving new (e.g., test) data to classify
       2. KNN: return the average of k-nearest neighbors, distance-weight assigning closer neighbor a higher weight, robust to noisy data by averaging k-nearest 

    ## Clustering

    1. Finding groups of objects (clusters) Objects similar to one another in the same group Objects different from the objects in other groups;

    2. A good clustering will produce high quality clusters with Homogeneity - high intra-class similarity Separation - low inter-class similarity

       1. Similarity Measurement: Euclidean / Manhattan / Minkowski distances
       2. Normalize different data type to enable distance calculation
       3. Jaccard distance/similarity of two sets s the size of their intersection divided by the size of their union

    3. Partitioning methods

       1. Construct a partition of n objects into k clusters,s.t. intra-cluster similarity maximized and inter-cluster similarity minimized

       2. The global optimal is NP hard

       3. Heuristic methods: k-means and k-medoids algorithms

          1. k-means : Each cluster is represented by the center
             of the cluster: this is easy to implement but depending on initial centroids, may terminate at a local optimum, and is sensitive to noisy data and outliers

             1.  Given k , and randomly choose k initial cluster centers
             2. Partition objects into k nonempty subsets by assigning
                each object to the cluster with the nearest centroid
             3. Update centroid, i.e. mean point of the cluster
             4. Go back to Step 2, stop when no more new
                assignment

          2. k-medoids : Each cluster is represented by one of the objects
             in the cluster, Instead of taking the mean value of the object
             in a cluster as a reference point, medoids can be used,
             which is the most centrally located object in a cluster

             1. Arbitrarily select k objects as medoid

             2.  Assign each data object in the given data set to most similar
                medoid.

             3.  Randomly select nonmedoid object O’

             4. Compute total cost, S, of swapping a medoid object to O’ (cost as

                total sum of absolute error)

             5. If S<0, then swap initial medoid with the new one

             6. Repeat until there is no change in the medoid.

    4. Hierarchical methods: Produces a set of nested clusters organized as a
       hierarchical tree

    5.  Outlier analysis and Evaluation: 

       1. Unsupervised (internal): Used to measure the goodness
          of a clustering structure without respect to external
          information. Sum of Squared Error (SSE)
          1.  Cluster Cohesion: how closely related are objects in a
             cluster;  Cluster Separation: how distinct or well-separated a
             cluster is from other clusters
       2.  Supervised (external): Used to measure the extent to
          which cluster labels match externally supplied class labels. Entropy
       3.  Relative: Used to compare two different clustering results; Often an external or internal index is used for this function, e.g., SSE
          or entropy

    ​