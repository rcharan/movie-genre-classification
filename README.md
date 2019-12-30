# Movie Genre Classification
This is my submission for a "bakeoff" for our Flatiron School Data Science program.

## Data
__Train__: Approximately 10,000 movies with their genre, title, year, director, cast (possibly missing)
and a plot description.

__Test__: Approximately 3,500 movies with the same information but possibly without a genre

The test set was held out by our instructor and was evaluated separately. The
relevant metric was class-weighted F1 score.

## Task
9-class classification, primarily with NLP techniques.

## Approach
I tried some combinations of a variety of strategies:
1. **Pre-process** Grid Search for the Parts of Speech
2. **Embedding** Count Vectorization, TF-IDF vectorization, a word embedding
(pre-computed by the spaCy package), or Latent Dirichlet Allocation
3. (optional) SMOTE
4. **Classify** primarily Bernoulli/Multinomial Naive Bayes, Complement Naive Bayes, Logistic Regression,
and Linear Discriminant Analysis (LDA).

A smattering of other classifiers (Support-Vector Classifier, AdaBoost, Random Forest, Gradient-boosted trees) were also tried.

Hyper-parameter searches of varying degrees of comprehensiveness were performed.

Finally, a stacked classifier was fit with Logistic Regression as the final estimator.
I also tried grid-search on the logistic regression and tuning the thresholds on this stacked
classifier without much change in results.

## Results
Most models were clustered in a range of F1 scores from 50–60% on a held-out validation set. The
stacked classifier was able to accomplish about 66%.

Note: fit times are reported in seconds of wall time, on a 2.7 GHz Intel Core i5 (dual core).
