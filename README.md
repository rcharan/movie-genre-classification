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

Note: fit times are reported in seconds of wall time, on a 2.7 GHz Intel Core i5 (dual core).<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>Model</th>      <th>Precision</th>      <th>Accuracy</th>      <th>Recall</th>      <th>F1</th>      <th>Cross-Entropy (Train)</th>      <th>Cross-Entropy (Test)</th>      <th>Fit Time</th>    </tr>    <tr>      <th>Number</th>      <th></th>      <th></th>      <th></th>      <th></th>      <th></th>      <th></th>      <th></th>      <th></th>    </tr>  </thead>  <tbody>    <tr>      <th>22</th>      <td>Stack -&gt; Tuned Thresholds</td>      <td>0.66</td>      <td>0.67</td>      <td>0.67</td>      <td>0.66</td>      <td>0.36</td>      <td>0.93</td>      <td>1211.32</td>    </tr>    <tr>      <th>21</th>      <td>Stack -&gt; Grid Search Logistic</td>      <td>0.66</td>      <td>0.67</td>      <td>0.67</td>      <td>0.66</td>      <td>0.36</td>      <td>0.93</td>      <td>12128.41</td>    </tr>    <tr>      <th>20</th>      <td>Stack -&gt; Logistic</td>      <td>0.66</td>      <td>0.67</td>      <td>0.67</td>      <td>0.67</td>      <td>0.33</td>      <td>0.94</td>      <td>2749.24</td>    </tr>    <tr>      <th>15</th>      <td>Variable Vectorizer, POS plot -&gt; Logistic</td>      <td>0.62</td>      <td>0.62</td>      <td>0.62</td>      <td>0.60</td>      <td>0.83</td>      <td>1.12</td>      <td>988.97</td>    </tr>    <tr>      <th>8</th>      <td>Count Vectorizer -&gt; GradientBoosting</td>      <td>0.56</td>      <td>0.55</td>      <td>0.55</td>      <td>0.52</td>      <td>0.85</td>      <td>1.28</td>      <td>156.57</td>    </tr>    <tr>      <th>10</th>      <td>Word-Embed Plot -&gt; LDA</td>      <td>0.60</td>      <td>0.60</td>      <td>0.60</td>      <td>0.60</td>      <td>1.05</td>      <td>1.29</td>      <td>27.59</td>    </tr>    <tr>      <th>13</th>      <td>Word Embed limited POS plot -&gt; LDA</td>      <td>0.60</td>      <td>0.61</td>      <td>0.61</td>      <td>0.60</td>      <td>1.08</td>      <td>1.34</td>      <td>2879.78</td>    </tr>    <tr>      <th>14</th>      <td>Latent Dirichlet Plot -&gt; LDA</td>      <td>0.45</td>      <td>0.45</td>      <td>0.45</td>      <td>0.42</td>      <td>1.36</td>      <td>1.35</td>      <td>18758.21</td>    </tr>    <tr>      <th>9</th>      <td>TF-IDF -&gt; Complement NB</td>      <td>0.54</td>      <td>0.56</td>      <td>0.56</td>      <td>0.54</td>      <td>0.84</td>      <td>1.64</td>      <td>55.18</td>    </tr>    <tr>      <th>18</th>      <td>Director -&gt; Bernoulli NB</td>      <td>0.42</td>      <td>0.44</td>      <td>0.44</td>      <td>0.41</td>      <td>0.93</td>      <td>1.67</td>      <td>20.82</td>    </tr>    <tr>      <th>12</th>      <td>Word-embed plot -&gt; SMOTE -&gt; LDA</td>      <td>0.60</td>      <td>0.53</td>      <td>0.53</td>      <td>0.54</td>      <td>1.53</td>      <td>1.90</td>      <td>1484.18</td>    </tr>    <tr>      <th>16</th>      <td>Latent Dirichlet plot -&gt; Random Forest</td>      <td>0.43</td>      <td>0.45</td>      <td>0.45</td>      <td>0.43</td>      <td>0.29</td>      <td>1.93</td>      <td>84.99</td>    </tr>    <tr>      <th>5</th>      <td>POS grid-search med -&gt; Complement NB</td>      <td>0.57</td>      <td>0.59</td>      <td>0.59</td>      <td>0.56</td>      <td>0.55</td>      <td>1.97</td>      <td>6503.43</td>    </tr>    <tr>      <th>7</th>      <td>Count Vectorizer, Plot, Year -&gt; AdaBoost(Tree)</td>      <td>0.45</td>      <td>0.44</td>      <td>0.44</td>      <td>0.41</td>      <td>2.02</td>      <td>2.04</td>      <td>570.44</td>    </tr>    <tr>      <th>19</th>      <td>Year -&gt; Complement NB</td>      <td>0.24</td>      <td>0.19</td>      <td>0.19</td>      <td>0.18</td>      <td>2.18</td>      <td>2.18</td>      <td>9.59</td>    </tr>    <tr>      <th>6</th>      <td>Limited parts-of-speech -&gt; SMOTE -&gt; Complement NB</td>      <td>0.57</td>      <td>0.49</td>      <td>0.49</td>      <td>0.49</td>      <td>0.73</td>      <td>2.47</td>      <td>11464.26</td>    </tr>    <tr>      <th>11</th>      <td>Count Vectorizer plot -&gt; Complement NB</td>      <td>0.55</td>      <td>0.56</td>      <td>0.56</td>      <td>0.55</td>      <td>0.57</td>      <td>3.46</td>      <td>53.39</td>    </tr>    <tr>      <th>3</th>      <td>Count-Vectorize -&gt; Complement NB</td>      <td>0.52</td>      <td>0.54</td>      <td>0.54</td>      <td>0.51</td>      <td>0.73</td>      <td>4.67</td>      <td>75.65</td>    </tr>    <tr>      <th>4</th>      <td>POS grid-search -&gt; Complement NB</td>      <td>0.52</td>      <td>0.54</td>      <td>0.54</td>      <td>0.51</td>      <td>0.67</td>      <td>4.77</td>      <td>184.41</td>    </tr>    <tr>      <th>1</th>      <td>Count-Vectorize -&gt; Multinomial NB</td>      <td>0.56</td>      <td>0.53</td>      <td>0.53</td>      <td>0.53</td>      <td>4.02</td>      <td>10.31</td>      <td>53.33</td>    </tr>    <tr>      <th>2</th>      <td>Count-Vectorize -&gt; Multinomial NB no prior</td>      <td>0.55</td>      <td>0.53</td>      <td>0.53</td>      <td>0.53</td>      <td>4.21</td>      <td>10.40</td>      <td>49.93</td>    </tr>    <tr>      <th>17</th>      <td>Count Vectorizer -&gt; SVC</td>      <td>0.58</td>      <td>0.56</td>      <td>0.56</td>      <td>0.51</td>      <td>NaN</td>      <td>NaN</td>      <td>86.97</td>    </tr>  </tbody></table>

