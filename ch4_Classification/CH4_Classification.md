# Classification

[TOC]

## Overview  

### What Is Classification

* Qualitative variables take values in an unordered set $\mathcal{C}$, such as:
  * Eye color $\in$ {brwon, blue, green}
  * email $\in$ {spam, ham}
* Given a feature vector $X$ and a qualitative response $Y$ taking values in the set $\mathcal{C}$ the classification task is to build a function $C(X)$ that takes as input the feature vector $X$ and predicts its value for $Y$; i.e. $C(X)\in\mathcal{C}$
* Often we are more interested in estimating the <span style="color:green">probabilities</span> that $X$ belongs tp each category in $\mathcal{C}$.
* ![](https://raw.githubusercontent.com/JingwenLiang/Statistical-Learning/master/ch4_Classification/creditCardDefualt.png)
  * Scatter plot: we can see from the scatter plot that balance is an important variable, there is a big seperation between the blues and the browns. Whereas with income, there doesn't seem to be much separation at all.
  *  Box plot (John Tukey): 
    * black line is the median, top 75th quartile, bottom 25th quartile (in the box)(distribution information), 
    * the hinges(things at the end) represent the fraction of the interquartile range, it gives you an idea of the spread of the data. If data points fall outside the hinges, they're considered to be outliers.

### Can We Use Linear Regression

* You can just fit a linear regression with RESPONSE as zero or 1 and classify as <span style="color:red">yes</span> if $\hat{Y} > 0.5$
  * In binary classes case, linear regression does a good job as classifier, and is equivalent to <span style="color:green">linear discriminant analysis</span> 
  * Since in the population $E(Y|X=x) = Pr(Y = 1|X=x)$ (the conditional expectation of a variable of zero or one, is the probability of that variable taking values of 1) might think  that regression is perfect for this task.
  * However, Linear regression might produce probabilities <span style="color:red">less than zero or bigger than one</span>, <span style="color:green;">Logistic Regression </span>is more appropriate.
* ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/logistic.png?raw=true)
  * linear regression: lower goes below 0---not a very good estimate of probability
  * not goes to high enough for linear regression
* also when you have more than 2 classes, assigning different values to the response as 1,2,3 seems dangerous. 

## Logistic Regression

### Logistic Regression Model

* let $p(X) = Pr(Y = 1|X)$ . Logistic regression uses the form

  ​	$$p(X) = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$$		

  * It is easy to see that no matter what values $\beta_0, \beta_1$or $X$ take, the $p(X)$ will have values between 0 and 1.

  * A bit of rearrangement gives 

  ​	$$log(\frac{p(X)}{1-p(X)}) = \beta_0 + \beta_1 X$$

  * This monotone transformation is called the <span style="color:green">log odds</span> or <span style="color:green">logit</span> transformation of $p(X)$.

### Parameter Estimate

* Maximum likelihood(Ronald Fisher):
  * $l(\beta_0, \beta_1) = \prod_{i:y_i = 1}p(x_i)\prod_{i:y_i = 0}(1-p(x_i))$
  * This <span style="color:green">likelihood</span> gives the probability of the observed zeros and ones in the data. We pikc $\beta_0$ and $\beta_1$ to maximize the likelihoood of the observed data. 

### Making Predictions

$$\hat{p}(X) = \frac{e^{\hat{\beta_0} + \hat{\beta_1}X}}{1 + e^{\hat{\beta_0} + \hat{\beta_1} X}}$$

#### Logistic Regression Using Categorical Data

* ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/binarydata.png?raw=true)




 

## Multivariate Logistic Regression

### Multivariate Logistic Regression

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

$$p(X) = \frac{e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}}$$

* ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/multivariate.png?raw=true)
  * the coefficient for student is negative this time (the correlations between the variable affect the signs)
  * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/multiexample.png?raw=true)
    * Students tend to have hight balances than non-students, so their marginal default rate is higher than for non-students.
    * But for each level of balance, students default less than non-students.
    * Multiple logistic regression can tease this out



 

## Case-Control Samling 

* We said in South Africa the risk for heart dissease is about 5% in this age category, But in our sample, we've got 160 cases and 302 controls, so in the sample we're showing a risk of 0.35=160/(160+302).

* In Logistic Regression, if you use all the samples that you have for the rare disease(in balanced class), and sample from the controls,  it can estimate the parameters of $X$'s accurately. But the constant term will be incorrect.

* Then you can just go ahead and correct the constant by a simple transformation.

* ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/caseControl1.png?raw=true)

  * $\tilde{\pi} = 0.35$  is the apparent risk of hear diseases
  * $\pi = 0.05$ is the true risk 
  * $log\frac{\tilde{\pi}}{1-\tilde{\pi}}$ is the logit transformation of the prior probability or the prior apparent probability.
  *  $log\frac{{\pi}}{1-{\pi}}$ is the logit transformation of the true risk 

* One thing to study the risk factors for heart diseases: take 1000 people and to follow them for 20 years and to record their risk factors and then see who gets heart disease.  We think about 5% will get hear disease.

  * but this takes 20 years and maybe more than a few thousand people to get enough ——not practical. We need a large ample, and we need many years to do it.
  * Case - control sampling says, 
    * let's not do things prospectively like that. Let's rather find people who we already know have heart disease or don't have heart disease and then sample them.
    * In this case we take 160 cases and 302 controls, record their risk factors.

* Very imblanced situations:

  * e.g: modelling the click-through rate on an ad on a web page, the probability of someone clicking is less than 1% maybe 0.005 or even smaller, which means if you just take a random sample of subjects who;ve been exposed to ads, you're going to get very very few 1's and a huge amount of 0's

* Do we need all the 0,1 data to fit the models?

  * No!  you can take a sample of the controls.

* ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/imbalanceddata.png?raw=true)

  * Ultimately the variance of  your parameter estimates has to do with the number of ases that you got, which is the smaller class.
  * the plot showing the variance of the coefficients as a function of the control/case  ratio.
  * So if you got a very sparse situation, sample about 5 or 6 controls for every cases, and now you can work with a much more manageable data set.


## Multiclass

### Logistic Regression With More Than Two Classes

* $$ Pr(Y = k | X) = \frac{e^{\beta_{0k} + \beta_{1k}X_1 + \cdots + \beta_{pk} X_p}}{\sum_{l = 1}^K e^{\beta_{0l} +\beta_{1l}X_1 + \dots + \beta_{pl}X_p}}$$
  * denominator: sum of those exponentials for all the classes.
  * in this case, each class gets its own linear model.
  * then we just weigh them against each other with this exponential function, sometimes called the <span style="color:green">softmax function</span> function.
  * some cancellation is possible, you only need $K-1$ linear functions 
  * <span style="color:green">Multinominal regression </span>



 ## Discriminant Analysis（a different classification method）

### What is Discriminant Analysis

*  Model the distribution if $X$ in each of the classes separately, and then use <span style="color:green">Bayes theorem</span> to flip things around and obtain $Pr(Y|X)$ 

* When we use normal distributions for each class, this leads to linear or quadratic discriminant analysis

* Other distributions can be used as well.

   

### Bayes Theorem for Classification

*  Bayes Theorem:

  ​	 $Pr(Y = k | X = x) = \frac{Pr(X = x | Y = k) \cdot Pr(Y = k)}{Pr(X=x)}$  

* $Pr(Y = k | X = x) = \frac{\pi_k f_k(x)}{\sum_{l = 1}^Kf_l(x)}$, where 
  * $f_k(x) = Pr(X=x|y=k)$ is the  <span style="color:green">density</span> for $X$ in class $k$. Here we will use normal densities for these, esparately in each class.
  * $\pi_k = Pr(Y = k)$ is the marginal or <span style="color:green">prior</span> probability for class $k$. 
* ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/discriminant.png?raw=trueg)
* When the classes are <span style="color:red">well-separated</span>, the parameter estimates for the <span style="color:red">logistic regression model are suprisingly unstable</span>. Linear discriminant analysis does not suffer from thie problem.
* If <span style="color:red"> is small and the distribution of the predictors  is approximately normal</span> in each of the classes, the linear discriminant model is again more stable than the logistic regression model.
* Linear discriminant analysis is <span style="color:red">popular when we have more than two reponse classes</span>, because it also provides low-dimensional views of the data. 

 

## Gaussian Discriminant - One Variable

### Linear Discriminant Analysis when $p = 1$

*  The Gaussian density has the form 
  * $f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_k}e^{-\frac{1}{2}\left(\frac{x-\mu_k}{\sigma_k}\right)^2}$ We will assume that all the $\sigma_k = \sigma$ are the same
  * Plugging this into Bayes formula, we get a rather complex cpmpression for $p_k(x) = Pr(Y = k | X = x)$:
    * $p_k = \frac{\pi_k \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(\frac{x-\mu_k}{\sigma}\right)^2}}{\sum_{l = 1}^K \pi_l \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(\frac{x-\mu_l}{\sigma}\right)^2} }$ 
    * To classifiy at the value $X = x$, we need to see which of the $p_k(x)$ is largest. Taking logs, and discarding terms that do not depend on $k$  we see that this is equivalent to assigning $x$  to the class with the largest <span style="color:green">discriminant score</span>:
      * $\delta_k(x) = x\cdot\frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + log(\pi_k)$ 
      * $\delta_k(x)$  is a <span style="color:green">linear</span> function of $x$   
    * If there are $K = 2$  classes and $\pi_1 = \pi_2 = 0.5$, then one can see that the <span style="color:green">decision boundary</span> is at 
      * $x = \frac{\mu_1 + \mu_2}{2}$ 
      * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/exampledisc.png?raw=true)
      * we need to estimate the means and the stand deviation
      *  ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/estimating.png?raw=true)
      * pooled variance estimate.

 

## Gaussian Discriminant - Many Variables

 ### Linear Discriminant Analysis when $p > 1$

* ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/gaussian2d.png?raw=true)

* 2D gaussian: bell function
* example：
  * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/p=2K=3.png?raw=true)
  * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/irisDATA.png?raw=true)
  * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/DiscriminantPlot.png?raw=true)
  * Once we have estimates $\hat{\delta_k}(x)$ , we can turn these into estimates for class probabilities:
    * $\hat{Pr}(Y = k | X = x) = \frac{e^{\hat{\delta_k}(x)} }{\sum_{l = 1}^K e^{\hat{\delta_l}(X)}}$   
  * so classifying to the largest $\hat{\delta_k(x)}$amounts to classifying to the class for which $\hat{Pr}(Y=k | X=x)$ is largest .
* example:
  * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/confusionmatrix.png?raw=true)

### Type of errors

* <span style="color:blue">False positive rate:</span> The fraction of negative examples that are classified as positive --- 0.2% in example
* <span style="color:blue">False negative rate:</span> The fraction of positive examples that are classified as negative --- 75.7% in example
* We produced this table by classifying to class <span style="color:red">yes</span> if 
  * $\hat{Pr}(\color{red}{\text{Default = Yes}}|\color{red}{\text{Balance, Student}}) \ge 0.5$  
* We can change the two error rates by changing the threshold from 0.5 to some other value in [0,1]:
  * $\hat{Pr}(\color{red}{\text{Default = Yes}}|\color{red}{\text{Balance, Student}}) \ge threshold$
  * and vary $threshold$
  * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/varyingthethreshold.png?raw=true)
  * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/ROC.png?raw=true)
  * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/Roc2.png?raw=true)
  * The Total Error is a weighted average of the False Positive Rate and False Negative Rage. The weights are determined by the Prior Probabilities of Positive and Negative Responses。

## Quadratic Discriminant Analyss and Naive Bayes

 ### Other forms of Discriminant Analysis

* By altering the forms for $f_k(x)$, we get different classifiers
  * With Gaussian but different $\Sigma_k$ in each class, we get <span style="color:green">quadratic discriminant analysis</span>.
  * With $f_k(x) = \prod_{j=1}^p f_{jk}(x_j)$  (conditional independence model) in each class we get <span style="color:green">naive Bayes</span>. For Gaussian this means the $\Sigma_k$ are diagonal. (independent features)
  * We end up with quiet flattened and maybe biased estimates for the probabilities, but in terms of classificatin, you just need to know which probability's the largest to classify it. So you can tolerate quite a lot of bias and still get good classification performance.
  * And what you get in return is much reduced variance from having to estimate far fewer parameters.
  * Many other forms, by proposing specific density models for $f_k(x)$ , including nonparametric approaches.
  * example:
    * ![](https://github.com/JingwenLiang/Statistical-Learning/blob/master/ch4_Classification/quadratic.png?raw=true) 

### Naive Bayes

* Assumes features are independent in each class.
* useful when p is large, and so multivariate methods like QDA and even LDA break down
  * Gaussian naive Bayes assumes each $\Sigma_k$ is diagonal:
    * $\begin{align*}\delta_k(x)  &\propto \log\left[\pi_k \prod_{j = 1}^p f_{kj}(x_j)\right] \\&=-\frac{1}{2}\sum_{j=1}^p\left[\frac{(x_j-\mu_{kj})^2}{\sigma_{kj}} + \log \sigma^2_{kj}\right] + \log \pi_k\end{align*}$
    * can use for <span style="color:green">mixed</span> feature vectors(qualitative and quantitative). If $X_j$ is qualitative, replace $f_{kj}(x_j)$  with probability mass function (histogram) over discrete categories.
  * Despite strong assumptions, naive Bayes often produces good classification results.

### Logistic Regression versus LDA

* For a two-class problem, one can show that for LDA
  * $\log\left(\frac{p_1(x)}{1-p_1(x)}\right) = \log \left(\frac{p_1(x)}{p_2(x)}\right) = c_0 + c_1x_1 + \cdots + c_px_p$
  * so it has the same form as logistic regression.
* The difference is in how the parameters are estimated.
  * Logistic regression uses the conditional likelihood based on $Pr(Y|X)$ (known as <span style="color:green">discriminative learning</span>) 
  * LDA uses the full likelihood based on $Pr(X, Y)$ (konwn as <span style="color:green">generative learning</span>)
  * Despite these differences, in practice the results are often very similar.
* logistic regression can also fit quadratic boundaries like QDA, by explicity including quadratic terms in the model.



## Summary

*  Logistic regression is very popular for classification especially when $K=2$
* LDA is useful when $n$ is small, or the classes are vell separated, and Gaussian assumptions are reasonable. Also when $K>2$ 
* Naive Bayes is useful when $p$  is very large. 