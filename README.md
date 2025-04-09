## Linear Algebra Exercise
Welcome to the exercise repository for day 3 of our course.
To help you, we have prepared unit-tests.
Use:
```shell
nox -s test
```
While coding, use `nox -s lint`, and `nox -s typing` to check your code.
Autoformatting help is available via `nox -s` format.
Feel free to read more about nox at https://nox.thea.codes/en/stable/ .

Today's goal is to get more familiar with the three important concepts of linear regression, fitting a polynomial of higher order and regularization.
You will be given some data in the form of pairs of (a, b)-values. 
So for each a-value there is one b-value. 
The main idea is to find an easy function (i.e. a polynomial) that best explains our data. 
That means, that if we plug in an a-value, the result should be close enough to the corresponding b-value. 

A general polynomial $f$ of order $n$ is given by:

$$ b = f(a) = c_1 + c_2 \cdot a^1 + c_3 \cdot a^2 + ... + c_n \cdot a^{n-1} $$

The $c$-values are the coefficients of the polynomial - these numbers are to be estimated!
The $a-$ and $b-$values are already given.
If we plug all the given values into the general form of the polynomial we get a system of linear equations which can be reformulated as a matrix multiplication:

$$
\begin{pmatrix}
          1       & a_1^1    & a_1^2  & \dots & a_1^{n-1}  \\\\ 
          1       & a_2^1    & a_2^2  & \dots & a_2^{n-1}  \\\\
          1       & a_3^1    & a_3^2  & \dots & a_3^{n-1}  \\\\
          \vdots  & \vdots   & \vdots  & \ddots & \vdots \\\\ 
          1       & a_m^1    & a_m^2  & \dots & a_m^{n-1}  \\\\
   \end{pmatrix} \cdot 
   \begin{pmatrix}
          c_1         \\\\ 
          \vdots   \\\\ 
          c_n       \\\\
   \end{pmatrix} = \begin{pmatrix}
          b_1         \\\\ 
          b_2       \\\\
          b_3       \\\\
          \vdots   \\\\ 
          b_m       \\\\
   \end{pmatrix}
$$

Or in short: 

$$\mathbf{A}_n\mathbf{c} = \mathbf{b}$$

The optimal $\mathbf{c}$ is given by 

$$\mathbf{A}_n^{\dagger}\mathbf{b} = \mathbf{c} ,$$

where $\mathbf{A}_n^{\dagger}$ is the Pseudo-Inverse.

For linear regression, the polynomial will be of order n=2 and will look like this:

$$ f(a) = c_1 + c_2 \cdot a $$

This will just be a straight line and there are only two coefficients $c_1$ and $c_2$ that have to be estimated.
Very often, this line is too simple to explain the data sufficiently.
That is why we want to fit polynomials of higher order, so $n > 2$.
Unfortunately, the more complex the model gets (i.e. the higher the order of the polynomial gets), the more noise will be tracked. Here we can make use of regularization techniques.
In the first and following part, you will be given artificial data and in the second part you will make use of real data!

### Part 1: Proof of concept
The line `b = pandas.read_csv('./data/noisy_signal.tab')` is used to load a noisy signal.
The line `x_axis = np.linspace(0, 1, num=len(b_noise))` will provide you with corresponding x-values (these are the a-values from above and the lecture).
This is some artificial data that serves as a means to try out the concepts you have learned about in the lecture.
The first part will be concerned with modeling this signal using polynomials.

####  ⊙ Task 1.1: Regression
Linear regression is usually a good first step. 

1. Start by implementing the function `set_up_point_matrix` from the `src/regularization.py` module. 
The function should produce polynomial-coordinate matrices $\mathbf{A}_n$ of the form:
$$
\mathbf{A}_n = 
\begin{pmatrix}
          1       & a_1^1    & a_1^2  & \dots & a_1^{n-1}  \\\\ 
          1       & a_2^1    & a_2^2  & \dots & a_2^{n-1}  \\\\
          1       & a_3^1    & a_3^2  & \dots & a_3^{n-1}  \\\\
          \vdots  & \vdots   & \vdots  & \ddots & \vdots \\\\ 
          1       & a_m^1    & a_m^2  & \dots & a_m^{n-1}  \\\\
   \end{pmatrix}
$$ 

2. Go to the main-function and use the function you just implemented to create the point-matrix A for n=2.

3. Now,
$$\mathbf{A}_2^{\dagger}\mathbf{b} = \mathbf{c} = \begin{pmatrix}
          c_1         \\\\ 
          c_2       \\\\
   \end{pmatrix} $$
will produce the coefficients for a straight line. 

4. Evaluate your first-degree polynomial via $c_1 + c_2 \cdot x$ and plot the result as well as the original data using `matplotlib.pyplot`'s `plot` function.


Solution:

![regression](./figures/regression.png)

#### ⊙Task 1.2: Fitting a Polynomial to a function
The straight line above is insufficient to model the data. 
So perform the very same steps as above, but change the degree of the polynomial to n=300  (to set up a square matrix since we have 300 data-points):
1. Set up the point matrix by setting n=300.
2. Estimate the coefficients via $$\mathbf{A}^{\dagger}\mathbf{b} = \mathbf{x}_{\text{fit}}.$$
3. Having estimated the coefficients $$\mathbf{A} \mathbf{x}_{\text{fit}}$$ computes the function values. 
   Plot the original points and the function values using matplotlib.
What do you see?


Solution:

![regression](./figures/polyfit.png)


#### ⊙Task 1.3: Regularization
Unfortunately, the fit is not ideal. The polynomial is too complex and tracks the noise.
The singular value decomposition (SVD) can help!
Recall that the SVD turns a matrix

$$\mathbf{A} \in \mathbb{R}^{m,n}$$

into the form:

$$\mathbf{A} = \mathbf{U} \Sigma \mathbf{V}^T$$

In the SVD-Form, computing the pseudoinverse is simple! Swap U and V  and replace every of the m singular values with it's inverse

$$1/\sigma_i .$$

This results in the matrix 
```math
\Sigma^\dagger = \begin{pmatrix}
      \sigma_1^{-1} & & & \\\\
      &  \ddots & \\\\
      &  & \sigma_m^{-1} \\\\ \hline
      & 0 &
\end{pmatrix}
```

A solution to the overfitting problem is to filter the singular values. 
The idea is, that small singular values often correspond to directions in the data where noise dominates. And now we want to create a filter matrix, that gets rid of singular values if they are too small (i.e. if they fall below a threshold $\epsilon$).
Compute a diagonal for a filter matrix by evaluating:

$$f_i = \sigma_i^2 / (\sigma_i^2 + \epsilon)$$


Roughly speaking multiplication by $f_i$ will filter a singular value when

$$\sigma_i \lt \epsilon ,$$
since in this case, $f_i$ will be close to $0$.
If however 
$$\sigma_i \geq \epsilon ,$$
$f_i$ will be closer to $1$ and the respective singular value will not be filtered out.

Apply the regularization by computing:

$$
    \mathbf{x}_r= \mathbf{V} \mathbf{F} \begin{pmatrix}
      \sigma_1^{-1} & & & \\\\
      &  \ddots & \\\\
      &  & \sigma_n^{-1} \\\\ \hline
      & 0 &
    \end{pmatrix}
    \mathbf{U}^T \mathbf{b} = \mathbf{V} \mathbf{F} \mathbf{\Sigma}^\dagger
    \mathbf{U}^T \mathbf{b}
$$


with

$$\mathbf{V} \in \mathbb{R}^{n,n}, \mathbf{F} \in \mathbb{R}^{n,n}, \Sigma^{\dagger} \in \mathbb{R}^{n,m}, \mathbf{U} \in \mathbb{R}^{m,m} \text{ and } \mathbf{b} \in \mathbb{R}^{m,1}.$$
  
Setting n=300 turns A into a square matrix. In this case, the zero block in the sigma-matrix disappears and you don't have to worry about transposing $\sigma$ when computing the pseudoinverse.

To sum it up, your tasks are:
1. Compute the SVD of A.

Perform the following steps 2. - 4. for epsilon equal to 0.1, 1e-6, and 1e-12.
2. Compute the diagonal for the filter matrix and turn it into a matrix. 
3. Estimate the regularized coefficients by applying the formula above.
4. Plot the result.

Solution:

![regression](./figures/regularized_fit.png)

#### ✪Task 1.4: Model Complexity (Optional):
Another solution to the overfitting problem is reducing the complexity of the model.
To assess the quality of polynomial fit to the data, compute and plot the Mean Squared Error (Mean Squared Error measure how close the regression line is to data points) for every degree of polynomial upto 20.
So as before:
1. Set up the point matrix for the current degree from 1 to 20.
2. Estimate the coefficients.
3. Compute the predictions.
4. Calculate the MSE.
MSE can be calculated using the following equation, where $N$ is the number of samples, $y_i$ is the original point and $\hat{y_i}$ is the predicted output.
$$MSE=\frac{1}{N} \sum_{i=1}^{N} (y_i-\hat{y_i})^2$$
5. Plot the MSE-error against the degree.
6. Are the degree of the polynomial and the MSE linked?
   From the plot, estimate the optimal degree of polynomial and fit the polynomial with this specific degree.

Solution:

![model_complexity](./figures/model_complexity_mse.png)



Solution:

From the plot we observe that after degree 7, the mean squared error doesn't reduce substantially.

![model_complexity](./figures/model_complexity_fit.png)

### Part 2: Rhine water level analysis
Now we are ready to deal with real data! Feel free to use your favorite time series data or work with the Rhine level data we provide.
The file `./data/pegel.tab` contains the Rhine water levels measured in Bonn over the last 100 years. 
Data source: https://pegel.bonn.de.

#### ⊙Task 2.1 Regression
The `src/pegel_bonn.py` file already contains code to pre-load the data for you.
The Rhine level measurements will be your new vector $\mathbf{b}$ from before.
Now we want to do the same as in Part 1 and start with linear regression!
1.  Generate a matrix A with n=2 using the timestamps for the data set as your x-values. 
2.  Compute $$\mathbf{A}^{\dagger}\mathbf{b}$$ to estimate the coefficients of the line.
3.  Evaluate your polynomial and plot the result.
4. Compute the zero-intercept with the y-axis. When do the regression line and the x-axis intersect? 
   Or in other words: On which day will the Rhine water level be at 0 cm? 
   > **Hint:** Plug in $y=0$ into the equation of your line with the estimated coefficients and solve for the date $x$.

Solution:

![regression](./figures/rhine_regression.png)


#### ⊙ Task 2.2: Fitting a higher-order Polynomial

Re-using the code you wrote for the proof of concept task, fit a polynomial of degree 20 to the data. Before plotting have a closer look at `datetime_stamps` and its values and scale the axis appropriately.

1. Scale the x-axis.
2. Set up the point matrix for the scaled x-axis with n=20.
3. Compute the coefficients.
4. Evaluate the polynomial.
5. Plot the result.

Solution:

![rhine_polyfit](./figures/rhine_polyfit.png)



#### ⊙Task 2.3: Regularization
Focus on the data from the year 2000 onward and filter the singular values.
We will use again a degree of 20.
Matrix A is not square in this case, because the degree is smaller than the number of datapoints! Consequently, a zero block must appear in your singular value matrix and when computing the Pseudoinverse from the SVD, $\sigma$ has to be transposed! 
Like in Part 1:
1. Compute the SVD of the point matrix from the previous task.

Perform the following steps 2. - 4. for epsilon equal to 0.1, 1e-3, and 1e-9.
2. Compute the filter matrix.
3. Estimate the regularized coefficients by applying the formula from before. 
   > **Hint:** Remember the zero-block! You need degree-many rows and number-of-datapoints-many columns!
4. Evaluate the regularized polynomial and plot the results.


Solution:

![rhine_reg_fit](./figures/rhine_regularized_fit.png)

