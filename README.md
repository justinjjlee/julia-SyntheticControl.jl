# julia implementation of Synthetic Control

The package is a julia implementation of [Abadie, Diamond, and Hainmueller (2010)](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.ap08746). The code was translated from MATLAB, the code provided by [Hainmueller](https://web.stanford.edu/~jhain/synthpage.html).

Some implementaitonal changes were made - most notably on calculating the loss function: instead of sum of square estimation, I log scale sum of square estimate. This helps optimization function to better locate the minima.

'''julia
ω = abs.(solveQadProg( H, f));
e = y₁pre - yₒpre*ω;

ln_ssr = log(sum(e.^2));
'''

Currently, the optimization step takes too much time. Unlike the MATLAB operation, the Quadratic form had to be linearized (using JuMP - accuracy of this can be seen by estimating based on MATLAB code's output of '''julia ν ''') and the loss function have to go through different optimization form.

See below plot

<p align="left">
  <img src="./src/plt.fin.svg">
</p>