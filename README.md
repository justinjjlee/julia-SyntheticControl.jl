# julia implementation of Synthetic Control method

The package is a julia implementation of [Abadie, Diamond, and Hainmueller (2010)](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.ap08746). The code was translated from MATLAB, the code provided by [Hainmueller](https://web.stanford.edu/~jhain/synthpage.html).

Some implementaitonal changes were made - most notably on calculating the loss function: instead of sum of square estimation, I log scale sum of square estimate. This helps optimization function to better locate the minima.

```julia
ω = abs.(solveQadProg( H, f));
e = y₁pre - yₒpre*ω;

ln_ssr = log(sum(e.^2));
```

Currently, the optimization step takes too much time. Unlike the MATLAB operation, the quadratic programming had to be linearized (using [JuMP](https://github.com/jump-dev/JuMP.jl) - see final estimation of  ``` ν```) and the loss function (which nests the quadratic programming, not able to linearize) using [Optim](https://julianlsolvers.github.io/Optim.jl/stable/).

## Example plot replicating [Abadie, Diamond, and Hainmueller (2010)](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.ap08746) 
<p align="left">
  <img src="https://github.com/justinjoliver/julia-synthetic_control/blob/main/src/plt_fin.svg">
</p>
