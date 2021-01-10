# Function for Synthetic Control
function normalz(xₒ, x₁, idx_control, idx_treat)
    x = hcat(xₒ, x₁); 
    k, n = size(x);
    kₒ = size(xₒ, 1);
    k₁ = size(x₁, 1);

    xscale = 1 ./ std(x, dims = 2);

    xmat = ((x') * (eye(k) .* xscale))';
    xₒscale = xmat[1:kₒ, idx_control];
    x₁scale = xmat[1:k₁, idx_treat];

    return xₒscale, x₁scale;
end

# Create diagonal identiy matrix
eye(n) = Matrix{Float64}(I, n, n);

# Standardize matrix
function func_std(mat)
    # Function to standardize data
    T, k = size(mat);
    μ = mean(mat; dims = 1);
    σ = std(mat; dims = 1);
    return (mat .- μ) ./ σ;
end

# Loss function
function loss_ssr(ν)
    l = size(yₒpre, 2);

    ν = [1; ν];
    D = (eye(length(ν)) .* ν);

    H = xₒ' * D * xₒ;
    f = -1 .* (x₁' * D * xₒ);
    
    ω = abs.(solveQadProg(H, f));

    e = y₁pre - yₒpre*ω;
    ln_ssr = log(sum(e.^2));
    return ln_ssr
end 
# More on the solver type: 
# https://jump.dev/JuMP.jl/v0.21.1/installation/;
# Useful website on vectorization
# https://www.softcover.io/read/7b8eb7d0/juliabook/optimization
function solveQadProg(H, f)#, ceq, beq)
    #model = Model(with_optimizer(GLPK.Optimizer))
    optimizer = Juniper.Optimizer
    nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)

    model = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver))

    k = size(H, 1);

    @variable(model, x[1:k])
    @constraint(model, [i = 1:k], 0 <= x[i] <= 1)

    @objective(model, Min, 
                    (
                        sum(0.5 * H[i,j] * x[i] * x[j] for i = 1:length(x), j = 1:length(x)) +
                        sum( f[i] * x[i] for i = 1:length(x))
                    )
                );
    @constraint(model, sum( x[i] for i = 1:length(x) ) == 1)
    
    optimize!(model)

    res = JuMP.value.(x);
    return res
end