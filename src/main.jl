using Pkg;
using DelimitedFiles, DataFrames
using LinearAlgebra, Statistics, JuMP, Ipopt, Juniper, Optim
using Gadfly, Cairo, Compose;
cd(@__DIR__) #src

include("func_synth.jl")
dat, ~ = readdlm("MLAB_data.txt", header = true); 

# Index to group the data: Control vs. treated ============================
idx_control = 1:38;
idx_treat   = 39;

# Group data ==============================================================
# Endogenous time-invariant predictor -------------------------------------
idx_predictor = 1:7
xₒ = dat[idx_predictor, idx_control];         # Control group
x₁ = dat[idx_predictor, idx_treat];           # Treatment group

# Rescale predictors
global xₒ, x₁ = normalz(xₒ, x₁, idx_control, idx_treat);

# Outcome data to evaluate ------------------------------------------------
idx_outcome = 8:38;
yₒ = dat[idx_outcome, idx_control];           # Control group
y₁ = dat[idx_outcome, idx_treat];             # Treatment group

idx_pretreat = 1:19;                          # Rows of pre-treat period
global yₒpre = yₒ[idx_pretreat, :];
global y₁pre = y₁[idx_pretreat, :];

# Optimization
s = std(hcat(xₒ, x₁)', dims = 1)';
s2 = s[2:end];
s1 = s[1];
ν₀ = ((s1./s2).^2);
 
# Optimization to estimate weights ========================================
#https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/
#https://julianlsolvers.github.io/Optim.jl/stable/#algo/samin/
q = (length(ν₀))
opt = optimize(loss_ssr, fill(0.0, q), fill(5e8, q), ν₀, 
                Fminbox(BFGS()), Optim.Options(iterations=100) # or 100_000
            );
# NOTE: Requires further improvement, as the optimization takes a long time.
# Can try with different optimization methods
opt = optimize(loss_ssr, fill(0.0, q), fill(5e8, q), ν₀, 
                  SAMIN(), Optim.Options(iterations=1_000_000) # or 100_000
            );

# Save results
ν = vcat(1, Optim.minimizer(opt));

#MATLAB optimal solutions - to check the accuracy of solveQadProg(H, f)
#=
ν = [1, 
      780.438610055690,
      24909.6348724115,
      37806.0921140682,
      124.042538397502,
      60413.0041399961,
      63630.9996910117];
=#
# Recover weights: w-weights
D = (eye(length(ν)) .* ν);

H = xₒ' * D * xₒ;
f = -1 .* (x₁' * D * xₒ);

ω = abs.(solveQadProg(H, f)); # we know this works now.

# Build counterfactual output.
ŷₒ = yₒ * ω;
ȳₒ = mean(yₒ, dims = 2); # Res of US, for plotting purpose

# plot ====================================================================
yr = 1970:1:2000;
plt_yavg = layer(x = yr, y = y₁, Geom.line, 
                  Theme(default_color=colorant"deepskyblue", 
                        line_style = [:solid]),
                  xintercept = [1988], 
                        Geom.vline(color = "black", style=[[0.5mm,0.5mm]])
            );
plt_y    = layer(x = yr, y = ȳₒ, Geom.line, 
                  Theme(default_color=colorant"red", line_style = [:solid]));
plt_yhat = layer(x = yr, y = ŷₒ, Geom.line, Theme(line_style = [:dash]));
plt_fin = plot(plt_yavg, plt_y, plt_yhat, 
            Guide.xlabel("Year"), 
            Guide.ylabel("Somking per capita (in packs)"),
            Guide.annotation(compose(context(), 
                        Compose.text(1988.5, 25, "← Proposition 99"))),
            Guide.annotation(compose(context(), 
                        Compose.text(1993, 110, "Rest of U.S."))),
            Guide.annotation(compose(context(), 
                        Compose.text(1971, 95, "Solid: California (actual) \n Dash: California (synthetic)")))
      )
pngout = SVG("plt_fin.svg", 5inch, 4inch)
draw(pngout, plt_fin)

