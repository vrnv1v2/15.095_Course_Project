# Best subset selection as a mixed-integer quadratic program (JuMP + Gurobi).
#
# Given training data X (n × p) and labels coded y ∈ {-1, +1} (NRB = +1), solve
#
#     min_{β0, β, z}  (1/n) ‖y − β0 − Zβ‖²  +  λ ‖β‖²
#     s.t.            z_j = 0  ⇒  β_j = 0          (indicator constraints, no big-M needed)
#                     Σ_j z_j ≤ k,   z ∈ {0, 1}^p
#
# where Z is X standardized with the training mean/std. The small ridge term λ keeps the
# problem well conditioned when descriptors are correlated. The k features with z_j = 1 are
# the selected subset.

using JuMP, Gurobi, LinearAlgebra, Statistics

const GUROBI_ENV = Ref{Any}(nothing)   # one shared env so the license is checked once

function gurobi_env()
    if GUROBI_ENV[] === nothing
        GUROBI_ENV[] = Gurobi.Env()
    end
    return GUROBI_ENV[]
end

"""
    best_subset(X, labels, k; λ=1e-2, time_limit=120) -> (selected, info)

`X` is a numeric matrix or DataFrame, `labels` a vector of "NRB"/"RB" strings.
Returns the sorted column indices of the selected features and a Dict with the solver
status, objective, MIP gap, and solve time.
"""
function best_subset(X, labels, k::Integer; λ = 1e-2, time_limit = 120)
    Xm = Matrix{Float64}(X)
    n, p = size(Xm)
    y = [l == "NRB" ? 1.0 : -1.0 for l in labels]
    μ = mean(Xm, dims = 1)
    σ = std(Xm, dims = 1)
    σ[σ .== 0] .= 1.0                      # constant columns stay at 0 after centering
    Z = (Xm .- μ) ./ σ

    m = Model(() -> Gurobi.Optimizer(gurobi_env()))
    set_silent(m)
    set_time_limit_sec(m, time_limit)
    @variable(m, β0)
    @variable(m, β[1:p])
    @variable(m, z[1:p], Bin)
    r = y .- β0 .- Z * β
    @objective(m, Min, sum(r .^ 2) / n + λ * sum(β .^ 2))
    @constraint(m, sum(z) <= k)
    @constraint(m, [j = 1:p], !z[j] => {β[j] == 0})

    t = @elapsed optimize!(m)
    status = termination_status(m)
    has_values(m) || error("Best subset solve for k=$k returned no solution (status $status)")

    selected = findall(value.(z) .> 0.5)
    info = Dict{String,Any}(
        "status" => string(status),
        "objective" => objective_value(m),
        "mip_gap" => relative_gap(m),
        "solve_seconds" => t,
        "coefficients_standardized" => Dict("x$j" => value(β[j]) for j in selected),
    )
    return selected, info
end
