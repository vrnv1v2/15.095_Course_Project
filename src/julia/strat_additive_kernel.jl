# Sparse additive Gaussian-kernel logistic regression on the stratified split.
#
# Model: one 1-D Gaussian kernel per descriptor, switched on/off by binaries s_j,
#
#     K(s) = Σ_j s_j K_j,   (K_j)_{ab} = exp(-γ (z_aj − z_bj)²),   Σ_j s_j ≤ k,  s ∈ {0,1}^41
#
# on standardized descriptors z, with the kernel logistic regression loss
#
#     c(s) = min_{α,b}  Σ_i [log(1 + e^{η_i}) − y_i η_i] + (λ/2) αᵀ K(s) α,   η = K(s) α + b.
#
# c(s) is convex in s (its dual is a max over β of functions linear in s), and at the optimum
# λα = y − p, so  ∂c/∂s_j = −(λ/2) αᵀ K_j α.  We attack  min_{s} c(s)  s.t. Σ s ≤ k  with outer
# approximation (Bertsimas & Van Parys cutting planes): a Gurobi master problem over the 41
# binaries collects the cuts  η ≥ c(s_t) + ∇c(s_t)ᵀ(s − s_t), giving a valid lower bound.
# The cuts converge slowly for this problem, so OA is capped and followed by a swap local search
# (see "subset search" below); the remaining optimality gap is logged for every solve.
#
# Unlike the notebook's fixed descriptor lists, the model picks its own k descriptors, and the
# selection is redone inside every CV fold (the held-out fold never influences it).
#
# Protocol otherwise matches MLProjectPart1_Orig.ipynb: development set = train_strat + val_strat
# (844), the notebook's stratified folds, settings (γ, λ) chosen by mean CV AUC, balanced and
# NRB-first thresholds from out-of-fold scores, final model refit on all 844; test only with
# --run-test. "all" = all 41 kernels on (no selection), for reference.
#
# Usage (from the repo root; run strat_setup.jl first):
#     julia +1.11.6/IAI/3.2.2 src/julia/strat_additive_kernel.jl [--run-test]

include(joinpath(@__DIR__, "strat_common.jl"))
using JuMP, Gurobi

const MODEL = "Additive kernel"
const OUT_DIR = joinpath(STRAT_OUT, "additive_kernel")
const RUN_TEST = "--run-test" in ARGS

const K_GRID = KS                       # 3, 5, 10 (plus "all" = no constraint)
const GAMMAS = [0.3, 1.0, 3.0]          # 1-D bandwidth on standardized descriptors
const LAMBDAS = [0.1, 1.0, 10.0]          # 0.01 was worst for every k in a first run
const OA_TOL = 1e-4                     # relative gap to stop outer approximation
const OA_MAXIT = 50                     # cuts per solve before handing over to the swap search
const SWAP_CANDIDATES = 6               # unselected descriptors tried per swap pass

const GENV = Gurobi.Env()

softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

# ---------------------------------------------------------------- kernels

function standardize(Xtr::Matrix{Float64}, Xother::Matrix{Float64}...)
    mu = mean(Xtr, dims = 1)
    sd = max.(std(Xtr, dims = 1), 1e-12)
    ((Xtr .- mu) ./ sd, [(X .- mu) ./ sd for X in Xother]...)
end

"n × n × p array of 1-D Gaussian kernels, one per descriptor."
function feature_kernels(Z, γ)
    n, p = size(Z)
    Ks = Array{Float64}(undef, n, n, p)
    Threads.@threads for j in 1:p
        z = Z[:, j]
        Ks[:, :, j] .= exp.(-γ .* (z .- z') .^ 2)
    end
    Ks
end

combine_kernels(Ks, sel) = dropdims(sum(view(Ks, :, :, sel), dims = 3), dims = 3)

cross_kernel(Zq, Ztr, γ, sel) = sum(exp.(-γ .* (Zq[:, j] .- Ztr[:, j]') .^ 2) for j in sel)

# ---------------------------------------------------------------- kernel logistic fit (Newton / IRLS)

function klr_objective(K, α, b, y, λ)
    η = K * α .+ b
    sum(softplus.(η) .- y .* η) + 0.5λ * dot(α, K * α)
end

"Fit kernel logistic regression for a fixed kernel matrix; returns (α, b, objective)."
function fit_klr(K, y, λ; maxit = 100, tol = 1e-10)
    n = length(y)
    α, b = zeros(n), 0.0
    obj = klr_objective(K, α, b, y, λ)
    A = Matrix{Float64}(undef, n + 1, n + 1)
    for _ in 1:maxit
        η = K * α .+ b
        p = 1 ./ (1 .+ exp.(-η))
        w = max.(p .* (1 .- p), 1e-10)
        rhs = w .* η .+ (y .- p)
        A[1:n, 1:n] .= w .* K
        for i in 1:n
            A[i, i] += λ
        end
        A[1:n, n+1] .= w
        A[n+1, 1:n] .= vec(w' * K)
        A[n+1, n+1] = sum(w)
        sol = A \ vcat(rhs, sum(rhs))
        dα, db = sol[1:n] .- α, sol[n+1] - b
        step, new_obj = 1.0, obj
        while true
            new_obj = klr_objective(K, α .+ step .* dα, b + step * db, y, λ)
            (new_obj <= obj + 1e-12 || step < 1e-6) && break
            step /= 2
        end
        α .+= step .* dα; b += step * db
        done = obj - new_obj < tol * max(1.0, abs(obj))
        obj = new_obj
        done && break
    end
    α, b, obj
end

"c(s) and its gradient ∂c/∂s_j = −(λ/2) αᵀ K_j α."
function cost_and_grad(Ks, y, s, λ)
    sel = findall(s .> 0.5)
    α, b, c = fit_klr(combine_kernels(Ks, sel), y, λ)
    p = size(Ks, 3)
    g = zeros(p)
    Threads.@threads for j in 1:p
        g[j] = -0.5λ * dot(α, view(Ks, :, :, j) * α)
    end
    c, g, α, b
end

# ---------------------------------------------------------------- subset search
#
# Outer approximation alone converges slowly here: the linear cuts give a weak lower bound when
# the kernel term dominates (gaps of 30–50% after 300 cuts in testing). So each solve runs
#   1. OA for OA_MAXIT cuts (explores subsets and gives a valid lower bound LB), then
#   2. a best-improvement swap search from the OA incumbent and from the gradient start,
# and keeps the subset with the lowest c(s). The reported gap (c − LB)/c is an honest
# optimality certificate: the subset is good but not proven optimal.

"Best-improvement swap search: swap a selected descriptor for one of the m unselected ones with
the most negative gradient, until no swap lowers c(s)."
function swap_search(Ks, y, sel, λ; m = SWAP_CANDIDATES)
    p = size(Ks, 3)
    sel = copy(sel)
    s = zeros(p); s[sel] .= 1
    c, g, α, b = cost_and_grad(Ks, y, s, λ)
    while true
        cands = filter(j -> !(j in sel), sortperm(g))[1:min(m, p - length(sel))]
        best = (c = c, i = 0, j = 0)
        for i in eachindex(sel), j in cands
            t = copy(sel); t[i] = j
            ct = fit_klr(combine_kernels(Ks, t), y, λ)[3]
            ct < best.c - 1e-9 && (best = (c = ct, i = i, j = j))
        end
        best.i == 0 && return sort(sel), c, α, b
        sel[best.i] = best.j
        s = zeros(p); s[sel] .= 1
        c, g, α, b = cost_and_grad(Ks, y, s, λ)
    end
end

"Best ≤ k descriptors for kernel logistic regression; returns (selected indices, α, b, info)."
function select_subset(Ks, y, k, λ)
    sel_oa, _, _, info = oa_select(Ks, y, k, λ)
    sel_a, c_a, α_a, b_a = swap_search(Ks, y, sel_oa, λ)
    sel_b, c_b, α_b, b_b = swap_search(Ks, y, info.grad_start, λ)
    sel, c, α, b = c_a <= c_b ? (sel_a, c_a, α_a, b_a) : (sel_b, c_b, α_b, b_b)
    c = min(c, info.objective)
    sel, α, b, (iters = info.iters, lower_bound = info.lower_bound, objective = c,
                gap = (c - info.lower_bound) / max(1.0, abs(c)))
end

"Outer approximation (capped at OA_MAXIT cuts); returns (incumbent, α, b, info)."
function oa_select(Ks, y, k, λ)
    p = size(Ks, 3)
    m = Model(() -> Gurobi.Optimizer(GENV))
    set_silent(m)
    @variable(m, s[1:p], Bin)
    @variable(m, η >= 0)
    @constraint(m, sum(s) <= k)
    @objective(m, Min, η)

    # Initial cut at s = 1 (valid: c is convex on [0,1]^p); start from the k largest |gradients|.
    c1, g1, _, _ = cost_and_grad(Ks, y, ones(p), λ)
    @constraint(m, η >= c1 + dot(g1, s .- 1))
    grad_start = sort(sortperm(g1)[1:k])
    st = zeros(p); st[grad_start] .= 1

    UB, LB, best = Inf, 0.0, (s = st, α = Float64[], b = 0.0)
    it = 0
    for outer it in 1:OA_MAXIT
        c, g, α, b = cost_and_grad(Ks, y, st, λ)
        if c < UB
            UB, best = c, (s = copy(st), α = α, b = b)
        end
        @constraint(m, η >= c + dot(g, s .- st))
        optimize!(m)
        LB = objective_value(m)
        (UB - LB) <= OA_TOL * max(1.0, abs(UB)) && break
        st = round.(value.(s))
    end
    sel = findall(best.s .> 0.5)
    sel, best.α, best.b, (iters = it, lower_bound = LB, objective = UB, grad_start = grad_start)
end

# ---------------------------------------------------------------- driver

predict(Zq, Ztr, γ, sel, α, b) = 1 ./ (1 .+ exp.(-(cross_kernel(Zq, Ztr, γ, sel) * α .+ b)))

function main()
    dev, test = load_strat()
    fold, _ = load_folds_and_sets(dev)
    X_dev = Matrix{Float64}(dev[:, DESCRIPTORS]); y_dev = Int.(dev.class .== "NRB")
    X_te = Matrix{Float64}(test[:, DESCRIPTORS]); y_te = Int.(test.class .== "NRB")
    p = length(DESCRIPTORS)
    set_k(s) = s == "all" ? p : parse(Int, s[2:end])
    println("threads=$(Threads.nthreads()), RUN_TEST=$RUN_TEST\n")

    # ---- cross-validation: oof scores and per-fold selections for every (set, γ, λ)
    settings = [(s, γ, λ) for s in SETS for γ in GAMMAS for λ in LAMBDAS]
    oof = Dict(st => zeros(length(y_dev)) for st in settings)
    fold_auc = Dict(st => Float64[] for st in settings)
    fold_sel = Dict{Tuple{String,Float64,Float64,Int},Vector{Int}}()
    oa_log = DataFrame()

    t0 = time()
    for f in 1:N_FOLDS
        tr, va = fold .!= f, fold .== f
        Ztr, Zva = standardize(X_dev[tr, :], X_dev[va, :])
        ytr = Float64.(y_dev[tr])
        for γ in GAMMAS
            Ks = feature_kernels(Ztr, γ)
            for s in SETS, λ in LAMBDAS
                if s == "all"
                    sel = collect(1:p)
                    α, b, _ = fit_klr(combine_kernels(Ks, sel), ytr, λ)
                else
                    sel, α, b, info = select_subset(Ks, ytr, set_k(s), λ)
                    push!(oa_log, (fold = f, set = s, gamma = γ, lambda = λ, iters = info.iters,
                                   gap = info.gap, selected = join(DESCRIPTORS[sel], " | ")))
                end
                pv = predict(Zva, Ztr, γ, sel, α, b)
                oof[(s, γ, λ)][va] = pv
                push!(fold_auc[(s, γ, λ)], auc(pv, y_dev[va]))
                fold_sel[(s, γ, λ, f)] = sel
            end
            @printf("  fold %d γ=%.1f done (%.0f s elapsed)\n", f, γ, time() - t0)
        end
    end

    grid = DataFrame(set = String[], gamma = Float64[], lambda = Float64[], cv_auc = Float64[], cv_auc_sd = Float64[])
    for (s, γ, λ) in settings
        push!(grid, (s, γ, λ, mean(fold_auc[(s, γ, λ)]), std(fold_auc[(s, γ, λ)])))
    end

    # ---- per set: pick (γ, λ) by mean CV AUC, refit selection + model on all 844
    mkpath(OUT_DIR)
    Zdev, Zte = standardize(X_dev, X_te)
    results, tests = DataFrame(), DataFrame()
    sel_tbl = DataFrame(set = String[], fold = String[], descriptors = String[])
    oof_tbl = DataFrame(dev_row = 1:length(y_dev), class = dev.class, fold = fold)
    for s in SETS
        gs = grid[grid.set .== s, :]
        ch = gs[argmax(gs.cv_auc), :]
        γ, λ = ch.gamma, ch.lambda
        t1 = time()
        Ks = feature_kernels(Zdev, γ)
        if s == "all"
            sel = collect(1:p)
            α, b, _ = fit_klr(combine_kernels(Ks, sel), Float64.(y_dev), λ)
            info = (iters = 0, gap = 0.0)
        else
            sel, α, b, info = select_subset(Ks, Float64.(y_dev), set_k(s), λ)
        end
        Ks = nothing; GC.gc()
        for f in 1:N_FOLDS
            push!(sel_tbl, (s, string(f), join(DESCRIPTORS[fold_sel[(s, γ, λ, f)]], " | ")))
        end
        push!(sel_tbl, (s, "final (all 844)", join(DESCRIPTORS[sel], " | ")))

        r_oof = oof[(s, γ, λ)]
        settings_str = "gamma=$γ, lambda=$λ" * (s == "all" ? "" : ", selected: " * join(DESCRIPTORS[sel], ", "))
        row = result_row(s, MODEL, settings_str, missing, length(sel), ch.cv_auc, ch.cv_auc_sd,
                         r_oof, y_dev, time() - t1)
        push!(results, row; cols = :union)
        oof_tbl[!, "score_nrb_$s"] = r_oof
        @printf("%-5s CV AUC %.3f ± %.3f | γ=%.1f λ=%.2f | final search: %d OA cuts, gap %.2f | %s\n",
                s, ch.cv_auc, ch.cv_auc_sd, γ, λ, info.iters, info.gap, join(DESCRIPTORS[sel], ", "))

        if RUN_TEST
            pt = predict(Zte, Zdev, γ, sel, α, b)
            push!(tests, test_row(row, pt, y_te); cols = :union)
            write_csv(joinpath(OUT_DIR, "predictions_test_$s.csv"),
                      DataFrame(row_id = 0:length(pt)-1, score_nrb = pt,
                                predicted_label = Int.(pt .>= row["threshold"])))
        end
    end

    write_csv(joinpath(OUT_DIR, "cv_results.csv"), results[:, RESULT_COLS])
    write_csv(joinpath(OUT_DIR, "cv_grid.csv"), sort(grid, [:set, order(:cv_auc, rev = true)]))
    write_csv(joinpath(OUT_DIR, "selected_features.csv"), sel_tbl)
    write_csv(joinpath(OUT_DIR, "oa_log.csv"), oa_log)
    write_csv(joinpath(OUT_DIR, "oof_predictions.csv"), oof_tbl)

    println("\nCV summary (out-of-fold, 844 chemicals):")
    show(select(results, "set", "cv_auc", "cv_auc_sd", "cv_bal_acc", "threshold", "thr_prec",
                "prec_rb_cleared", "descriptors_used"), eltypes = false, summary = false)
    @printf("\n\nSubset search over all CV solves: median OA cuts %d, max %d; optimality gap median %.2f, max %.2f\n",
            round(Int, median(oa_log.iters)), maximum(oa_log.iters), median(oa_log.gap), maximum(oa_log.gap))
    if RUN_TEST
        write_csv(joinpath(OUT_DIR, "test_results.csv"), tests[:, TEST_COLS])
        println("\nTest (frozen thresholds):")
        show(tests[:, TEST_COLS], eltypes = false, summary = false, allcols = true)
        println()
    else
        println("Test evaluation is switched off (pass --run-test).")
    end
    println("\nWrote outputs to outputs/strat/additive_kernel/")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
