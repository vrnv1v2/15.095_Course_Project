# Best subset selection → OCT-H for QSAR biodegradation. NRB = 1 (positive), RB = 0.
#
# 1. For each k in K_VALUES, pick the best k features with a mixed-integer best subset
#    selection model (see best_subset.jl), then fit OCT-H on only those features.
# 2. Tune k, max_depth, and cp with non-stratified K-fold CV on the training split.
#    Feature selection is re-run inside every fold on that fold's training rows only,
#    so the held-out fold never influences which features are chosen.
# 3. Refit the best setting on the full training split (selection + tree), report
#    validation metrics, and evaluate once on the test split.
#
# Usage (from the repo root, with the IAI system image installed via juliaup):
#     julia +1.11.6/IAI/3.2.2 -t auto src/julia/oct_h_best_subset.jl [folds] [seed] [metric]

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "best_subset.jl"))

const OUT_DIR = joinpath(ROOT, "outputs", "oct_h_best_subset")

const N_FOLDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const SEED = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 42
const METRIC = length(ARGS) >= 3 ? ARGS[3] : "balanced_accuracy"
METRIC in METRIC_KEYS || error("metric must be one of $METRIC_KEYS")

const K_VALUES = [3, 5, 10]
const GRID = (
    max_depth = [1, 2, 3],
    cp = [0.001, 0.005, 0.01, 0.03],
)
const SPARSITY = :all          # hyperplanes may use all k selected features
const MINBUCKET = 5
const RIDGE_λ = 1e-2

make_model(depth, cp) = IAI.OptimalTreeClassifier(
    random_seed = SEED,
    max_depth = depth,
    cp = cp,
    minbucket = MINBUCKET,
    hyperplane_config = (sparsity = SPARSITY,),
)

function main()
    splits = load_all()
    X_tr, y_tr = splits["train"]
    p = ncol(X_tr)

    folds = kfold_indices(nrow(X_tr), N_FOLDS, SEED)
    all_idx = 1:nrow(X_tr)

    # Feature selection doesn't depend on depth/cp, so solve once per (k, fold) and reuse.
    println("\nBest subset selection per fold:")
    fold_sel = Dict{Tuple{Int,Int},Vector{Int}}()
    for k in K_VALUES, (f, te_idx) in enumerate(folds)
        tr_idx = setdiff(all_idx, te_idx)
        sel, info = best_subset(X_tr[tr_idx, :], y_tr[tr_idx], k; λ = RIDGE_λ)
        fold_sel[(k, f)] = sel
        @printf("  k=%-2d fold %d: %-40s %s gap=%.2g (%.1fs)\n", k, f, join("x" .* string.(sel), ","),
                info["status"], info["mip_gap"], info["solve_seconds"])
    end

    println("\nCV over k × max_depth × cp:")
    rows = DataFrame()
    for k in K_VALUES, depth in GRID.max_depth, cp in GRID.cp
        t = @elapsed begin
            fold_metrics = map(enumerate(folds)) do (f, te_idx)
                tr_idx = setdiff(all_idx, te_idx)
                cols = fold_sel[(k, f)]
                lnr = make_model(depth, cp)
                IAI.fit!(lnr, X_tr[tr_idx, cols], y_tr[tr_idx])
                metrics(y_tr[te_idx], score_nrb(lnr, X_tr[te_idx, cols]))
            end
        end
        row = merge(Dict{String,Any}("k" => k, "max_depth" => depth, "cp" => cp),
                    summarize_folds(fold_metrics))
        push!(rows, row; cols = :union)
        @printf("  k=%-2d depth=%d cp=%.3f  cv_%s=%.4f ± %.4f  (%.0fs)\n", k, depth, cp, METRIC,
                row["cv_$(METRIC)_mean"], row["cv_$(METRIC)_std"], t)
    end
    cv_cols = ["k"; "max_depth"; "cp"; cv_metric_cols()]
    rows = sort(rows, "cv_$(METRIC)_mean", rev = true)[:, cv_cols]

    println("\n$N_FOLDS-fold CV on train ($(nrow(rows)) settings), top 5 by $METRIC:")
    show(first(rows[:, ["k", "max_depth", "cp", "cv_$(METRIC)_mean", "cv_$(METRIC)_std"]], 5),
         allrows = true, summary = false, eltypes = false)
    println("\n\nBest setting per k:")
    show(combine(groupby(rows, "k"), first)[:, ["k", "max_depth", "cp", "cv_$(METRIC)_mean", "cv_$(METRIC)_std"]],
         allrows = true, summary = false, eltypes = false)
    println()

    # Final selections on the full training split, for every k (reported) and the best k (used).
    full_sel = Dict{Int,Vector{Int}}()
    full_info = Dict{Int,Dict{String,Any}}()
    for k in K_VALUES
        full_sel[k], full_info[k] = best_subset(X_tr, y_tr, k; λ = RIDGE_λ)
    end

    sel_rows = DataFrame(feature = String[], k = Int[], selected_full_train = Bool[], fold_selection_count = Int[])
    for k in K_VALUES, j in 1:p
        cnt = count(f -> j in fold_sel[(k, f)], 1:N_FOLDS)
        (cnt > 0 || j in full_sel[k]) && push!(sel_rows, ("x$j", k, j in full_sel[k], cnt))
    end
    sort!(sel_rows, ["k", order("fold_selection_count", rev = true), "feature"])

    best_k, best_depth, best_cp = rows[1, "k"], rows[1, "max_depth"], rows[1, "cp"]
    cols = full_sel[best_k]
    println("\nBest params: k=$best_k, max_depth=$best_depth, cp=$best_cp, sparsity=$SPARSITY, minbucket=$MINBUCKET")
    println("Selected features on full train (k=$best_k): ", join("x" .* string.(cols), ", "))

    lnr = make_model(best_depth, best_cp)
    IAI.fit!(lnr, X_tr[:, cols], y_tr)

    results = Dict{String,Any}(
        "best_params" => Dict("k" => best_k, "max_depth" => best_depth, "cp" => best_cp,
                              "sparsity" => string(SPARSITY), "minbucket" => MINBUCKET,
                              "ridge_lambda" => RIDGE_λ),
        "selected_features" => "x" .* string.(cols),
        "best_subset_full_train" => Dict(string(k) => merge(full_info[k],
                                         Dict{String,Any}("features" => "x" .* string.(full_sel[k])))
                                         for k in K_VALUES),
        "cv_folds" => N_FOLDS, "cv_metric" => METRIC,
        "cv_best" => Dict(c => rows[1, c] for c in cv_metric_cols()),
        "num_nodes" => IAI.get_num_nodes(lnr),
    )
    merge!(results, evaluate_and_export("best subset + OCT-H", X -> score_nrb(lnr, X[:, cols]),
                                        splits, OUT_DIR))

    write_csv(joinpath(OUT_DIR, "cv_results.csv"), rows)
    write_csv(joinpath(OUT_DIR, "selected_features.csv"), sel_rows)
    write_json(joinpath(OUT_DIR, "metrics.json"), results)
    IAI.write_html(joinpath(OUT_DIR, "tree.html"), lnr)

    println("\nFinal tree ($(IAI.get_num_nodes(lnr)) nodes):")
    println(lnr)
    println("\nWrote outputs to outputs/oct_h_best_subset/ (tree.html has the interactive tree)")
end

main()
