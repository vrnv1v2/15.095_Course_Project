# OCT-H (Optimal Classification Trees with hyperplane splits) for QSAR biodegradation.
# NRB = 1 (positive), RB = 0.
#
# 1. Tune max_depth and cp with non-stratified K-fold CV on the training split
#    (folds are built here by hand so they are guaranteed non-stratified).
# 2. Refit the best setting on the full training split and report validation metrics.
# 3. Evaluate that finalized tree once on the test split.
#
# Usage (from the repo root, with the IAI system image installed via juliaup):
#     julia +1.11.6/IAI/3.2.2 -t auto src/julia/oct_h.jl [folds] [seed] [metric]
# e.g. julia +1.11.6/IAI/3.2.2 -t auto src/julia/oct_h.jl 5 42 balanced_accuracy

include(joinpath(@__DIR__, "common.jl"))

const OUT_DIR = joinpath(ROOT, "outputs", "oct_h")

const N_FOLDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const SEED = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 42
const METRIC = length(ARGS) >= 3 ? ARGS[3] : "balanced_accuracy"
METRIC in METRIC_KEYS || error("metric must be one of $METRIC_KEYS")

const GRID = (
    max_depth = [1, 2, 3],
    cp = [0.001, 0.005, 0.01, 0.03],
)
const SPARSITY = :all          # hyperplane sparsity; e.g. 2 for at most 2 features per split
const MINBUCKET = 5

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

    folds = kfold_indices(nrow(X_tr), N_FOLDS, SEED)
    all_idx = 1:nrow(X_tr)

    rows = DataFrame()
    for depth in GRID.max_depth, cp in GRID.cp
        t = @elapsed begin
            fold_metrics = map(folds) do te_idx
                tr_idx = setdiff(all_idx, te_idx)
                lnr = make_model(depth, cp)
                IAI.fit!(lnr, X_tr[tr_idx, :], y_tr[tr_idx])
                metrics(y_tr[te_idx], score_nrb(lnr, X_tr[te_idx, :]))
            end
        end
        row = merge(Dict{String,Any}("max_depth" => depth, "cp" => cp), summarize_folds(fold_metrics))
        push!(rows, row; cols = :union)
        @printf("  depth=%d cp=%.3f  cv_%s=%.4f ± %.4f  (%.0fs)\n", depth, cp, METRIC,
                row["cv_$(METRIC)_mean"], row["cv_$(METRIC)_std"], t)
    end
    cv_cols = ["max_depth"; "cp"; cv_metric_cols()]
    rows = sort(rows, "cv_$(METRIC)_mean", rev = true)[:, cv_cols]

    println("\n$N_FOLDS-fold CV on train ($(nrow(rows)) settings), top 5 by $METRIC:")
    show(first(rows[:, ["max_depth", "cp", "cv_$(METRIC)_mean", "cv_$(METRIC)_std"]], 5),
         allrows = true, summary = false, eltypes = false)
    println()

    best_depth, best_cp = rows[1, "max_depth"], rows[1, "cp"]
    println("\nBest params: max_depth=$best_depth, cp=$best_cp, sparsity=$SPARSITY, minbucket=$MINBUCKET")

    lnr = make_model(best_depth, best_cp)
    IAI.fit!(lnr, X_tr, y_tr)

    results = Dict{String,Any}(
        "best_params" => Dict("max_depth" => best_depth, "cp" => best_cp,
                              "sparsity" => string(SPARSITY), "minbucket" => MINBUCKET),
        "cv_folds" => N_FOLDS, "cv_metric" => METRIC,
        "cv_best" => Dict(c => rows[1, c] for c in cv_metric_cols()),
        "num_nodes" => IAI.get_num_nodes(lnr),
    )
    merge!(results, evaluate_and_export("OCT-H", X -> score_nrb(lnr, X), splits, OUT_DIR))

    write_csv(joinpath(OUT_DIR, "cv_results.csv"), rows)
    write_json(joinpath(OUT_DIR, "metrics.json"), results)
    IAI.write_html(joinpath(OUT_DIR, "tree.html"), lnr)

    println("\nFinal tree ($(IAI.get_num_nodes(lnr)) nodes):")
    println(lnr)
    println("\nWrote outputs to outputs/oct_h/ (tree.html has the interactive tree)")
end

main()
