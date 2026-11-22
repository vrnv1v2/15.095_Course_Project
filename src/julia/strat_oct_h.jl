# OCT-H on the stratified split, following MLProjectPart1_Orig.ipynb's tree protocol exactly
# (same as its `cv_tree`, with hyperplane splits switched on):
#
#   * development set = train_strat + val_strat (844); test = test_strat (211), untouched
#     unless --run-test is passed
#   * folds and k = 3, 5, 10 / all descriptor lists from outputs/strat (run strat_setup.jl first)
#   * per fold: IAI GridSearch over max_depth 2–4 × minbucket 5/10/20 (cp tuned by IAI),
#     scored by AUC on the held-out fold
#   * simplicity rule: among settings within TOL_AUC of the best mean CV AUC, take the
#     shallowest, then the largest minbucket; final cp = median of the per-fold cp values
#   * out-of-fold scores → balanced threshold and NRB-first (90% recall) threshold
#   * final tree refit on all 844 chemicals
#
# Usage (from the repo root):
#     julia +1.11.6/IAI/3.2.2 -t auto src/julia/strat_oct_h.jl [--run-test]
#
# Outputs (outputs/strat/oct_h/): cv_results.csv (notebook columns), cv_depth_curve.csv,
# oof_predictions.csv, tree_<set>.html, and with --run-test: test_results.csv,
# predictions_test_<set>.csv.

include(joinpath(@__DIR__, "strat_common.jl"))

const MODEL = "OCT-H"
const OUT_DIR = joinpath(STRAT_OUT, "oct_h")
const RUN_TEST = "--run-test" in ARGS
const HYPERPLANE = (sparsity = :all,)

tree_leaves(lnr) = count(n -> IAI.is_leaf(lnr, n), 1:IAI.get_num_nodes(lnr))

"Descriptors used anywhere in the tree (hyperplane splits use several; parallel splits one)."
function tree_features(lnr)
    used = String[]
    for n in 1:IAI.get_num_nodes(lnr)
        IAI.is_leaf(lnr, n) && continue
        if IAI.is_hyperplane_split(lnr, n)
            append!(used, [String(f) for (f, w) in IAI.get_split_weights(lnr, n)[1] if w != 0])
        else
            push!(used, String(IAI.get_split_feature(lnr, n)))
        end
    end
    unique(used)
end

octh(; kw...) = IAI.OptimalTreeClassifier(; random_seed = SEED, hyperplane_config = HYPERPLANE, kw...)

function cv_octh(X_dev, ys_dev, y_dev, fold, fs)
    t0 = time()
    per_fold = DataFrame[]
    for f in 1:N_FOLDS
        tr, va = fold .!= f, fold .== f
        grid = IAI.GridSearch(octh(), max_depth = TREE_DEPTHS, minbucket = TREE_MINBUCKETS)
        IAI.fit!(grid, X_dev[tr, fs], ys_dev[tr], X_dev[va, fs], ys_dev[va], validation_criterion = :auc)
        s = select(IAI.get_grid_result_summary(grid), :max_depth, :minbucket, :cp, :valid_score)
        s.fold .= f
        push!(per_fold, s)
    end
    allf = vcat(per_fold...)
    summ = combine(groupby(allf, [:max_depth, :minbucket]),
                   :valid_score => mean => :cv_auc, :valid_score => std => :cv_sd, :cp => median => :cp)
    ok = sort(summ[summ.cv_auc .>= maximum(summ.cv_auc) - TOL_AUC, :],
              [:max_depth, order(:minbucket, rev = true)])
    ch = ok[1, :]

    fold_cp(f, d, mb) = allf[(allf.fold .== f) .& (allf.max_depth .== d) .& (allf.minbucket .== mb), :cp][1]

    oof = zeros(length(y_dev))
    for f in 1:N_FOLDS
        tr, va = fold .!= f, fold .== f
        lnr = octh(max_depth = ch.max_depth, minbucket = ch.minbucket, cp = fold_cp(f, ch.max_depth, ch.minbucket))
        IAI.fit!(lnr, X_dev[tr, fs], ys_dev[tr])
        oof[va] = IAI.predict_proba(lnr, X_dev[va, fs])[!, "NRB"]
    end

    # Depth curve: best leaf size at each depth; refit per fold for leaves and out-of-fold scores
    dcurve = DataFrame(max_depth = Int[], minbucket = Int[], cv_auc = Float64[], cv_sd = Float64[],
                       cv_bal_acc = Float64[], mean_leaves = Float64[], chosen = Bool[])
    for d in sort(unique(summ.max_depth))
        sdd = summ[summ.max_depth .== d, :]
        b = sdd[argmax(sdd.cv_auc), :]
        oofd, lv = zeros(length(y_dev)), Int[]
        for f in 1:N_FOLDS
            tr, va = fold .!= f, fold .== f
            lnr = octh(max_depth = d, minbucket = b.minbucket, cp = fold_cp(f, d, b.minbucket))
            IAI.fit!(lnr, X_dev[tr, fs], ys_dev[tr])
            oofd[va] = IAI.predict_proba(lnr, X_dev[va, fs])[!, "NRB"]
            push!(lv, tree_leaves(lnr))
        end
        push!(dcurve, (d, b.minbucket, b.cv_auc, b.cv_sd,
                       metrics(oofd, y_dev; t = best_threshold(oofd, y_dev)).bal_acc, mean(lv), d == ch.max_depth))
    end

    params = (max_depth = ch.max_depth, minbucket = ch.minbucket, cp = ch.cp)
    final = octh(; params...)
    IAI.fit!(final, X_dev[:, fs], ys_dev)
    (predict = X -> IAI.predict_proba(final, X[:, fs])[!, "NRB"], model = final,
     settings = "max_depth=$(params.max_depth), minbucket=$(params.minbucket), cp=$(round(params.cp, sigdigits = 3)), hyperplane sparsity=all",
     cv_auc = ch.cv_auc, cv_sd = ch.cv_sd, oof = oof, dcurve = dcurve, time = time() - t0)
end

function main()
    dev, test = load_strat()
    fold, feature_sets = load_folds_and_sets(dev)
    X_dev, ys_dev = dev[:, DESCRIPTORS], dev.class
    y_dev = Int.(ys_dev .== "NRB")
    X_te, y_te = test[:, DESCRIPTORS], Int.(test.class .== "NRB")
    println("threads=$(Threads.nthreads()), RUN_TEST=$RUN_TEST\n")

    mkpath(OUT_DIR)
    results, tests = DataFrame(), DataFrame()
    depth_tbl = DataFrame()
    oof_tbl = DataFrame(dev_row = 1:nrow(dev), class = ys_dev, fold = fold)

    for s in SETS
        fs = feature_sets[s]
        r = cv_octh(X_dev, ys_dev, y_dev, fold, fs)
        row = result_row(s, MODEL, r.settings, tree_leaves(r.model), length(tree_features(r.model)),
                         r.cv_auc, r.cv_sd, r.oof, y_dev, r.time)
        push!(results, row; cols = :union)
        oof_tbl[!, "score_nrb_$s"] = r.oof
        dc = copy(r.dcurve); dc.set .= s; dc.model .= MODEL
        append!(depth_tbl, dc; cols = :union)
        IAI.write_html(joinpath(OUT_DIR, "tree_$s.html"), r.model)
        @printf("%-5s %-6s CV AUC %.3f ± %.3f | %s | %d leaves, %d descriptors used | %.1f s\n",
                s, MODEL, r.cv_auc, r.cv_sd, r.settings, tree_leaves(r.model),
                length(tree_features(r.model)), r.time)

        if RUN_TEST
            p = r.predict(X_te)
            push!(tests, test_row(row, p, y_te); cols = :union)
            write_csv(joinpath(OUT_DIR, "predictions_test_$s.csv"),
                      DataFrame(row_id = 0:length(p)-1, score_nrb = p,
                                predicted_label = Int.(p .>= row["threshold"])))
        end
    end

    write_csv(joinpath(OUT_DIR, "cv_results.csv"), results[:, RESULT_COLS])
    write_csv(joinpath(OUT_DIR, "cv_depth_curve.csv"), depth_tbl)
    write_csv(joinpath(OUT_DIR, "oof_predictions.csv"), oof_tbl)

    println("\nCV summary (out-of-fold, 844 chemicals):")
    show(select(results, "set", "cv_auc", "cv_auc_sd", "cv_bal_acc", "threshold", "thr_prec",
                "prec_rb_cleared", "leaves", "descriptors_used"), eltypes = false, summary = false)
    println()
    if RUN_TEST
        write_csv(joinpath(OUT_DIR, "test_results.csv"), tests[:, TEST_COLS])
        println("\nTest (frozen thresholds):")
        show(tests[:, TEST_COLS], eltypes = false, summary = false, allcols = true)
        println()
    else
        println("\nTest evaluation is switched off (pass --run-test).")
    end
    println("\nWrote outputs to outputs/strat/oct_h/")
end

main()
