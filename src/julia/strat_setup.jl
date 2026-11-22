# Shared setup for the stratified-split models (RF, XGBoost, kernel logistic, OCT-H).
#
# Reproduces exactly what MLProjectPart1_Orig.ipynb (Parts 1, 4, 5) does, and saves it so
# every model — Python or Julia — uses identical rows, folds, and descriptor lists:
#   * development set = biodeg_train_strat + biodeg_val_strat (844 rows), test = biodeg_test_strat
#   * stratified 5-fold CV folds, identical descriptor rows kept in one fold, FOLD_SEED = 2026
#   * k = 3, 5, 10 descriptor lists from IAI OptimalFeatureSelectionRegressor (SEED = 1)
#     on the standardized development set, plus "all" 41 descriptors
#
# Writes outputs/strat/cv_folds.csv and outputs/strat/feature_sets.csv.
#
# Usage (from the repo root):
#     julia +1.11.6/IAI/3.2.2 src/julia/strat_setup.jl

include(joinpath(@__DIR__, "strat_common.jl"))

function main()
    dev, test = load_strat()
    X_dev, ys_dev = dev[:, DESCRIPTORS], dev.class
    y_dev = Int.(ys_dev .== "NRB")

    # ---- Part 4: stratified, duplicate-aware folds (verbatim algorithm from the notebook)
    row_key = [Tuple(r) for r in eachrow(X_dev)]
    id_of   = Dict{Any,Int}()
    grp     = [get!(id_of, k, length(id_of) + 1) for k in row_key]

    fold = zeros(Int, nrow(dev))
    rngf = MersenneTwister(FOLD_SEED)
    for c in (0, 1)
        gids  = unique(grp[y_dev .== c]); shuffle!(rngf, gids)
        gsize = Dict(g => count(==(g), grp) for g in gids)
        sort!(gids, by = g -> -gsize[g], alg = MergeSort)       # pairs first, then singles
        load = zeros(Int, N_FOLDS)
        for g in gids
            f = argmin(load); fold[grp .== g] .= f; load[f] += gsize[g]
        end
    end
    @assert all(length(unique(fold[grp .== g])) == 1 for g in unique(grp))
    for f in 1:N_FOLDS
        idx = fold .== f
        println("fold $f: ", sum(idx), " chemicals, NRB ", round(100 * mean(y_dev[idx]), digits = 1), "%")
    end

    # ---- Part 5: sparse descriptor lists
    mu_d = [mean(X_dev[!, c]) for c in DESCRIPTORS]
    sd_d = [max(std(X_dev[!, c]), 1e-12) for c in DESCRIPTORS]
    Z_dev = DataFrame((Matrix{Float64}(X_dev) .- mu_d') ./ sd_d', DESCRIPTORS)

    feature_sets = Dict{String,Vector{String}}("all" => DESCRIPTORS)
    for k in KS
        ofs = IAI.OptimalFeatureSelectionRegressor(sparsity = k, random_seed = SEED)
        IAI.fit!(ofs, Z_dev, Float64.(y_dev))
        w = Dict(String(f) => b for (f, b) in IAI.get_prediction_weights(ofs)[1])
        feature_sets["k$k"] = [c for c in DESCRIPTORS if get(w, c, 0.0) != 0]
        println(rpad("k = $k", 8), join(feature_sets["k$k"], ", "))
    end

    mkpath(STRAT_OUT)
    write_csv(joinpath(STRAT_OUT, "cv_folds.csv"),
              DataFrame(dev_row = 1:nrow(dev), class = ys_dev, fold = fold))
    fs_tbl = DataFrame(set = String[], order = Int[], descriptor = String[])
    for s in SETS, (i, c) in enumerate(feature_sets[s])
        push!(fs_tbl, (s, i, c))
    end
    write_csv(joinpath(STRAT_OUT, "feature_sets.csv"), fs_tbl)
    println("\nWrote outputs/strat/cv_folds.csv and outputs/strat/feature_sets.csv")
end

main()
