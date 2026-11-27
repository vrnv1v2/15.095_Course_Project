# =============================================================================
# common.jl — shared setup for every notebook:   include("common.jl")
#
# Data loading, settings, metrics, folds and descriptor lists (written by
# notebook 01), the generic cross-validation loop, the shared output format,
# and the plot style. Change settings HERE so all notebooks stay identical.
# =============================================================================
using CSV, DataFrames, Statistics, LinearAlgebra, Random

# ---- Paths ------------------------------------------------------------------
DATA_DIR = "data"                    # folder with the stratified split files (from DataPrep_ML.ipynb / data_split.py)
DATA_EXT = ".xls"                    # the split files are semicolon-separated text saved as biodeg_*_strat.xls
OUT_DIR  = "outputs"                 # model outputs (CSV) shared between notebooks
FIG_DIR  = "figures"
mkpath(OUT_DIR); mkpath(FIG_DIR)
isfile(joinpath(DATA_DIR, "biodeg_train_strat$(DATA_EXT)")) ||
    error("Cannot find biodeg_train_strat$(DATA_EXT) in '$(abspath(DATA_DIR))'. Notebook is running from '$(pwd())'. " *
          "Run DataPrep_ML.ipynb or change DATA_DIR at the top of common.jl.")

# ---- Settings shared by all notebooks ---------------------------------------
SEED, FOLD_SEED, N_FOLDS = 1, 2026, 5
KS        = [3, 5, 10]                 # descriptor budgets (plus all 41)
SELECTORS = ["reg", "log"]             # reg = best-subset least squares, log = sparse logistic
SEL_NAME  = Dict("reg" => "regression selector", "log" => "logistic selector", "none" => "all descriptors")
SETS      = vcat([["reg_k$k", "log_k$k"] for k in KS]..., ["all"])
NRB_RECALL_TARGET = 0.90

set_k(s)    = s == "all" ? 41 : parse(Int, split(s, "_k")[2])
set_sel(s)  = s == "all" ? "none" : String(split(s, "_")[1])
setlabel(s) = s == "all" ? "all 41" : "k=$(set_k(s)) $(set_sel(s))."

DESCRIPTORS = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-",
    "C%", "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L",
    "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)", "nCIR",
    "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d", "B04[C-Br]",
    "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon", "SpMax_B(m)",
    "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX",
]

# ---- Data -------------------------------------------------------------------
function read_part(name)
    d = CSV.read(joinpath(DATA_DIR, "biodeg_$(name)_strat$(DATA_EXT)"), DataFrame; delim = ';', header = false)
    @assert ncol(d) == 42 "biodeg_$(name)_strat$(DATA_EXT) has $(ncol(d)) columns"
    rename!(d, vcat(DESCRIPTORS, "class"))
    d.class = String.(strip.(string.(d.class)))
    d
end
to_labels(y) = ifelse.(y .== 1, "NRB", "RB")

# Development set = train rows then validation rows (dev_row 1..844 in this order); test kept apart.
function load_data()
    dev, test = vcat(read_part("train"), read_part("val")), read_part("test")
    (dev = dev, test = test, X_dev = dev[:, DESCRIPTORS], ys_dev = dev.class, y_dev = Int.(dev.class .== "NRB"),
     X_te = test[:, DESCRIPTORS], y_te = Int.(test.class .== "NRB"))
end

# ---- Folds and descriptor lists (written by notebook 01) ---------------------
load_folds() = CSV.read(joinpath(OUT_DIR, "folds.csv"), DataFrame).fold

# feature_sets.csv: set, fold (0 = final list on all 844), order, descriptor
function load_feature_sets()
    fsdf = CSV.read(joinpath(OUT_DIR, "feature_sets.csv"), DataFrame)
    fold_feats  = Dict{Tuple{String,Int},Vector{String}}()
    final_feats = Dict{String,Vector{String}}("all" => DESCRIPTORS)
    for g in groupby(sort(fsdf, [:set, :fold, :order]), [:set, :fold])
        s, f, d = String(g.set[1]), g.fold[1], String.(g.descriptor)
        f == 0 ? (final_feats[s] = d) : (fold_feats[(s, f)] = d)
    end
    for f in 1:N_FOLDS
        fold_feats[("all", f)] = DESCRIPTORS
    end
    fold_feats, final_feats
end

# Everything a model notebook needs, in one named tuple.
function setup()
    D = load_data()
    fold = load_folds()
    @assert length(fold) == nrow(D.dev) "folds.csv does not match the development set; rerun notebook 01"
    fold_feats, final_feats = load_feature_sets()
    (; D..., fold = fold, fold_feats = fold_feats, final_feats = final_feats)
end

# ---- Metrics ----------------------------------------------------------------
function metrics(p, y; t = 0.5)
    pred = p .>= t
    np, nn = sum(y .== 1), sum(y .== 0)
    rec  = np > 0 ? sum(pred .& (y .== 1)) / np : NaN
    spec = nn > 0 ? sum(.!pred .& (y .== 0)) / nn : NaN
    (recall = rec, specificity = spec, bal_acc = (rec + spec) / 2,
     nrb_missed = sum(.!pred .& (y .== 1)), rb_cleared = sum(.!pred .& (y .== 0)))
end
function best_threshold(p, y)                      # balanced operating point
    cands = sort(unique(vcat(p, 0.5)))
    cands[argmax([metrics(p, y; t = t).bal_acc for t in cands])]
end
function recall_threshold(p, y, target)            # NRB-first operating point
    for t in sort(unique(p), rev = true)
        metrics(p, y; t = t).recall >= target && return t
    end
    minimum(p)
end
roc_points(p, y) = (ts = sort(unique(p), rev = true);
    (vcat(0.0, [mean(p[y .== 0] .>= t) for t in ts], 1.0), vcat(0.0, [mean(p[y .== 1] .>= t) for t in ts], 1.0)))
function auc(p, y)
    pos, neg = p[y .== 1], p[y .== 0]
    sum((a > b) + 0.5 * (a == b) for a in pos, b in neg) / (length(pos) * length(neg))
end

# ---- Standardization (always fitted on the training part only) ---------------
standardize_fit(X, fs) = (M = Matrix{Float64}(X[:, fs]); (vec(mean(M, dims = 1)), vec(max.(std(M, dims = 1), 1e-12))))
zs(X, fs, mu, sd) = (Matrix{Float64}(X[:, fs]) .- mu') ./ sd'

# ---- Generic cross-validation for non-tree models -----------------------------
# For each setting in `grid`: fit on 4 folds (with that fold's descriptor list),
# predict the 5th, average fold AUCs. Best setting is refit on all 844 with the
# final list; its test-set scores are saved (not evaluated) for notebook 04.
function cv_generic(C, s, grid, fitfun, predfun)
    t0 = time()
    res = []
    for prm in grid
        oof, aucs = zeros(length(C.y_dev)), Float64[]
        for f in 1:N_FOLDS
            tr, va = C.fold .!= f, C.fold .== f
            m = fitfun(C.X_dev[tr, :], C.y_dev[tr], C.fold_feats[(s, f)], prm)
            oof[va] = predfun(m, C.X_dev[va, :])
            push!(aucs, auc(oof[va], C.y_dev[va]))
        end
        push!(res, (prm = prm, cv_auc = mean(aucs), cv_sd = std(aucs), oof = oof))
    end
    ch = res[argmax([r.cv_auc for r in res])]
    final = fitfun(C.X_dev, C.y_dev, C.final_feats[s], ch.prm)
    (settings = join(["$k=$v" for (k, v) in pairs(ch.prm)], ", "), cv_auc = ch.cv_auc, cv_sd = ch.cv_sd,
     oof = ch.oof, test = predfun(final, C.X_te), model = final, dcurve = nothing, time = time() - t0)
end

# ---- Shared output format (see README) ----------------------------------------
#   outputs/cv_<model>.csv   one row per set: settings, size, CV AUC, s.d., time
#   outputs/oof_<model>.csv  dev_row, fold, set, model, score  (out-of-fold P(NRB))
#   outputs/test_<model>.csv test_row, set, model, score       (P(NRB) on test; evaluated only in 04)
function new_store()
    (cv = DataFrame(set = String[], model = String[], settings = String[], leaves = Union{Int,Missing}[],
                    descriptors_used = Int[], max_per_split = Union{Int,Missing}[],
                    cv_auc = Float64[], cv_auc_sd = Float64[], time_s = Float64[]),
     oof = DataFrame(dev_row = Int[], fold = Int[], set = String[], model = String[], score = Float64[]),
     test = DataFrame(test_row = Int[], set = String[], model = String[], score = Float64[]),
     depth = DataFrame())
end
function record!(st, C, m, s, r; leaves = missing, used = length(C.final_feats[s]), maxps = missing)
    push!(st.cv, (s, m, r.settings, leaves, used, maxps, r.cv_auc, r.cv_sd, r.time))
    n, nt = length(r.oof), length(r.test)
    append!(st.oof, DataFrame(dev_row = 1:n, fold = C.fold, set = fill(s, n), model = fill(m, n), score = r.oof))
    append!(st.test, DataFrame(test_row = 1:nt, set = fill(s, nt), model = fill(m, nt), score = r.test))
    if r.dcurve !== nothing
        dc = copy(r.dcurve); dc[!, :set] .= s; dc[!, :model] .= m
        append!(st.depth, dc; cols = :union)
    end
end
model_tag(m) = replace(lowercase(m), r"[^a-z0-9]+" => "_")
function save_store(st, m)
    t = model_tag(m)
    CSV.write(joinpath(OUT_DIR, "cv_$t.csv"), st.cv)
    CSV.write(joinpath(OUT_DIR, "oof_$t.csv"), st.oof)
    CSV.write(joinpath(OUT_DIR, "test_$t.csv"), st.test)
    nrow(st.depth) > 0 && CSV.write(joinpath(OUT_DIR, "depth_$t.csv"), st.depth)
    println("Saved outputs for $m → $(OUT_DIR)/*_$t.csv")
end

# ---- Plot style (same in every notebook) --------------------------------------
using Plots          # install once with:  import Pkg; Pkg.add("Plots")
gr()
default(fontfamily = "sans-serif", titlefontsize = 10, guidefontsize = 9, tickfontsize = 8,
        legendfontsize = 8, gridalpha = 0.15, framestyle = :axes, dpi = 150)

READABLE = ["OCT", "CART", "OCT-H", "Logistic", "Additive kernel"]
FLEXIBLE = ["Kernel logistic", "Nadaraya-Watson", "Random forest", "XGBoost"]
GRAY = "#898781"
POS_COLOR, NEG_COLOR, MID_COLOR = "#d6402a", "#2a78d6", "#f0efec"
STYLE = Dict(
    "OCT"             => (color = "#2a78d6", marker = :circle),
    "CART"            => (color = "#eb6834", marker = :rect),
    "Logistic"        => (color = "#1baf7a", marker = :diamond),
    "OCT-H"           => (color = "#eda100", marker = :utriangle),
    "Additive kernel" => (color = "#e87ba4", marker = :star5),
    "Kernel logistic" => (color = GRAY, marker = :diamond),
    "Nadaraya-Watson" => (color = GRAY, marker = :utriangle),
    "Random forest"   => (color = GRAY, marker = :circle),
    "XGBoost"         => (color = GRAY, marker = :rect))
isflex(m) = m in FLEXIBLE
# filled marker in the model's color for readable models, hollow gray for flexible ones
mkw(m) = (markershape = STYLE[m].marker, markercolor = isflex(m) ? :white : STYLE[m].color,
          markerstrokecolor = STYLE[m].color, markerstrokewidth = 1.6, color = STYLE[m].color)
savefig_(fig, name) = (savefig(fig, joinpath(FIG_DIR, name)); fig)

println("common.jl loaded: $(length(SETS)) feature sets, outputs in '$(OUT_DIR)', figures in '$(FIG_DIR)'.")