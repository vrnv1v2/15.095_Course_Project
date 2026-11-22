# Shared settings and helpers for the stratified-split pipeline. Constants and helper
# functions mirror MLProjectPart1_Orig.ipynb (Parts 1, 3, 6) so results are comparable
# with the notebook's CART / OCT / logistic regression rows.
#
# Note: CSV.jl from the global environment conflicts with the IAI system image's bundled
# Parsers.jl, so files are read and written with DelimitedFiles / plain IO instead.

using DataFrames, DelimitedFiles, Printf, Random, Statistics, LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(ROOT, "data")
const STRAT_OUT = joinpath(ROOT, "outputs", "strat")

# ---- settings (same values as the notebook's Part 3)
const SEED      = 1
const FOLD_SEED = 2026
const N_FOLDS   = 5
const KS        = [3, 5, 10]
const SETS      = vcat(["k$k" for k in KS], "all")
const TREE_DEPTHS, TREE_MINBUCKETS = 2:4, [5, 10, 20]
const TOL_AUC   = 0.01
const NRB_RECALL_TARGET = 0.90

const DESCRIPTORS = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-",
    "C%", "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L",
    "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)", "nCIR",
    "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d", "B04[C-Br]",
    "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon", "SpMax_B(m)",
    "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX",
]

# ---------------------------------------------------------------- data

"The *_strat.xls files are ';'-separated text (not Excel), no header: 41 descriptors + class."
function read_part(name)
    path = joinpath(DATA_DIR, "biodeg_$(name)_strat.xls")
    d = readdlm(path, ';')
    @assert size(d, 2) == 42 "$path has $(size(d, 2)) columns"
    df = DataFrame(Float64.(d[:, 1:41]), DESCRIPTORS)
    df.class = String.(strip.(string.(d[:, 42])))
    @assert issubset(Set(df.class), Set(["NRB", "RB"]))
    df
end

function load_strat(; verbose = true)
    dev  = vcat(read_part("train"), read_part("val"))
    test = read_part("test")
    if verbose
        for (name, d) in [("development", dev), ("test", test)]
            y = d.class .== "NRB"
            println(rpad(name, 12), length(y), " chemicals | NRB ", sum(y), " (",
                    round(100 * mean(y), digits = 1), "%) | RB ", length(y) - sum(y))
        end
    end
    dev, test
end

"Folds and descriptor lists written by strat_setup.jl."
function load_folds_and_sets(dev)
    f = readdlm(joinpath(STRAT_OUT, "cv_folds.csv"), ',', skipstart = 1)
    @assert size(f, 1) == nrow(dev) "cv_folds.csv doesn't match the development set; rerun strat_setup.jl"
    @assert all(strip.(string.(f[:, 2]), '"') .== dev.class) "cv_folds.csv classes don't match; rerun strat_setup.jl"
    fold = Int.(f[:, 3])
    s = readdlm(joinpath(STRAT_OUT, "feature_sets.csv"), ',', String, skipstart = 1)
    unq(x) = strip(x, '"')
    feature_sets = Dict(set => [unq(s[i, 3]) for i in 1:size(s, 1) if unq(s[i, 1]) == set] for set in SETS)
    fold, feature_sets
end

# ---------------------------------------------------------------- metrics (verbatim from the notebook)

function metrics(p, y; t = 0.5)
    pred = p .>= t
    rec  = sum(pred .& (y .== 1)) / sum(y .== 1)
    spec = sum(.!pred .& (y .== 0)) / sum(y .== 0)
    (acc = mean(pred .== (y .== 1)), recall = rec, specificity = spec, bal_acc = (rec + spec) / 2,
     nrb_missed = sum(.!pred .& (y .== 1)), rb_cleared = sum(.!pred .& (y .== 0)))
end

function best_threshold(p, y)
    cands = sort(unique(vcat(p, 0.5)))
    cands[argmax([metrics(p, y; t = t).bal_acc for t in cands])]
end

# Highest threshold that still catches at least `target` of NRB (most RB cleared at that recall)
function recall_threshold(p, y, target)
    for t in sort(unique(p), rev = true)
        metrics(p, y; t = t).recall >= target && return t
    end
    minimum(p)
end

function auc(p, y)
    pos, neg = p[y .== 1], p[y .== 0]
    sum((a > b) + 0.5 * (a == b) for a in pos, b in neg) / (length(pos) * length(neg))
end

# ---------------------------------------------------------------- output

const RESULT_COLS = ["set", "model", "settings", "leaves", "descriptors_used", "threshold",
                     "cv_auc", "cv_auc_sd", "cv_bal_acc", "cv_recall", "cv_specificity",
                     "cv_nrb_missed", "cv_rb_cleared", "thr_prec", "prec_recall",
                     "prec_specificity", "prec_nrb_missed", "prec_rb_cleared", "time_s"]

const TEST_COLS = ["set", "model", "test_auc", "test_bal_acc", "test_recall", "test_specificity",
                   "nrb_missed", "rb_cleared", "prec_recall", "prec_specificity", "prec_nrb_missed",
                   "cv_auc", "cv_bal_acc"]

csv_cell(x::AbstractString) = occursin(r"[,\"]", x) ? "\"" * replace(x, "\"" => "\"\"") * "\"" : x
csv_cell(::Missing) = ""
csv_cell(x) = string(x)

function write_csv(path, df::DataFrame)
    open(path, "w") do io
        println(io, join(csv_cell.(names(df)), ","))
        for r in eachrow(df)
            println(io, join([csv_cell(r[c]) for c in names(df)], ","))
        end
    end
end

"Notebook-style summary row from out-of-fold scores (both operating points)."
function result_row(set, model, settings, leaves, nused, cv_auc, cv_sd, oof, y, time_s)
    thr = best_threshold(oof, y)
    mt  = metrics(oof, y; t = thr)
    tp  = recall_threshold(oof, y, NRB_RECALL_TARGET)
    mp  = metrics(oof, y; t = tp)
    Dict{String,Any}(
        "set" => set, "model" => model, "settings" => settings, "leaves" => leaves,
        "descriptors_used" => nused, "threshold" => thr, "cv_auc" => cv_auc, "cv_auc_sd" => cv_sd,
        "cv_bal_acc" => mt.bal_acc, "cv_recall" => mt.recall, "cv_specificity" => mt.specificity,
        "cv_nrb_missed" => mt.nrb_missed, "cv_rb_cleared" => mt.rb_cleared,
        "thr_prec" => tp, "prec_recall" => mp.recall, "prec_specificity" => mp.specificity,
        "prec_nrb_missed" => mp.nrb_missed, "prec_rb_cleared" => mp.rb_cleared, "time_s" => time_s)
end

"Notebook Part 9 row: frozen thresholds applied to test scores."
function test_row(r, p, y)
    mt = metrics(p, y; t = r["threshold"])
    mp = metrics(p, y; t = r["thr_prec"])
    Dict{String,Any}(
        "set" => r["set"], "model" => r["model"], "test_auc" => auc(p, y),
        "test_bal_acc" => mt.bal_acc, "test_recall" => mt.recall, "test_specificity" => mt.specificity,
        "nrb_missed" => mt.nrb_missed, "rb_cleared" => mt.rb_cleared,
        "prec_recall" => mp.recall, "prec_specificity" => mp.specificity, "prec_nrb_missed" => mp.nrb_missed,
        "cv_auc" => r["cv_auc"], "cv_bal_acc" => r["cv_bal_acc"])
end
