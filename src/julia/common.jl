# Shared data loading, metrics, CV folds, and output helpers for the Julia models.
# NRB = 1 (positive), RB = 0.
#
# Note: CSV.jl from the global environment conflicts with the IAI system image's bundled
# Parsers.jl, so data is read with the DelimitedFiles stdlib instead.

using DataFrames, DelimitedFiles, Printf, Random, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(ROOT, "data")
const METRIC_KEYS = ["accuracy", "nrb_recall", "rb_specificity", "balanced_accuracy", "auc"]

# ---------------------------------------------------------------- data

"Split CSVs are ';'-separated, no header: 41 descriptors then the RB/NRB label."
function load_split(name)
    d = readdlm(joinpath(DATA_DIR, "biodeg_$name.csv"), ';')
    X = DataFrame(Float64.(d[:, 1:end-1]), ["x$i" for i in 1:size(d, 2)-1])
    labels = String.(strip.(string.(d[:, end])))
    all(in(("NRB", "RB")), labels) || error("Unexpected labels in $name: $(setdiff(labels, ["NRB", "RB"]))")
    return X, labels
end

function load_all()
    splits = Dict(name => load_split(name) for name in ("train", "val", "test"))
    share(y) = round(mean(y .== "NRB"), digits = 3)
    println(join(["$n $(size(splits[n][1]))" for n in ("train", "val", "test")], ", "),
            "; NRB share ", join(["$n=$(share(splits[n][2]))" for n in ("train", "val", "test")], " "),
            "; threads=$(Threads.nthreads())")
    return splits
end

"Non-stratified K-fold: shuffle row indices with a seeded RNG, then cut into contiguous folds."
function kfold_indices(n, k, seed)
    perm = randperm(MersenneTwister(seed), n)
    sizes = fill(n ÷ k, k)
    sizes[1:(n % k)] .+= 1
    stops = cumsum(sizes)
    starts = [1; stops[1:end-1] .+ 1]
    return [perm[starts[f]:stops[f]] for f in 1:k]
end

# ---------------------------------------------------------------- metrics

"AUC via the Mann-Whitney rank statistic (average ranks for ties)."
function auc(y::AbstractVector{Int}, score::AbstractVector{<:Real})
    n = length(score)
    order = sortperm(score)
    ranks = zeros(n)
    i = 1
    while i <= n
        j = i
        while j < n && score[order[j+1]] == score[order[i]]
            j += 1
        end
        ranks[order[i:j]] .= (i + j) / 2
        i = j + 1
    end
    npos, nneg = sum(y .== 1), sum(y .== 0)
    return (sum(ranks[y .== 1]) - npos * (npos + 1) / 2) / (npos * nneg)
end

function metrics(labels, score; threshold = 0.5)
    y = Int.(labels .== "NRB")
    pred = Int.(score .>= threshold)
    tp = sum((y .== 1) .& (pred .== 1)); fn = sum((y .== 1) .& (pred .== 0))
    tn = sum((y .== 0) .& (pred .== 0)); fp = sum((y .== 0) .& (pred .== 1))
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return Dict{String,Any}(
        "accuracy" => (tp + tn) / length(y),
        "nrb_recall" => recall,
        "rb_specificity" => spec,
        "balanced_accuracy" => (recall + spec) / 2,
        "auc" => auc(y, score),
        "confusion_matrix" => Dict("tn" => tn, "fp" => fp, "fn" => fn, "tp" => tp),
    )
end

"Mean and (population) std of each metric across folds, keyed cv_<metric>_mean / _std."
function summarize_folds(fold_metrics)
    row = Dict{String,Any}()
    for k in METRIC_KEYS
        vals = [m[k] for m in fold_metrics]
        row["cv_$(k)_mean"] = mean(vals)
        row["cv_$(k)_std"] = std(vals, corrected = false)
    end
    return row
end

cv_metric_cols() = vcat([["cv_$(k)_mean", "cv_$(k)_std"] for k in METRIC_KEYS]...)

function print_metrics(title, m)
    println("\n", title)
    for k in METRIC_KEYS
        @printf("  %-18s %.4f\n", k, m[k])
    end
    cm = m["confusion_matrix"]
    println("  confusion (rows=true RB/NRB, cols=pred RB/NRB): ",
            "[[$(cm["tn"]), $(cm["fp"])], [$(cm["fn"]), $(cm["tp"])]]")
end

# ---------------------------------------------------------------- IAI helpers

score_nrb(lnr, X) = Vector{Float64}(IAI.predict_proba(lnr, X)[!, "NRB"])

# ---------------------------------------------------------------- output helpers

json_str(x::AbstractString) = "\"" * escape_string(x) * "\""
json_str(x::Real) = isfinite(x) ? string(x) : "null"
json_str(x::Symbol) = json_str(string(x))
json_str(x::AbstractVector) = "[" * join(json_str.(x), ", ") * "]"
json_str(x::Dict) = "{" * join(["$(json_str(string(k))): $(json_str(v))" for (k, v) in sort(collect(x), by = first)], ", ") * "}"

write_json(path, obj) = write(path, json_str(obj) * "\n")

function write_predictions(path, score)
    open(path, "w") do io
        println(io, "row_id,score_nrb,predicted_label")
        for (i, s) in enumerate(score)   # row_id = 0-based row position within the split CSV
            println(io, i - 1, ",", s, ",", Int(s >= 0.5))
        end
    end
end

function write_csv(path, df::DataFrame)
    open(path, "w") do io
        println(io, join(names(df), ","))
        for r in eachrow(df)
            println(io, join([r[c] isa AbstractString ? "\"$(r[c])\"" : r[c] for c in names(df)], ","))
        end
    end
end

"Score val and test, print metrics, write predictions_{split}.csv; returns Dict split => metrics."
function evaluate_and_export(model_name, predict_score, splits, out_dir)
    mkpath(out_dir)
    results = Dict{String,Any}()
    for name in ("val", "test")
        X, y = splits[name]
        s = predict_score(X)
        results[name] = metrics(y, s)
        print_metrics("$(uppercase(name)) ($model_name fit on train)", results[name])
        write_predictions(joinpath(out_dir, "predictions_$name.csv"), s)
    end
    return results
end
