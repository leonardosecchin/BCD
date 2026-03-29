using Plots
using Format
using BenchmarkProfiles

# run_id: 0 cyclic, 1 metis cyclic

# LATEX TABLE
# Formats
fmt_d = generate_formatter("%'d")
fmt_lf = generate_formatter("%6.2lf")
fmt_e = generate_formatter("%8.2e")
fmt_etex(v) = replace(fmt_e(v), "e+" => "e\$+\$")

function lplsq_table(; run_id = 0, target_ni = 100)
    results = jld2_read("results.jld2", "results")
    if isnothing(results)
        return
    end
    results = results[(results.run_id .== run_id) .& (results.target_ni .== target_ni),:]

    tex = open("lp-lsq_results.tex", "w")
    write(tex, "\\begin{tabular}{lcccrrr}\n\\toprule\n")
    write(tex, "Name & \$(m,n)\$ & \$q\$ & iter & \$f\$ & opt & time (s)\\\\ \\midrule\n")
    for r in eachrow(results)
        name = replace(basename(string(r.instance)), "_" => "\\_")
        it = (r.gsupn > 1e-3) ? "\\it " : ""
        write(tex, "\\texttt{$(name)} & ($(fmt_d(r.size[1])); $(fmt_d(r.size[2]))) & $(fmt_d(r.nblocks)) & $(it)$(fmt_d(r.iter)) & $(it)$(fmt_etex(r.f)) & $(it)$(fmt_etex(r.gsupn)) & $(it)$(fmt_lf(r.time)) \\\\ \n")
    end
    write(tex, "\\bottomrule\n\\end{tabular}")
    close(tex)
end

# PERFORMANCE PROFILES
function pp_blk(; target_ni = 100, runs = [0;1], p = 1.5)
    results = jld2_read("results.jld2","results")
    results = results[(results.target_ni .== target_ni) .& (results.p .== p),:]

    results[results.st .!= 0,:iter] .= -1

    algs = Dict(
        0 => "Cyclic",
        1 => "Cyclic w/ Metis"
    )

    labels = String[]
    iters = []
    for r in runs
        if isempty(iters)
            iters = Float64.(results[results.run_id .== r,:iter])
        else
            iters = hcat(iters, Float64.(results[results.run_id .== r,:iter]))
        end
        push!(labels, algs[r])
    end
    iters[iters .< 0] .= Inf

    fig = performance_profile(PlotsBackend(), iters, labels, title = "Outer iterations")
    Plots.savefig(fig, "pp_blk_iter.pdf")
end

function pp_S()
    results = jld2_read("results.jld2","results")
    results = results[(results.run_id .== 0),:]

    results[results.st .!= 0,:time] .= Inf
    results[results.st .!= 0,:iter] .= -1

    S = sort(unique(results[:,:target_ni]))
    times = []
    iters = []
    for ni in S
        if isempty(times)
            times = results[results.target_ni .== ni,:time]
            iters = Float64.(results[results.target_ni .== ni,:iter])
        else
            times = hcat(times, results[results.target_ni .== ni,:time])
            iters = hcat(iters, Float64.(results[results.target_ni .== ni,:iter]))
        end
    end
    iters[iters .< 0] .= Inf

    fig = performance_profile(PlotsBackend(), times, string.(S), title = "CPU time")
    Plots.savefig(fig, "pp_S_time.pdf")

    fig = performance_profile(PlotsBackend(), iters, string.(S), title = "Outer iterations")
    Plots.savefig(fig, "pp_S_iter.pdf")
end