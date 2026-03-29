function jld2_read(filename, objectname)
    output = nothing
    if isfile(filename)
        jld2file = jldopen(filename, "r")
        output = read(jld2file, objectname)
        close(jld2file)
    end
    return output
end
