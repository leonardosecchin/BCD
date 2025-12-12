# standard strategies for switching to cubic regularization
cubic_dyn(si, opts, blocks::Vector{Block}, curr_id, iter::IterInfo, par::Param) =
    (opts[curr_id] <= sqrt(par.eps))
