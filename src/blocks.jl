"""
Structure for store block information.

## Fields
- `ni       :: Int64`: number of variables
- `idx      :: Vector{Cint}`: variable indices
"""
struct Block
    ni      ::Int64
    idx     ::Vector{Cint}
end

"""
    blocks = create_blocks(nblocks, idx)

Returns a vector of `Block` structure given the number of blocks `nblocks`
and a vector of block indices for each variable `idx`. For better efficiency,
place the indexes in `idx` in non-descending order if possible.

`Block` structure is immutable, so its properties can not be changed after its
creation.

## Examples

Creating 3 blocks, the first formed by variables 1, 2 and 3, the second by
variables 4 and 5 and the third by variable 6:

`blocks = create_blocks(3, [1;1;1;2;2;3])`\\
`3-element Vector{Block}:`\\
 `Block(3, Int32[1, 2, 3])`\\
 `Block(2, Int32[4, 5])`\\
 `Block(1, Int32[6])`

Creating `n` blocks, one for each variable:

`n = 10`\\
`blocks = create_blocks(n, 1:n)`
"""
function create_blocks(nblocks, idx)
    @assert nblocks > 0 throw(ArgumentError("Invalid number of blocks"))

    n = 0
    blocks = Block[]
    for i = 1:nblocks
        idxs = consec_range((1:length(idx))[idx .== i])
        @assert length(idxs) > 0 throw("Empty block found")
        n += length(idxs)
        push!(blocks, Block(length(idxs), Cint.(idxs)))
    end
    @assert n == length(idx) throw("Blocks and variables do not match")

    return blocks
end


"""
Cyclical selection of blocks in ascending order.
"""
function blk_cyclic(blocks, curr_id, eligible, opts)
    return mod(curr_id, length(blocks)) + 1
end

function blk_random!()
    perm = []
    counter = 0
    function rand_bid(blocks, curr_id, eligible, opts)
        if curr_id < 0
            counter = 0
            return
        end
        counter = mod(counter, length(blocks)) + 1
        # current cycle ends, restart it
        if counter == 1
            perm = randperm(length(blocks))
        end
        # choose next eligible block in the permutation
        while (counter < length(blocks)) && !eligible[perm[counter]]
            counter += 1
        end
        return perm[counter]
    end
    return rand_bid
end

"""
Random selection of blocks.
"""
blk_random = blk_random!()

function blk_max!()
    visited = []
    function max_bid(blocks, curr_id, eligible, opts)
        if curr_id < 0
            return
        end
        # initialize vector of visited blocks in the current cycle
        if isempty(visited)
            visited = falses(length(blocks))
        end
        # discard non-eligible blocks
        for k in 1:length(blocks)
            if !eligible[k]
                visited[k] = true
            end
        end
        # choose the unvisited block with the largest violation
        p = sortperm(opts, rev = true)
        if all(visited)
            # current cycle ends, restart it
            visited .= false
            bid = 1
        else
            bid = findfirst(.!visited[p])
        end
        visited[p[bid]] = true
        return p[bid]
    end
    return max_bid
end

"""
Select the block with the greatest violation of optimality first.
"""
blk_max = blk_max!()