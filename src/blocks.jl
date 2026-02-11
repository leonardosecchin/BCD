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
function blk_cyclic(blocks, curr_id, elegible, opts)
    return mod(curr_id, length(blocks)) + 1
end

function blk_random!()
    perm = []
    counter = 0
    function rand_bid(blocks, curr_id, elegible, opts)
        if curr_id < 0
            counter = 0
            return
        end
        counter = mod(counter, length(blocks)) + 1
        if counter == 1
            perm = randperm(length(blocks))
        end
        return perm[counter]
    end
    return rand_bid
end

"""
Random selection of blocks, which guarantees that all blocks are visited in
each cycle.
"""
blk_random = blk_random!()
