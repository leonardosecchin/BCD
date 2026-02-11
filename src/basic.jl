"""
Structure for iteration information.

## Fields

- `iter    :: Int64`: number of iterations
- `status  :: Int64`: exit flag (type `?bcd` for details)
- `x       :: Vector{Float64}`: final iterate
- `f       :: Float64`: objective value at the final iterate
- `sig     :: Float64`: final penalization parameter
- `opt     :: Float64`: maximum supnorm over all partial gradients
- `nf      :: Int64`: number of evaluations of the objective function
- `ng      :: Int64`: number of evaluations of the partial gradients
- `nB      :: Int64`: number of evaluations of the approximate Hessian
"""
mutable struct IterInfo
    iter    ::Int64
    status  ::Int64
    x       ::Vector{Float64}
    f       ::Float64
    sig     ::Float64
    opt     ::Float64
    nf      ::Int64
    ng      ::Int64
    nB      ::Int64
end

mutable struct Param
    eps       ::Float64
    alpha     ::Float64
    theta     ::Float64
    maxit     ::Int64
    sig0      ::Float64
    fest      ::Float64
    maxsig    ::Float64
    maxfnoimpr::Int64
end

"""
Returns a `Param` structure with default values.
"""
function default_params()
    return Param(1e-4, 1e-4, 1.0, 10000, 1.0, -Inf, 1e+20, 0)
end

# convert a Vector{Int64} into a UnitRange if possible
function consec_range(v::Vector{Int64})
    if isempty(v)
        return v
    else
        sort!(v)
        @inbounds if v[end] - v[1] + 1 == length(v)
            return v[1]:v[end]
        else
            return v
        end
    end
end

# eigenvalue decomposition of a 2x2 symmetric, nondiagonal,
# matrix D = A[k:k+1,k:k+1]
function eigfac!(d, Q, A, k)
    # A = [a b]
    #     [b c]
    @inbounds a, b, c = A[k,k], A[k,k+1], A[k+1,k+1]

    sub = k:(k+1)

    # eigenvalues
    t = (a + c)/2
    s = sqrt(((a - c)/2.0)^2 + b^2)
    λ = max(0.0, t - s)
    d[sub] .= [λ; max(0.0, t + s)]

    # matrix of eigenvectors
    v1 = [λ - c; b]
    normalize!(v1, 2)
    Q[sub,k] .= v1
    Q[sub,k+1] .= [-v1[2]; v1[1]]
end
