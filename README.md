# BCD

This is a Julia implementation of the Block Coordinate Descent (BCD) method with quadratic and cubic regularizations for minimizing functions with Hölder continuous gradient, as described in

Amaral, Andreani, Secchin, Silva. Flexible block coordinate descent methods for unconstrained optimization under Hölder continuity. 2025

## Installation and use

`]add https://github.com/leonardosecchin/BCD.git`

For usage instructions, type `?bcd` after load the package `BCD`. Cubic regularization needs HSL MA57 working. Please visit <https://github.com/JuliaSmoothOptimizers/HSL.jl> for instructions on how install it. For proper working, load HSL package before the BCD:

```
using HSL
using BCD
```

You can test if HSL is working by running `LIBHSL_isfunctional()`, which should return true.


## Funding

This research was supported by the National Council for Scientific and Technological Development (CNPq) (grant 302520/2025-2), Brazil.


## How to cite

If you use this code in your publications, please cite us. For now, you can cite the preprint:

Amaral, Andreani, Secchin, Silva. Flexible block coordinate descent methods for unconstrained optimization under Hölder continuity. 2025
