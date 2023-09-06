# Documentation

```@docs
domath
greet
```

## Types

```@docs
Amlet.LogitUtility
```

## Functions

### Objective 

This is the computation of the objective function 
```math
-\dfrac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{n_{alt}} y_{ij}\log\big( \mathbb{P}(j \vert \beta , X) \big) 
```

```@docs
F
Fs
Fs!
```

### Gradient 
```@docs
grad
grad!
grads
grads!
```

### Hessian
```@docs
H
H! 
Hdotv
Hdotv!
```

### Hessian approxination : BHHH
```@docs
BHHH
BHHH!
BHHHdotv
BHHHdotv!
```