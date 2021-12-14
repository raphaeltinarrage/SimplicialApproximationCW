# SimplicialApproximationCW

Numerical experiments relative to the paper *Simplicial approximation to CW complexes in practice*.

A notebook can be visualized by clicking on *Demo.ipynb*.

Moreover, the file *G24_1691.txt* represents a simplicial complex, potentially homotopy equivalent to the Grassmannian of 2-planes in R^4.
It is a list of lists of integers. In can be open with pickle via:
```pyton
    import pickle
    with open(G24_1691.txt, "rb") as fp: 
        SimplicialComplex = pickle.load(fp)
```
