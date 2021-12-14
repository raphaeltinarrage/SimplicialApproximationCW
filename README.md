# SimplicialApproximationCW

Numerical experiments relative to the paper *Simplicial approximation to CW complexes in practice*.

A notebook can be visualized by clicking on **Demo.ipynb**.

Moreover, the file **G24.txt** represents a simplicial complex, with 1691 vertices, potentially homotopy equivalent to the Grassmannian of 2-planes in R^4.
It is a list of lists of integers. It can be open with pickle via:
```pyton
    import pickle
    with open(G24.txt, "rb") as fp: 
        SimplicialComplex = pickle.load(fp)
```
