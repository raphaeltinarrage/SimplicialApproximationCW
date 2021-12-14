'''----------------------------------------------------------------------------
LIST OF FUNCTIONS

Misc:
- ChronometerStart
- ChronometerStop

Triangulation:
- ClassTriangulation
- ClassCell
- TriangulateSphere
- TriangulateProjectiveSpace
- GetGluingLocationMap
- RadialProjection
- LocationMapSphereRadialFast
- TriangulatePoint
- LocationMapBallFast
- TriangulateBall
- GlueCell

Combinatorics:
- GeneralizedSubdivision
- CheckWeakStarCondition

Contractions:
- ContractTriangulation
- ContractSimplexTreeFast

Projective Spaces:
- GluingMapProjectiveSpace
- InvCharacteristicMapProjectiveSpace
- DomainProjectiveSpace
- SimplifyDelaunay

Grassmannian:
- TwoBallsToBall
- BallToTwoBalls
- ReducedEchelonForm
- GluingMapGrassmannian
- InvCharacteristicMapGrassmannian
- DomainGrassmannian
----------------------------------------------------------------------------'''

import velour
import gudhi
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import time
from datetime import timedelta
import sys
import pickle
import types
import scipy

'''----------------------------------------------------------------------------
Misc
----------------------------------------------------------------------------'''

def ChronometerStart(msg):
    start_time = time.time()
    sys.stdout.write(msg); sys.stdout.flush()
    return start_time

def ChronometerStop(start_time, method='ms'):
    elapsed_time_secs = time.time() - start_time
    if method == 'ms':
        msg = 'Execution time: '+repr(round(elapsed_time_secs*1000))+' ms.\n'
    if method == 's':
        msg = 'Execution time: '+repr(round(elapsed_time_secs))+' s.\n'
    sys.stdout.write(msg); sys.stdout.flush()
    
'''----------------------------------------------------------------------------
Triangulation
----------------------------------------------------------------------------'''

class ClassTriangulation:
    ''' 
    The ClassTriangulation represents the data of a simplicial complex K and a homeomorphism f: X --> |K|
    where X is a submanifold of R^n.

    It contains:
    - Complex: a gudhi.SimplexTree, representing K.
        It allows a faster computation of the LocationMap.
    - LocationMap: a function, representing the map f. It takes a point x of R^n as an input,
        and returns a simplex of K. 
    - Domain: a function, representing the domain X. It takes a point x of R^n as an input,
        and returns True if x is in X, or False else. 
        
    Optional instances:
    - Coordinates: a dict v:numpy.array, representing the points f^-1(v) for v vertex of K.
    - Tree: a list of dict, representing the subdivision structure of K. 

    And the functions:
    - add_LocationMap: allows to endow the object with a LocationMap. It should be used as follows:
        (1) Define:  def LocationMap(self): return a function of self.Coordinates, self.Complex...
        (2) Use:     add_LocationMap(self, LocationMap)
    - add_Domain: allows to endow the object with a Domain. Same use as before.
    ''' 
    def __init__(self):
        self.Complex = None
        self.LocationMap = None
        self.Domain = None
        self.Coordinates = None
        self.Tree = None
        
    def add_LocationMap(self, LocationMap):
        def LocationMapSelf(x, self=self):
            return LocationMap(x, self)
        self.LocationMap = LocationMapSelf   

    def add_Domain(self, Domain):
        def DomainSelf(x, self=self):
            return Domain(x, self)
        self.Domain = DomainSelf

class ClassCell:
    ''' 
    The ClassCell represents the data of a cell coming from a CW structure on a space X, where 
    X is a submanifold of R^n. The corresponding gluing map is a map phi: S^{d-1} --> X, where 
    S^{d-1} is the unit sphere of $R^d$. The characteristic map is an extension f: D^d --> X 
    of the latter map, where D^d is the unit disc of R^d. The inverse characteristic map is an 
    inverse f^-: X --> D^d.

    It contains:
    - Dimension: an int, representing the dimension d of the cell.
    - Domain: a function, representing the domain E. It takes a point x of R^n as an input,
        and returns True if x is in E, or False else. 
    - GluingMap: a function, representing the gluing map for the cell E.
        It takes a point x of S^(d-1) as an input, seen as a subset of R^d,
        and returns a point of X.
    - InverseCharacteristicMap: a function, representing the inverse of the characteristic map f.
        It takes a point x of E, and returns a point of the disc D^d. 
        
    And the functions:
    - add_Domain: allows to endow the object with a Demain. It should be used as follows:
        (1) Define:  def Domain(self): return a function of self.Dimension, self.SchubertSymbol...
        (2) Use:     add_Domain(self, Domain)
    - add_GluingMap: allows to endow the object with a GluingMap. Same use as before.
    - add_InvCharacteristicMap: allows to endow the object with a InvCharacteristicMap. 
        Same use as before.
    ''' 
    def __init__(self):
        self.Dimension = None
        self.Domain = None
        self.GluingMap = None
        self.InvCharacteristicMap = None
        
    def add_Domain(self, Domain):
        def DomainSelf(x, self=self):
            return Domain(x, self)
        self.Domain = DomainSelf   
        
    def add_GluingMap(self, GluingMap):
        def GluingMapSelf(x, self=self):
            return GluingMap(x, self)
        self.GluingMap = GluingMapSelf   
        
    def add_InvCharacteristicMap(self, InvCharacteristicMap):
        def InvCharacteristicMapSelf(x, self=self):
            return InvCharacteristicMap(x, self)
        self.InvCharacteristicMap = InvCharacteristicMapSelf   
        
def TriangulateSphere(dim = 2, subdivise = 0, method = 'barycentric', verbose = False):
    ''' 
    Gives a triangulation of the unit sphere S^d of R^{d+1}. It has d+2 vertices. 
    The coordinates of the vertices are found by 
    (1) Considering the canonical vector e_1, ..., e_{d+2} of R^{d+2},
    (2) Projecting them into the hyperplane they span,
    (3) Center them around the origin (1,...,1)/(d+2)
    The sphere is then subdivided n times, where n is the parameter 'subdivise', and following 
    the subdivision method given by 'method'.
    The triangulation of the sphere is endowed with the radial projection map.
    
    Input:
    - dim: an int, the dimension d of the sphere.
    - subdivise: an int, the number of subdivisions to apply to the complex.
    - method: can be 'barycentric', 'edgewise', 'Delaunay-barycentric' or 'Delaunay-edgewise'.
        Indicates the method of subdivision.
    - verbose: can be True or False. Whether to print commments.
    
    Output:
    - Sphere_Triangulation: a ClassTriangulation object.
    
    Example:
        Sphere = TriangulateSphere()
        Sphere.LocationMap(np.array([1,0,0]))
        >> [1, 2, 3]
    ''' 
    # Build the simplicial complex
    st_sphere = gudhi.SimplexTree()                  #the vertices of the sphere will be range(dim+2)
    st_sphere.insert(range(dim+2))                   #simplex of dimension d+1    
    st_sphere.remove_maximal_simplex(range(dim+2))   #triangulation of the sphere of dimension d
    
    # Build the coordinates
    origin = np.ones(dim+2)/(dim+2)

    SimplexVectors = {i:np.zeros(dim+2) for i in range(dim+2)} #the canonical vectors of R^(d+2)
    for i in range(dim+2): SimplexVectors[i][i] = 1
    
    AffineVectors = np.zeros((dim+2,dim+1)) #centering the vectors
    for i in range(dim+1): AffineVectors[:,i] = SimplexVectors[i]-origin    
    Q, r = np.linalg.qr(AffineVectors)

    ProjectedVectors = {i:np.dot(Q.T, SimplexVectors[i]-origin[i]) for i in range(dim+2)}
        #projecting the vectors onto the hyperplane they span
        
    CoordinatesSphere = {i:ProjectedVectors[i]/np.linalg.norm(ProjectedVectors[i]) for i in range(dim+2)}
        #renormalize the vectors

    # Create data structure ClassTriangulation
    Sphere = ClassTriangulation()
    Sphere.Complex = st_sphere
    Sphere.Coordinates = CoordinatesSphere
    Sphere.Tree = []
    Sphere.add_LocationMap(LocationMapSphereRadialFast)

    # Subdivise
    for i in range(subdivise): GeneralizedSubdivision(Sphere, method=method, verbose=False)

    if verbose:
        result_str = 'Dimension/Simplices/Vertices = '+repr(Sphere.Complex.dimension())+'/'+repr(Sphere.Complex.num_simplices())+'/'+ repr(Sphere.Complex.num_vertices())+'.\n'
        sys.stdout.write('| TriangulateSphere      | '+result_str)
    return Sphere

def TriangulateProjectiveSpace(dim = 2, verbose = False):
    ''' 
    Gives a triangulation of the projective space of dimension dim. 
    It is obtained by triangulating the sphere via the boundary of the (dim+1)-simplex, subdivising it
    barycentrically once, and quotienting the complex by the antipodal relation.
    
    Input:
    - dim: an int, the dimension d of the projective space.
    - verbose: can be True or False. Whether to print commments.
    
    Output:
    - ProjectiveSpace: a ClassTriangulation object.
    '''     
    # Triangulate sphere
    Sphere = TriangulateSphere(dim=dim, verbose=False)
    Sphere, dict_NewVertices = GeneralizedSubdivision(Sphere, method='barycentric', return_dict_NewVertices = True, verbose=verbose)
        # subdivise once
        
    # Triangulate projective space
    # Represent the equivalence relation Sphere/~ = ProjectiveSpace
    VerticesSimplex = set(range(dim+2))
    dict_NewVertices_inv = {dict_NewVertices[v]:v for v in dict_NewVertices}
    EquivalenceRelation = {}                     
    for t in dict_NewVertices:
        EquivalenceRelation[t] = tuple(VerticesSimplex.difference(t))
        
    # Choose a representative set (we choose the element in the class with the lowest values)
    RepresentativeSet = {}                             
    for s in EquivalenceRelation:
        t = EquivalenceRelation[s]
        S = [s,t]
        smin = min(s)
        tmin = min(t)
        I = [smin, tmin]
        argmin = I.index(min(I))
        RepresentativeSet[s] = S[argmin]  
    RepresentativeSet = {v:dict_NewVertices[RepresentativeSet[dict_NewVertices_inv[v]]] for v in dict_NewVertices.values()}

    st_proj = gudhi.SimplexTree() #triangulation of the projective space
    for filtr in Sphere.Complex.get_filtration():
        simplex = filtr[0]
        simplex_repr = [RepresentativeSet[v] for v in simplex]
        st_proj.insert(simplex_repr)
      
    # Define face map    
    def LocationMapProjectiveSpace(proj, self):
        vect = velour.ProjectionToVector(proj)
        face = self.Parent.LocationMap(vect)
        face_repr = [self.RepresentativeSet[v] for v in face]
        return face_repr
        
    # Create data structure ClassTriangulation
    ProjectiveSpace = ClassTriangulation()
    ProjectiveSpace.Complex = st_proj
    ProjectiveSpace.Parent = Sphere
    ProjectiveSpace.RepresentativeSet = RepresentativeSet
    ProjectiveSpace.add_LocationMap(LocationMapProjectiveSpace)
    
    if verbose:
        result_str = 'Complex of dimension ' + repr(ProjectiveSpace.Complex.dimension()) + ', ' + \
                        repr(ProjectiveSpace.Complex.num_simplices()) + ' simplices, ' + \
                        repr(ProjectiveSpace.Complex.num_vertices()) + ' vertices.\n'
        sys.stdout.write('| TriangulateProjectiveSpace | '+result_str)

    return ProjectiveSpace

def GetGluingLocationMap(tr_K, tr_L, Map, LocationMap = None, verbose = False):
    ''' 
    Compute the concatenation LocationMap \circ Map: |K| --> X --> |L|, where K is given as a 
    ClassTriangulation tr_K, and L as a ClassTriangulation tr_L.
    The map |K| --> X is given by Map, and X --> |L| is the LocationMap of L.
    The parameter LocationMap is a dictionary, giving the points on which the face map has already 
    been computed.
    
    Input:
    - tr_K: a ClassTriangulation, representing K.
    - tr_L: a ClassTriangulation, representing L.
    - Map: a function, taking as an input a point of R^n (vertex of K), and returns a point 
        of R^{n'} (point of X).
    - LocationMap: a dict (vertex of tr_K.Complex):(simplex of tr_L.Complex). Output of a previous 
        usage of the function. Allow a faster computation, by only computing the face map of 
        the new points. /!\ Works only for an identic labelling of the vertices. /!\ 
    - verbose: can be True or False. Whether to print commments.
    
    Output:
    - LocationMap: a dict (vertex of tr_K.Complex):(simplex of tr_L.Complex), representing the map 
        LocationMap \circ Map.
    '''
    if LocationMap == None: LocationMap = dict()
    
    if verbose: 
        val = 100*(len(tr_K.Coordinates)-len(LocationMap))/len(tr_K.Coordinates)
        val_str = float("{0:.1f}". format(val))
        msg = '| GetGluingLocationMap   | '+repr(len(tr_K.Coordinates)-len(LocationMap))+'/'+repr(len(tr_K.Coordinates))+' new vertices ('+repr(val_str)+'%).'
        
    NewVertices = list(set(velour.GetVerticesSimplexTree(tr_K.Complex)).difference(LocationMap.keys()))
    Vertices_len = len(NewVertices)
    if verbose: start_time = time.time()
    for i in range(len(NewVertices)):
        if verbose:
            elapsed_time_secs = time.time() - start_time
            expected_time_secs = (Vertices_len-i-1)/(i+1)*elapsed_time_secs
            msg1 = 'Vertex '+repr(i+1)+'/'+repr(Vertices_len)+'. '
            msg2 = 'Duration %s' % timedelta(seconds=round(elapsed_time_secs))
            msg3 = '/%s.' % timedelta(seconds=round(expected_time_secs))
            sys.stdout.write('\r'+msg+' '+msg1+msg2+msg3)
        j = NewVertices[i]
        point = Map(tr_K.Coordinates[j])        
        LocationMap[j] = tr_L.LocationMap(point)       
    if verbose: sys.stdout.write('\n')

    return LocationMap

def RadialProjection(v, list_vertices):
    '''
    Gives the point of intersection between a line (spanned by v) and a hyperplane 
    (spanned by the points of list_vertices).
    Returns False if they do not intersect. If they do, return True, the barycentric 
    coordinates of the intersection points, and the distance between the point v 
    and the intersection point.    
    If h is a vector orthogonal to this hyperplane, and x_0 any point of this hyperplane, 
    then this intersection point is l*v, where l = <x_0,h>/<v,h>.
    Hence the distance between v and this intersection point is |1-l|*norm(v).
    In order to get the barycentric coordinates of this intersection point, we 
    first compute the linear coordinates of it in the basis given by 
    {list_vertices[i+1] - list_vertices[0], i}.

    Input: 
    - v (np.array): size (1xm).
    - list_vertices (list of np.array): list of length m of (1xm) arrays. 
    
    Output:
    - intersect (bool): True or False, whether they intersect.
    - coord_bar (np.array): size (1xm), the barycentric coordinates.
    - distance (float): distance between the point v and the intersection point.
    '''
    # (1) Find an intersection with the hyperplane
    d = len(list_vertices)-1
    face_linspace = np.zeros((d+1,d))
        #a matrix containing a basis of the corresponding linear subspace, origin being list_vertices[0]
    for i in range(d): face_linspace[:, i] = list_vertices[i+1] - list_vertices[0]
    q,r = np.linalg.qr(face_linspace, mode = 'complete')
        # QR decomposition (remark : r is invertible since the vertices are affinely independant)   
    h = q[:, d]
        #h is a vector orthogonal to the affine hyperplane spanned by list_vertices
    s = np.inner(v, h)
    if s==0:
        intersect = False
        coord_bar = np.NAN
        distance = np.NAN
    else:
        # (2) Compute intersection point
        intersect = True
        l = np.inner(list_vertices[0], h)/s
        distance = np.abs(1 - l)    
        v_intersection = l*v

        # (3) Compute barycentric coordinates
        w = v_intersection-list_vertices[0]         
        q = q[:,0:d]; r = r[0:d,:]
        coord_lin = w.dot(q)
            #coordinates of w in the orthonormal basis given by q
        coord_face = np.linalg.inv(r).dot(coord_lin)
            #coordinates of w in the basis given by face_linspace    
        coord_bar = np.append( 1-sum(coord_face),coord_face)
            #barycentric coordinates of v_intersection
    return intersect, coord_bar, distance

def LocationMapSphereRadialFast(vect, self, epsilon = 1e-10):
    ''' 
    Gives a LocationMap for the sphere, triangulated by the simplicial complex K via proj: S^d --> |K|.
    It takes a point x of S^d (seen in R^{d+1}) as an input, and returns a simplex of K. By definition, 
    LocationMap(x) is the unique simplex sigma of K such that proj(x) is in (the interior of) sigma.

    The computation goes as follows:
    (1) For every maximal face of K, compute the intersection of the line spanned by x and the hyperplane 
        spanned by sigma, in barycentric coordinates. Select the simplices sigma such that x projects 
        inside sigma (i.e., the barycentric coordinates are nonnegative. In practice, we ask them greater 
        than -epsilon).
    (2) Return the simplex sigma for which the distance from x to sigma is minimized. If there are
        several minimizers, return their intersection.

    If self.Tree is not None, a fast computation is used, based on the Tree self.Tree. We first apply the 
    previous process to the simplices of generation 1, then of generation 2, etc...
    
    Input: 
    - vect (np.array) of size (1xm): represent the point to project
    - self (ClassTriangulation): represent the simplicial complex K
    
    Output:
    - minimal_face (list of int): represent the face map of vect 
    '''
    vect = vect/np.linalg.norm(vect) #normalize the vector
    d = self.Complex.dimension()

    # Fast implementation
    if self.Tree is None or self.Tree == []:
        Current_simplices = [filtr[0] for filtr in self.Complex.get_filtration() if len(filtr[0])==d+1]
            #the Current_simplices are all the simplices
        number_repetition_fast = 0
    else:
        Current_simplices = list(self.Tree[0].keys())
            #the Current_simplices are the simplices of first generation in self.Tree
        number_repetition_fast = len(self.Tree)

    for i in range(number_repetition_fast+1):
        # (1) Find simplices such that vect projects on it
        DistancesToFaces = dict()
        for simplex in Current_simplices:
            list_vertices = [self.Coordinates[v] for v in simplex]
            intersect, bar_coords, distance = RadialProjection(vect, list_vertices)
            if intersect and all(bar_coords>=-epsilon): #if vect project inside the simplex
                DistancesToFaces[tuple(simplex)] = distance
        if len(DistancesToFaces)==0: #if vect does not project on any simplex 
            raise ValueError('Problem in LocationMapSphere! No projection. vect = '+repr(vect)+'.')

        # (2) Find minimizers
        minimal_distance = min(DistancesToFaces.values())
        minimal_faces = [simplex for simplex in DistancesToFaces if DistancesToFaces[simplex]==minimal_distance]
            #the minimizers
        if i < number_repetition_fast: #repeat the process
            minimal_face = minimal_faces[0] #arbitrary choice of minimizer
            Current_simplices = self.Tree[i][tuple(sorted(minimal_face))]    
        else:         
            minimal_face = minimal_faces[0] #arbitrary choice of minimizer
#                minimal_face = [v for v in minimal_faces[0] if all([v in simplex for simplex in minimal_faces])]
#                    #other possibility: intersection of the minimizers
    return minimal_face

def TriangulatePoint():
    '''
    Returns a triangulation of the topological spa e consisting of only one point.
    
    Output:
    - tr: a ClassTriangulation() object.
    '''
    tr = ClassTriangulation()
    
    # Add simplicial complex
    tr.Complex = gudhi.SimplexTree()
    tr.Complex.insert([0])
    
    # Add LocationMap
    def Domain(vect, self): #always returns True
        return True
    def LocationMap(x, self): #always returns [0]
        return [0]
    tr.add_LocationMap(LocationMap)
    tr.add_Domain(Domain)
        
    return tr

def LocationMapBallFast(vect, self, epsilon = 1e-10):
    ''' 
    Gives a LocationMap for the ball, triangulated by the simplicial complex K via proj: B^d --> |K|.
    It takes a point x of B^{d+1} (seen in R^{d+1}) as an input, and returns a simplex of K. By 
    definition, LocationMap(x) is the unique simplex sigma of K such that proj(x) is in (the interior of) 
    sigma.

    The computation goes as follows:
    (1) Normalize the vector x, and find its LocationMap on the sphere
    (2) Among all its corresponding simplices, find the ones on which it projects,that is, with
        positive barycentric coordinates. In practice, we ask them >=-epsilon.
    (3) Among the simplices on which it projects, find the one with the smallest barycentric 
        coordinates. If there are several minimizers, return their intersection.
        
    Input:
    - vect: a np.array, representing the point x of B^{d+1}.
    - self: a ClassTriangulation object, representing the triangulation of the ball.
    
    Output:
    - minimal_face: a list of int, representing the simplex that contains 'vect'.
    '''
    d = self.Complex.dimension()

    # Find if vect is at the origin
    if (np.abs(vect) < epsilon).all(): return [self.ConingPoint]

    # (1) Find projection on the sphere
    vect_normalized = vect/np.linalg.norm(vect)
    simplex_sphere = self.Boundary.LocationMap(vect_normalized)

    # (2) Find simplices on which it project
    Simplices = self.CorrespondingSimplices[tuple(sorted(simplex_sphere))]

    IsInSimplex = dict() #will contain simplices on which vect project
    for simplex in Simplices:
        # Compute barycentric coordinates
        R = np.zeros((d+1,d+1))
        for i in range(d+1):
            R[0:d,i] = self.Coordinates[simplex[i]]
            R[d,i] = 1
        V = np.concatenate((vect,[1]))
        coord = np.linalg.inv(R).dot(V) #barycentric coordinates

        # Check if inside simplex. The barycentric coordinates should be nonnegative
        # to ensure that the point is inside the simplex
        if all(coord >= -epsilon): IsInSimplex[tuple(sorted(simplex))] = min(coord)

    # (3) Find minimizer
    if len(IsInSimplex)>0:
        minimal_distance = max(IsInSimplex.values())
        minimal_faces = [simplex for simplex in IsInSimplex if IsInSimplex[simplex]==minimal_distance]
            #the minimizers
        minimal_face = [v for v in minimal_faces[0] if all([v in simplex for simplex in minimal_faces])]
            #intersection of the minimizers
        if len(minimal_face)==0: #if the intersection is empty
            raise ValueError('Problem in LocationMapBallFast! Empty intersection. vect = '+repr(vect)+'.')
        else: return minimal_face

    # If vect is not in the interior of any maximal simplex, it is associated to the (unique) maximal
    # simplex that contains the simplex of the sphere on which vect_normalized projects
    else: return Simplices[0] 

def TriangulateBall(tr_sphere, beta = 1/2, verbose = False):
    ''' 
    Gives a triangulation L of the unit ball B^{d+1} of R^{d+1}, based on a triangulation K of 
    the sphere S^d. It is built as follows:
        (1) Start with the sphere K
        (2) Build the simplicial product K x [0,1]
        (3) Cone the inner part of K x [0,1] to a new vertex x* 
    The resulting simplicial complex has three layers of vertices: the outer sphere (at distance 1
    from the origin), an inner sphere (at distance beta), and a coning point (the origin).
    
    Input:
    - tr_sphere: a ClassTriangulation object, representing the simplicial complex K.
        Must contain at least tr_sphere.Complex and tr_sphere.Coordinates
    - beta: float in (0,1). Parameter to build the coordinates of the inner vertices.
    - verbose: can be True or False. Whether to print commments.
    
    Output:
    - tr_ball: a ClassTriangulation object, representing the simplicial complex L.
    
    Example:
        tr = TriangulateSphere()
        tr = TriangulateBall(tr)
        >> ...
    ''' 
    if verbose: sys.stdout.write('| TriangulateBall        |')
    dim = tr_sphere.Complex.dimension()+1 #dimension of the ball

    # (1) Copy the sphere
    st_cell = velour.CopySimplexTree(tr_sphere.Complex)
    index_cone = max(velour.GetVerticesSimplexTree(tr_sphere.Complex))+1 
        #new starting index for vertices
    dict_intervertices = {i:(i+index_cone+1) for i in tr_sphere.Coordinates}
        #give, for each vertex of tr_sphere, the corresponding inner vertex of tr_ball
    
    # (2) Triangulation product K x [0,1]
    CorrespondingSimplices = dict() 
    MaximalSimplices = (filtr[0] for filtr in tr_sphere.Complex.get_filtration() if len(filtr[0])==dim)
    for simplex in MaximalSimplices:
        CorrespondingSimplices[tuple(simplex)] = []
        simplex_concatenate = simplex + [dict_intervertices[i] for i in simplex] 
            #product set (simplex x [0,1])
        for i in range(dim): #product (simplex x [0,1])
            simplex_hyperprism = simplex_concatenate[i:(i+dim+1)]
            st_cell.insert(simplex_hyperprism)
            CorrespondingSimplices[tuple(simplex)]+=[simplex_hyperprism]
        
    # (3) Coning
    st_cell.insert([index_cone]) #add origin
    MaximalSimplices = (filtr[0] for filtr in tr_sphere.Complex.get_filtration() if len(filtr[0])==dim)
    for simplex in MaximalSimplices:
        simplex_inter = [dict_intervertices[i] for i in simplex]
        simplex_coning = simplex_inter+[index_cone]
        st_cell.insert(simplex_coning)
        CorrespondingSimplices[tuple(simplex)] += [simplex_coning]

    # Get coordinates vertices
    cell_coords = tr_sphere.Coordinates.copy()
    cell_coords[index_cone] = cell_coords[0]*0 #coordinates of origin
    for i in tr_sphere.Coordinates:
        cell_coords[dict_intervertices[i]] = tr_sphere.Coordinates[i]*beta
        
    # Define a ClassTriangulation   
    tr_ball = ClassTriangulation()
    tr_ball.Complex = st_cell
    tr_ball.Coordinates = cell_coords
    tr_ball.Boundary = tr_sphere
    tr_ball.CorrespondingSimplices = CorrespondingSimplices  
    tr_ball.ConingPoint = index_cone
    tr_ball.add_LocationMap(LocationMapBallFast)

    if verbose:
        result_str = 'Dimension/Simplices/Vertices = '+repr(tr_ball.Complex.dimension())+'/'+repr(tr_ball.Complex.num_simplices())+'/'+ repr(tr_ball.Complex.num_vertices())+'.\n'
        sys.stdout.write(' '+result_str)
    return tr_ball

def GlueCell(tr_cell, tr_base, Cell, SimplicialMap, method='homotopy', alpha=2, verbose=False):
    ''' 
    Glue the ClassTriangulation tr_cell to tr_base, according to SimplicialMap.
    The gluing comes from a CW structure, whose current cell is described in the ClassCell Cell.
    The gluing is obtained as follows:
        (1) Define the characteristic map. (dict int-->int). It is a simplicial map from the vertices 
            of tr_cell.Complex to a new set L of vertices, such that it is injective on its interior, 
            and such that, restricted to its boundary, it takes values in the vertices of tr_base.Complex
        (2) Build the triangulation of the gluing
         
    
    Input:
    - tr_cell: a ClassTriangulation object, representing the cell to glue.
        Must come from a gluing procedure.
    - tr_base: a ClassTriangulation object, representing the codomain of the gluing.
        ...
    - Cell: a ClassCell. Describes the current cell of the CW structure. 
    - SimplicialMap:
    - method: can be 'homotopy' or 'direct'. Affects the computation of the LocationMap of the output.
        In order to make the process of gluing cells work, as described in the paper, the method
        has to be 'homotopy'.
    - alpha: float in (1,+inf). Parameter to build the homotopy.
    - verbose: can be True or False. Whether to print commments.
    
    Output:
    - tr_gluing: a ClassTriangulation object, representing the gluing.
    ''' 
    if verbose: sys.stdout.write('| GlueCell               |')
        
    # (1) Define CharacteristicMap
    Vertices_cell = (filtr[0][0] for filtr in tr_cell.Complex.get_skeleton(0))
    dict_vertices_boundary = {v:False for v in Vertices_cell}
    Vertices_boundary = (filtr[0][0] for filtr in tr_cell.Boundary.Complex.get_skeleton(0))
    for v in Vertices_boundary: dict_vertices_boundary[v] = True
        #dict_vertices_boundary takes as an input a vertex v of tr_cell, and is True if and
        #only if v is in its boundary

    new_index_simplex = max(velour.GetVerticesSimplexTree(tr_base.Complex))+1 #new index to add vertices
            
    CharacteristicMap = dict()
    for t in dict_vertices_boundary:
        if dict_vertices_boundary[t]==True: CharacteristicMap[t] = SimplicialMap[t]
            #if the vertex is in the boundary, glue it according to SimplicialMap
        else: CharacteristicMap[t] = t + new_index_simplex
            #if the vertex is in the interior, give it a new index
    
    # (2) Define glued simplicial complex
    st = gudhi.SimplexTree()    
      #(2)-1 Copy st_base
    for filtr in tr_base.Complex.get_filtration(): st.insert(filtr[0])            
      #(2)-2 Insert st_cell according to CharacteristicMap
    dim_cell = tr_cell.Complex.dimension()
    maximal_simplices = (filtr[0] for filtr in tr_cell.Complex.get_filtration() if len(filtr[0])==dim_cell+1)
    for simplex in maximal_simplices: 
        simplex_glued = [CharacteristicMap[i] for i in simplex]
        st.insert(simplex_glued)
        
    # Define LocationMap
    def LocationMapGluingDirect(vect, self):
        ''' 
        Gives a LocationMap for the glued complex.
        '''
        if self.NewCell.Domain(vect):
            x = self.NewCell.InvCharacteristicMap(vect)
            simplex_cell = self.NewTriangulation.LocationMap(x)
            simplex = [self.CharacteristicMap[v] for v in simplex_cell]
            simplex = list(set(simplex))
            simplex.sort()
            return simplex
        else:
            return self.LowerSkeleton.LocationMap(vect)
    
    def LocationMapGluing(vect, self):
        ''' 
        Gives a LocationMap for the glued complex. The glued complex can be seen as the (non-disjoint)
        union of K and L, where L is the base, and K the new cell (whose boundary is glued). 
        We have
        - self.Cell.GluingMap: a function that takes a point of the boundary of the cell as an input,
            and return a point of the base.
        - self.Cell.InvCharacteristicMap: a function that takes a point of the glued complex as an 
            input (the point being inside the corresponding cell), and return a point of the cell.
            It may be noncontinuous on the boundary of the cell.

        Let vect be a point of the CW complex. In order to compute LocationMapGluing, we:
            (1) Check if it is in the domain in the new cell
            (2) If it is, compute its image x in the ball via InvCharacteristicMap
            (3) See whether the norm of x is lower than 1/alpha.
                If it is, x is multiplied by alpha, we compute its LocationMap in the ball, and return
                its image in the glued complex.
                If not, x is normalized, we compute its image in the LowerSkeleton via GluingMap,
                and return its LocationMap in the lower skeleton.
        '''
        # (1) Check if vect is in the domain of the new cell
        if self.NewCell.Domain(vect):
            # (2) Compute its image in the ball
            x = self.NewCell.InvCharacteristicMap(vect)

            # (3)-1 If the point, via the homotopy of the paper, lies inside the ball
            if np.linalg.norm(x)<1/2:
                x = 2*x              
                simplex_cell = self.NewTriangulation.LocationMap(x) #LocationMap of the point in the ball
                simplex = [self.CharacteristicMap[v] for v in simplex_cell] 
                    #image of the simplex in the glued complex
                simplex = sorted(list(set(simplex)))
                return simplex

            # (3)-1 If the point, via the homotopy of the paper, lies in the boundary the ball
            else:
                x = x/np.linalg.norm(x)     #the point is normalized
                xx = self.NewCell.GluingMap(x) #image of the point in the lower skeleton
                simplex = self.LowerSkeleton.LocationMap(xx)  
                return simplex
        else: return self.LowerSkeleton.LocationMap(vect)
        
    # Define a ClassTriangulation
    tr_gluing = ClassTriangulation()
    tr_gluing.Complex = st
    tr_gluing.NewTriangulation = tr_cell
    tr_gluing.NewCell = Cell
    tr_gluing.LowerSkeleton = tr_base
    tr_gluing.CharacteristicMap = CharacteristicMap #dict
    
    if method == 'homotopy': tr_gluing.add_LocationMap(LocationMapGluing)
    if method == 'direct': tr_gluing.add_LocationMap(LocationMapGluingDirect)

    if verbose:
        result_str = 'Dimension/Simplices/Vertices = '+repr(st.dimension())+'/'+repr(st.num_simplices())+'/'+ repr(st.num_vertices())+'.\n'
        sys.stdout.write(' '+result_str)
    
    return tr_gluing

'''----------------------------------------------------------------------------
Combinatorics
----------------------------------------------------------------------------'''

def GeneralizedSubdivision(tr, VerticesToSubdivise = None, method = 'barycentric', normalize = True, define_tree = True, return_dict_NewVertices = False, verbose = False):
    '''
    Subdivise the simplicial complex K, given by the ClassTriangulation tr. It produces 
    a generalized subdivision, whose fixed complex is K\K^-, where K^- is the union of 
    the open stars of the vertices in VerticesToSubdivise. In other words, the subdivided 
    edges are the ones who admit a point in VerticesToSubdivise.
    Four different generalized subdivisions are available:
        - method = 'barycentric'          ---> barycentric subdivision
        - method = 'edgewise'             ---> edgewise subdivision
        - method = 'Delaunay-barycentric' ---> Delaunay barycentric subdivision
        - method = 'Delaunay-edgewise'    ---> Delaunay edgewise subdivision
    /!\ edgewise (non-Delaunay) works only if the dimension of K is lower or equal to 3 /!\
    /!\ Delaunay works only if K is a subset of the sphere /!\
    The new coordinates are computed as barycenters of the corresponding simplices, and then
    normalized if normalize == True. The new points are:
        - if method = 'barycentric' or 'Delaunay-barycentric' ---> the barycenters of all 
            the modified simplices.
        - if method = 'edgewise' or 'Delaunay-edgewise' ---> the barycenters of all the 
            modified edges (i.e. their midpoints).

    Input: 
    - tr: a ClassTriangulation, representing the simplicial complex to subdivise.
    - VerticesToSubdivise: a list of int, representing vertices of K. If it is None, all
        the vertices are subdivised.
    - method: can be 'barycentric', 'edgewise', 'Delaunay-barycentric' or 'Delaunay-edgewise'. 
        The method of subdivision.
    - normalize: can be True or False, whether to normalize the vertices.
    - define_tree: can be True or False, whether to endow the triangulation with the forest.
    - return_dict_NewVertices: can be True or False, whether to return the dictionary describing
        the new vertices as permutations of simplices (only for barycentric).
    - verbose: can be True or False. Whether to print commments.
        
    Output:
    - tr_sub: a ClassTriangulation, representing the subdivised simplicial complex.
    /!\ By running the function, the input tr is modified into tr_sub /!\
    '''
    if verbose: 
        if VerticesToSubdivise is None: sys.stdout.write('| GeneralizedSubdivision | '+method+' Global subdivision. ')
        else: sys.stdout.write('| GeneralizedSubdivision | '+method+'. '+\
                                  repr(len(VerticesToSubdivise))+'/'+repr(tr.Complex.num_vertices())+' points to subdivise. ')
    st = tr.Complex       
    if VerticesToSubdivise is None: VerticesToSubdivise = [filtr[0][0] for filtr in st.get_skeleton(0)]
        #subdivise all
    
    # Get simplices to subdivise (simplices that admit a vertex in PointsToSubdivise)
    Vertices =  (filtr[0][0] for filtr in st.get_skeleton(0))
    dict_VerticesToSubdivise = {v:False for v in Vertices}
    for v in VerticesToSubdivise: dict_VerticesToSubdivise[v] = True

    Simplices = (filtr[0] for filtr in st.get_filtration() if len(filtr[0])>1) 
        #only simplices of positive dimension
    dict_SimplicesToSubdivise = {i:dict() for i in range(1,st.dimension()+1)} 
        #one dict per positive dimension. Equals true is the simplex is modified
    for simplex in Simplices: 
        dict_SimplicesToSubdivise[len(simplex)-1][tuple(simplex)] = any([dict_VerticesToSubdivise[v] for v in simplex])
        #dict_SimplicesToSubdivise[dim(simplex)][simplex] is True if the simplex is to subdivise, and False else.         
            
    # Label new vertices
    Vertices = (filtr[0][0] for filtr in st.get_skeleton(0))
    dict_NewVertices = {tuple([v]):v for v in Vertices} #labeling of the new vertices
        #dict_NewVertices[simplex] is the index of the points corresponding to the new simplex.
    l = max(dict_NewVertices.values())+1 #new index to start
    if method =='barycentric' or method=='Delaunay-barycentric': 
        dimensions_NewVertices = range(1,st.dimension()+1) #subdivise positive-dimensional simplices
    elif method =='edgewise' or method=='Delaunay-edgewise': 
        dimensions_NewVertices = [1] #only subdivise edges

    for d in dimensions_NewVertices:
        for simplex in dict_SimplicesToSubdivise[d]:
            if dict_SimplicesToSubdivise[d][simplex] == True:
                dict_NewVertices[simplex] = l
                l += 1
                
    # Compute coordinates
    NewCoordinates = {dict_NewVertices[v]:np.mean([tr.Coordinates[w] for w in v],0) for v in dict_NewVertices}
    if normalize: NewCoordinates = {v:NewCoordinates[v]/np.linalg.norm(NewCoordinates[v]) for v in NewCoordinates}
        #normalize the coordinates
        
    # Create subdivised complex 
    if method=='Delaunay-barycentric' or method=='Delaunay-edgewise':
        # Compute Delaunay triangulation (given by convex hull of the vertices)        
        points = np.array([NewCoordinates[v] for v in NewCoordinates])
        hull = scipy.spatial.ConvexHull(points)

        # New ordering of the vertices (in case of dict_NewVertices.values() not being the interval, but missing values)
        dict_NewVertices_list = list(dict_NewVertices.values())
        dict_NewVertices_index = {i:dict_NewVertices_list[i] for i in range(len(dict_NewVertices))}
        
        # Create new complex
        st_sub = gudhi.SimplexTree()
        for simplex in hull.simplices: 
            #re-index the vertices of simplex
            simplex_NewNertices = sorted([dict_NewVertices_index[v] for v in simplex])
            st_sub.insert(simplex_NewNertices)
            
        if define_tree == True:
            # Get new tree. Obtained by associating a simplex (in the Delaunay triangulation)
            # to the simplices (of the original triangulation) that share a vertex
            MaximalSimplices = (tuple(filtr[0]) for filtr in st.get_filtration() if len(filtr[0])==st.dimension()+1)
            Vertices = (filtr[0][0] for filtr in st.get_skeleton(0))
            Simplices_by_vertices = {v:list() for v in Vertices}
            for simplex in MaximalSimplices:
                for v in simplex: Simplices_by_vertices[v].append(simplex)
                    #Simplices_by_vertices associates to each vertex v the simplices that contain v

            MaximalSimplices = (tuple(filtr[0]) for filtr in st.get_filtration() if len(filtr[0])==st.dimension()+1)
                #/!\ works only for pure simplicial complex /!\
            NewTree = {simplex:list() for simplex in MaximalSimplices}

            dict_NewVertices_inv = {dict_NewVertices[v]:v for v in dict_NewVertices}
            for simplex in hull.simplices: 
                simplex_NewVertices = sorted([dict_NewVertices_index[v] for v in simplex])
                simplex_OldVertices = [dict_NewVertices_inv[v] for v in simplex_NewVertices]
                simplex_OldVertices = list(set([w for v in simplex_OldVertices for w in v]))

                simplices_parent = [simplex_parent for v in simplex_OldVertices for simplex_parent in Simplices_by_vertices[v]]
                simplices_parent = list(set(simplices_parent))
                for simplex_parent in simplices_parent: NewTree[simplex_parent].append(tuple(simplex_NewVertices))

            # Suppress repetitions
            for simplex in NewTree: NewTree[simplex] = list(set(NewTree[simplex]))
        else: NewTree = None
    
    else: #if not Delaunay
        # Initialize subdivision
        Simplices = (filtr[0] for filtr in st.get_filtration() if len(filtr[0])>1)
        dict_Modifications = {tuple(simplex):list() for simplex in Simplices} 
            #dict_Modifications[simplex] gathers the modification of simplex, i.e.,
            #with what is replaced each simplex.
        Vertices = (filtr[0][0] for filtr in st.get_skeleton(0))
        for v in Vertices: dict_Modifications[tuple([v])] = [tuple([v])]
 
        # Create new complex and insert vertices
        st_sub = gudhi.SimplexTree()
        Vertices = (filtr[0][0] for filtr in st.get_skeleton(0))
        for v in Vertices: st_sub.insert([v])
            
        if method == 'barycentric': 
            # Insert simplices
            for d in range(1,st.dimension()+1):
                for simplex in dict_SimplicesToSubdivise[d]:
                    if dict_SimplicesToSubdivise[d][simplex] == False: #if the simplex is not modified, insert as it is
                        st_sub.insert(simplex)
                        dict_Modifications[simplex] = [simplex]
                    else: #if the simplex is modified, cone its barycenter to its (modified) boundary
                        new_vertex = dict_NewVertices[simplex]
                        Faces = itertools.combinations(simplex, len(simplex)-1)
                        for face in Faces: 
                            for subface in dict_Modifications[face]   :#add the cones
                                simplex_cone = subface+tuple([new_vertex])
                                st_sub.insert(simplex_cone)                  
                                dict_Modifications[simplex].append( tuple(sorted(simplex_cone)) )

        if method == 'edgewise': 
            # Define RepairEdgewiseSubdivision
            RepairEdgewiseSubdivision = {1: {(): [((0,), (1,))], (0,): [((0,), (0, 1)), ((1,), (0, 1))], (1,): [((0,), (0, 1)), ((1,), (0, 1))], (0, 1): [((0,), (0, 1)), ((1,), (0, 1))]}, 2: {(): [((2,), (0,), (1,))], (0,): [((0, 1), (0,), (0, 2)), ((0, 1), (2,), (1,)), ((0, 1), (2,), (0, 2))], (1,): [((0, 1), (1, 2), (1,)), ((0, 1), (2,), (1, 2)), ((0, 1), (2,), (0,))], (2,): [((1, 2), (0, 2), (1,)), ((0,), (0, 2), (1,)), ((1, 2), (2,), (0, 2))], (0, 1): [((0, 1), (1, 2), (1,)), ((0, 1), (0,), (0, 2)), ((1, 2), (2,), (0, 2)), ((0, 1), (0, 2), (1, 2))], (0, 2): [((0, 1), (1, 2), (1,)), ((0, 1), (0,), (0, 2)), ((1, 2), (2,), (0, 2)), ((0, 1), (0, 2), (1, 2))], (1, 2): [((0, 1), (1, 2), (1,)), ((0, 1), (0,), (0, 2)), ((1, 2), (2,), (0, 2)), ((0, 1), (0, 2), (1, 2))], (0, 1, 2): [((0, 1), (1, 2), (1,)), ((0, 1), (0,), (0, 2)), ((1, 2), (2,), (0, 2)), ((0, 1), (0, 2), (1, 2))]}, 3: {(): [((2,), (0,), (3,), (1,))], (0,): [((0, 1), (2,), (3,), (1,)), ((0, 1), (0, 3), (3,), (0, 2)), ((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (2,), (3,), (0, 2))], (1,): [((0, 1), (2,), (3,), (1, 2)), ((0, 1), (2,), (0,), (3,)), ((0, 1), (1, 3), (3,), (1, 2)), ((0, 1), (1, 2), (1, 3), (1,))], (2,): [((1, 2), (2, 3), (0, 2), (3,)), ((0,), (3,), (0, 2), (1,)), ((1, 2), (3,), (0, 2), (1,)), ((1, 2), (2,), (2, 3), (0, 2))], (3,): [((2,), (0, 3), (1, 3), (1,)), ((0, 3), (1, 3), (2, 3), (3,)), ((2,), (0, 3), (1, 3), (2, 3)), ((2,), (0, 3), (0,), (1,))], (0, 1): [((0, 1), (0, 3), (1, 3), (1, 2)), ((1, 2), (0, 3), (3,), (0, 2)), ((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (0, 3), (0, 2), (1, 2)), ((1, 2), (0, 3), (1, 3), (3,)), ((1, 2), (2,), (3,), (0, 2)), ((0, 1), (1, 2), (1, 3), (1,))], (0, 2): [((0, 1), (0, 3), (3,), (1, 2)), ((1, 2), (0, 3), (2, 3), (3,)), ((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (0, 3), (0, 2), (1, 2)), ((1, 2), (0, 3), (2, 3), (0, 2)), ((0, 1), (1, 2), (3,), (1,)), ((1, 2), (2,), (2, 3), (0, 2))], (0, 3): [((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (0, 3), (2, 3), (0, 2)), ((0, 3), (1, 3), (2, 3), (3,)), ((0, 1), (2,), (1, 3), (1,)), ((0, 1), (2,), (2, 3), (0, 2)), ((0, 1), (0, 3), (1, 3), (2, 3)), ((0, 1), (2,), (1, 3), (2, 3))], (1, 2): [((0, 1), (1, 3), (2, 3), (1, 2)), ((0, 1), (2, 3), (0, 2), (3,)), ((0, 1), (1, 3), (2, 3), (3,)), ((0, 1), (0,), (3,), (0, 2)), ((0, 1), (2, 3), (0, 2), (1, 2)), ((0, 1), (1, 2), (1, 3), (1,)), ((1, 2), (2,), (2, 3), (0, 2))], (1, 3): [((0, 1), (0, 3), (1, 3), (1, 2)), ((1, 2), (2,), (0, 3), (2, 3)), ((0, 1), (2,), (0, 3), (0,)), ((0, 3), (1, 3), (2, 3), (3,)), ((0, 1), (2,), (0, 3), (1, 2)), ((0, 1), (1, 2), (1, 3), (1,)), ((1, 2), (0, 3), (1, 3), (2, 3))], (2, 3): [((0, 3), (0,), (0, 2), (1,)), ((1, 2), (0, 3), (2, 3), (0, 2)), ((0, 3), (1, 3), (2, 3), (3,)), ((1, 2), (0, 3), (1, 3), (1,)), ((1, 2), (0, 3), (1, 3), (2, 3)), ((1, 2), (0, 3), (0, 2), (1,)), ((1, 2), (2,), (2, 3), (0, 2))], (1, 2, 3): [((0, 1), (0, 3), (1, 3), (1, 2)), ((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (0, 3), (0, 2), (1, 2)), ((1, 2), (0, 3), (2, 3), (0, 2)), ((0, 3), (1, 3), (2, 3), (3,)), ((0, 1), (1, 2), (1, 3), (1,)), ((1, 2), (0, 3), (1, 3), (2, 3)), ((1, 2), (2,), (2, 3), (0, 2))], (0, 2, 3): [((0, 1), (0, 3), (1, 3), (1, 2)), ((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (0, 3), (0, 2), (1, 2)), ((1, 2), (0, 3), (2, 3), (0, 2)), ((0, 3), (1, 3), (2, 3), (3,)), ((0, 1), (1, 2), (1, 3), (1,)), ((1, 2), (0, 3), (1, 3), (2, 3)), ((1, 2), (2,), (2, 3), (0, 2))], (0, 1, 3): [((0, 1), (0, 3), (1, 3), (1, 2)), ((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (0, 3), (0, 2), (1, 2)), ((1, 2), (0, 3), (2, 3), (0, 2)), ((0, 3), (1, 3), (2, 3), (3,)), ((0, 1), (1, 2), (1, 3), (1,)), ((1, 2), (0, 3), (1, 3), (2, 3)), ((1, 2), (2,), (2, 3), (0, 2))], (0, 1, 2): [((0, 1), (0, 3), (1, 3), (1, 2)), ((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (0, 3), (0, 2), (1, 2)), ((1, 2), (0, 3), (2, 3), (0, 2)), ((0, 3), (1, 3), (2, 3), (3,)), ((0, 1), (1, 2), (1, 3), (1,)), ((1, 2), (0, 3), (1, 3), (2, 3)), ((1, 2), (2,), (2, 3), (0, 2))], (0, 1, 2, 3): [((0, 1), (0, 3), (1, 3), (1, 2)), ((0, 1), (0, 3), (0,), (0, 2)), ((0, 1), (0, 3), (0, 2), (1, 2)), ((1, 2), (0, 3), (2, 3), (0, 2)), ((0, 3), (1, 3), (2, 3), (3,)), ((0, 1), (1, 2), (1, 3), (1,)), ((1, 2), (0, 3), (1, 3), (2, 3)), ((1, 2), (2,), (2, 3), (0, 2))]}}
            
            # Insert simplices (insert midpoint edges, and repair triangles and tetrahedra)
            for d in range(1,st.dimension()+1):
                for simplex in dict_SimplicesToSubdivise[d]:
                    subdivised_vertices = [v for v in simplex if dict_VerticesToSubdivise[v]==True]
                        #the vertices in VerticesToSubdivise and simplex

                    # Relabel the vertices in [0,1,2,...]
                    dict_relabel = {simplex[i]:i for i in range(len(simplex))}
                    subdivised_vertices_relabel = [dict_relabel[v] for v in subdivised_vertices]
                        #the vertices of subdivised_vertices, now indexed in [0,1,2,...]

                    # Get reparation of the triangle
                    simplices_repair = RepairEdgewiseSubdivision[d][tuple(subdivised_vertices_relabel)] 
                        #reparations, i.e., simplices to fill the triangle  

                    # Relabel back to X
                    dict_relabel_inv = {dict_relabel[v]:v for v in dict_relabel}
                    simplices_repair_relabel = [ tuple( [ dict_NewVertices[tuple([dict_relabel_inv[v] for v in vertex])] for vertex in simplex] ) for simplex in simplices_repair]

                    # Insert the simplices
                    for simplex_repair in simplices_repair_relabel: 
                        st_sub.insert(simplex_repair)       
                        dict_Modifications[simplex].append( tuple(sorted(simplex_repair)) )

        # Get new tree
        MaximalSimplices = [tuple(filtr[0]) for filtr in st.get_filtration() if len(filtr[0])==st.dimension()+1]
            #/!\ works only for pure simplicial complex /!\
        NewTree = {simplex:dict_Modifications[simplex] for simplex in MaximalSimplices}

    # Modify the a ClassTriangulation tr
    tr.Complex = st_sub
    tr.Coordinates = NewCoordinates
    if NewTree is not None and tr.Tree is not None: tr.Tree += [NewTree]
    
    if verbose:
        result_str = 'Dim/Simp/Vert = '+repr(st_sub.dimension())+'/'+repr(st_sub.num_simplices())+'/'+ repr(st_sub.num_vertices())+'.\n'
        sys.stdout.write(result_str)    

    if return_dict_NewVertices: return tr, dict_NewVertices
    else: return tr

def CheckWeakStarCondition(st_X, st_Y, LocationMap, method = 'weak', verbose = False): 
    '''
    Check if the map LocationMap: X --> Y satisfies the weak star condition.
    If it does, returns a (randomly choosen) weak simplicial approximation.
    If not, returns the vertices on which f does not satisfy the weak star condition.
    Two methods:
    - method == 'weak'   ---> check the weak star condition
    - method == 'closed' ---> check the closed weak star condition
    /!\ closed weak star approximations may not be simplicial maps /!\     
    It first computes the dictionaries CorrespondingSimplices and AdmissibleVertices:
    - CorrespondingSimplices is a dictionary { (vertex of st_X):(list of simplices of st_Y) }, 
        where each simplex is the image of the neighbors of the vertices of st_X by the LocationMap.
    - AdmissibleVertices is a dictionary { (vertex of st_X):(list of vertices of st_Y) }, 
        where the list contains the vertices of st_Y which satisfies the weak star condition 
        for the vertices in st_X.
    The weak approximation is selected by picking for each vertex of st_X a random admissible vertex of st_Y.
    
    Input: 
    - st_X (gudhi.SimplexTree): simplex tree, domain of LocationMap.
    - st_Y (gudhi.SimplexTree): simplex tree, codomain of LocationMap.
    - method: can be 'weak' or 'closed': whether to look for a (closed) weak star approximation.
    - LocationMap (dic int:(list of int): a map (vertex of st_X):(simplex of st_Y).

    Output:
    - SatisfyCondition: can be True or False, whether the map satisfies the weak star condition.
    - Vertices: depends on SatisfyCondition:
        - if SatisfyCondition == True: a dict {int:int}: a map (vertex of st_X):(vertex of st_Y),
            a weak simplicial approximation to LocationMap.
        - if SatisfyCondition == False: a list of int: the vertices of st_X on which LocationMap does
            not satisfy the weak star condition.
    '''       
    if verbose: sys.stdout.write('| CheckWeakStarCondition | ')
    
    # Get CorrespondingSimplices, i.e., LocationMap of neighbors
    Vertices_X = (filtr[0][0] for filtr in st_X.get_skeleton(0))
    CorrespondingSimplices = {v:[LocationMap[v]] for v in Vertices_X}
    Edges = (filtr[0] for filtr in st_X.get_skeleton(1) if len(filtr[0])==2)
    for edge in Edges:
            CorrespondingSimplices[edge[0]].append(LocationMap[edge[1]])
            CorrespondingSimplices[edge[1]].append(LocationMap[edge[0]])

    # Get AdmissibleVertices, i.e., vertices w in Y such that St(w) contains the CorrespondingSimplices
    # If method is "closed", the closed star is used
    Vertices_X = (filtr[0][0] for filtr in st_X.get_skeleton(0))
    AdmissibleVertices = {v:list() for v in Vertices_X}      
    if method=='weak':
        for v in AdmissibleVertices:  
            for w in LocationMap[v]:
                if all([w in simplex for simplex in CorrespondingSimplices[v]]):
                    AdmissibleVertices[v].append(w)
    elif method=='closed':
        for v in AdmissibleVertices:  
            for w in LocationMap[v]:
                if all([st_Y.find(list(set(simplex+[w]))) for simplex in CorrespondingSimplices[v]]):
                    AdmissibleVertices[v].append(w)
    else:
        raise ValueError('Problem in CheckWeakStarCondition! Non-admissible method. method = '+repr(method)+'.')
    
    # Define VerticesToSubdivise
    VerticesToSubdivise = [v for v in AdmissibleVertices if AdmissibleVertices[v]==[]]
    
    if len(VerticesToSubdivise)>0: #the map does not satisfy the weak star condition
        if verbose:
            val = 100*len(VerticesToSubdivise)/len(AdmissibleVertices)
            val_str = float("{0:.3f}". format(val))
            sys.stdout.write(method+' star condition not satisfied for '+repr(val_str)+'% of the vertices.\n')
        return False, VerticesToSubdivise
    
    else: #the map satisfies the weak star condition
        #Get a random (closed) weak approximation
        if verbose: sys.stdout.write('The map satisfies the '+method+' star condition. ')
        RandomChoiceAdmissibleVertices = {v:random.choice(AdmissibleVertices[v]) for v in AdmissibleVertices}    
        
        # Check if it is a simplicial map (it must be if the method is 'weak')
        is_simplicial = True
        Simplices = (filtr[0] for filtr in st_X.get_filtration())
        for simplex in Simplices:
            simplex_image = sorted(set([RandomChoiceAdmissibleVertices[v] for v in simplex]))
            if not st_Y.find(simplex_image):
                is_simplicial = False
                break
        if verbose:
            if is_simplicial: sys.stdout.write('The map is simplicial.\n')
            else: sys.stdout.write('The map is not simplicial. Problem with '+repr(simplex)+' --> '+repr(simplex_image)+'.\n')
        return True, RandomChoiceAdmissibleVertices
    
'''----------------------------------------------------------------------------
Contractions
----------------------------------------------------------------------------'''
    
def ContractTriangulation(tr, method = 'fast', verbose = False):
    '''
    Contract the simplicial complex K given as a ClassTriangulation tr, via repeated 
    edge contractions.
    
    Input:
    - tr: a ClassTriangulation, representing the simplicial complex to contract.
    - method: 'fast' or 'random':
        - 'fast': use a fast implementation of link condition
        - 'random': compute the link condition for each edges
        
    Output:
    - tr_quotient: a ClassTriangulation, the contracted complex.
    '''

    st = tr.Complex
    InitialVertices = [filtr[0][0] for filtr in st.get_skeleton(0)]
    Vertices = InitialVertices.copy()
    
    st_quotient, Edges_contracted = ContractSimplexTreeFast(st, method = method, verbose = verbose)
                
    # Get DictQuotient, the simplicial quotient map
    DictQuotientEdges = {i:i for i in InitialVertices}
    for edge in Edges_contracted:
        DictQuotientEdges[edge[1]] = edge[0]        
    DictQuotient = {}
    for v in InitialVertices:
        imv = DictQuotientEdges[v]; imimv = DictQuotientEdges[imv]
        while imv != imimv:
            imv = imimv; imimv = DictQuotientEdges[imimv]
        DictQuotient[v] = imv

    # Define a ClassTriangulation
    tr_quotient = ClassTriangulation()   
    tr_quotient.Complex = st_quotient
    tr_quotient.Parent = tr
    tr_quotient.NewVertices = DictQuotient
    def Domain(v,self):
        return self.Parent.Domain(v)
    tr_quotient.add_Domain(Domain)
    def LocationMap(v,self):
        simplex = self.Parent.LocationMap(v)
        simplex_quotient = [self.NewVertices[i] for i in simplex]
        return list(set(simplex_quotient))
    tr_quotient.add_LocationMap(LocationMap)    

    if verbose==True:
        result_str = 'Dim/Simp/Vert = '+repr(st_quotient.dimension())+'/'+repr(st_quotient.num_simplices())+'/'+ repr(st_quotient.num_vertices())+'.\n'
        sys.stdout.write(' '+result_str)      
    return tr_quotient

def ContractSimplexTreeFast(st, method = 'random', verbose = False):
    '''
    Contract a simplicial complex via repeated edge contractions.
    
    Input:
    - st: a gudhi.SimplexTree, representing the simplicial complex to contract.
    - method: 'fast' or 'random':
        - 'fast': use a fast implementation of link condition
        - 'random': compute the link condition for each edges
        
    Output:
    - st: a gudhi.SimplexTree, the contracted complex.
    - LIST_CONTRACTED_EDGES: a list of lists, the contracted edges.

    '''
    # Define auxiliary functions
    def ComputeLinkConditions(LinkConditions, Links_vertices, Links_edges, method='all'):
        '''
        Compute the link conditions.
        If method=='all', compute for all edges.
        If method=='first', stop the process when a first satisfied link condition is found.        
        '''
        if method=='first':
            if True in LinkConditions.values():
                return LinkConditions        
        for edge in LinkConditions:
            if LinkConditions[edge]==None:
                intersection_set = Links_vertices[edge[0]].intersection(Links_vertices[edge[1]])
                condition = intersection_set.issubset(Links_edges[edge])
                if condition:
                    LinkConditions[edge] = True
                    if method=='first': return LinkConditions
                else: LinkConditions[edge] = False
        return LinkConditions    

    def GetLinkVertex(st, v):
        star = [filtr[0] for filtr in st.get_star([v])]
        link = set()
        for simplex in star:
            simplex.remove(v)
            link.add(tuple(simplex))
        if tuple() in link: link.remove(tuple())
        return link

    def GetLinkEdge(st, edge):
        star = [filtr[0] for filtr in st.get_star(edge)]
        link = set()
        for simplex in star:
            simplex.remove(edge[0])
            simplex.remove(edge[1])
            if simplex != []:
                link.add(tuple(simplex))
        return link
    
    LIST_CONTRACTED_EDGES = []
    
    # Copy simplex tree
    st = velour.CopySimplexTree(st)

    # Get vertices
    Vertices = [filtr[0][0] for filtr in st.get_skeleton(0)]

    # Define simplices
    MaximalSimplices = [filtr[0] for filtr in st.get_filtration() if len(filtr[0])>1]

    # Get links of vertices
    Links_vertices ={i:[] for i in Vertices}
    for simplex in MaximalSimplices:
        for i in simplex:
            simplexcopy = simplex.copy()
            simplexcopy.remove(i)
            simplexcopy_tuple = tuple(simplexcopy)
            Links_vertices[i] += [simplexcopy_tuple]        
    for i in Vertices:
        Links_vertices[i] = set(Links_vertices[i])

    # Get edges
    Edges = [tuple(filtr[0]) for filtr in st.get_skeleton(1) if len(filtr[0])==2]
        
    # Get links of edges
    Links_edges ={i:[] for i in Edges}
    for simplex in MaximalSimplices:
        for edge in itertools.combinations(simplex, 2): #edge is sorted, and is a tuple
            simplexcopy = simplex.copy()
            simplexcopy.remove(edge[0]) 
            simplexcopy.remove(edge[1])
            if simplexcopy != []:
                Links_edges[edge] += [tuple(simplexcopy)]
    for edge in Edges:
        Links_edges[edge] = set(Links_edges[edge])

    # Get link conditions
    LinkConditions = {edge:None for edge in Edges}
    if method=='fast': LinkConditions = ComputeLinkConditions(LinkConditions, Links_vertices, Links_edges, method='first')
    if method=='random': LinkConditions = ComputeLinkConditions(LinkConditions, Links_vertices, Links_edges, method='all')
    # Get neighbors
    Neighbors = {v:[] for v in Vertices}
    for filtr in st.get_skeleton(1):
        edge = filtr[0] 
        if len(edge)==2:
            v0 = edge[0]
            v1 = edge[1]
            Neighbors[v0].append(v1)    
            Neighbors[v1].append(v0)
            
    # Main loop
    if verbose: 
        i = 0
        len_vertices = st.num_vertices()
        start_time_contract = time.time()
        
    while True in LinkConditions.values():
        # Pick edge to contract 
        contractable_edges = [edge for edge in LinkConditions if LinkConditions[edge]==True]        
        if method=='fast': edge_contracted = contractable_edges[0]
        if method=='random': edge_contracted = random.choice(contractable_edges)
        LIST_CONTRACTED_EDGES.append(edge_contracted)
                
        v0 = edge_contracted[0]
        v1 = edge_contracted[1] # we shall remove v1
        v0_neigh_closed = Neighbors[v0]+[v0] 
        v1_neigh = Neighbors[v1] 
                
        # Update Neighbors
        for v in v1_neigh:
            if v!=v0:
                Neighbors[v].remove(v1)
                if v0 not in Neighbors[v]:
                    Neighbors[v].append(v0)
        Neighbors[v0] += Neighbors[v1]
        Neighbors[v0] = set(Neighbors[v0])
        Neighbors[v0].remove(v0)
        Neighbors[v0].remove(v1)
        Neighbors[v0] = list(Neighbors[v0])
        Neighbors.pop(v1)        
        
        # New edges to delete
        edges_to_delete = [] #edges that contains v1
        for v in v1_neigh:
            edge = [v1,v]
            edge.sort()
            edges_to_delete.append(tuple(edge))

        # New edges to modify or with v0
        edges_to_modify = [] #edges with an element that is a neighbor of v1, and without v0
        edges_with_v0 = []
        for v in v1_neigh:
            for w in Neighbors[v]:
                if w != v1:
                    edge = [v,w]
                    edge.sort()
                    if v0 in edge:
                        edges_with_v0.append(tuple(edge))
                    else:
                        edges_to_modify.append(tuple(edge))
        edges_to_modify = list(set(edges_to_modify))
        edges_with_v0 = list(set(edges_with_v0)) 
                
        # New edges to add
        edges_to_add = []
        for v in set(v1_neigh).difference(set(v0_neigh_closed)):
            edge = [v0,v]
            edge.sort()
            edges_to_add.append(tuple(edge))
        
        # Update simplex tree
        cofaces = [filtr[0] for filtr in st.get_cofaces([v1], 0)]
        cofaces = list(reversed(sorted(cofaces, key = len)))
        for simplex in cofaces: #remove simplices
            st.remove_maximal_simplex(simplex)
        cofaces.remove([v1])
        for simplex in cofaces: #update simplices
            simplex.remove(v1)
            if v0 not in simplex:
                simplex.append(v0)
        for simplex in cofaces: #insert simplices
            st.insert(simplex)

        # Update vertices
        Vertices.remove(v1)

        # Update link vertices
        Links_vertices.pop(v1)
        for v in v1_neigh:
            NewLink = set()
            for simplex in Links_vertices[v]:
                simplex = list(simplex)
                if v1 in simplex:
                    simplex.remove(v1)
                    if v0 not in simplex:
                        simplex.append(v0)
                        simplex.sort()
                NewLink.add(tuple(simplex))
            Links_vertices[v] = NewLink
        Links_vertices[v0] = GetLinkVertex(st, v0)

        # Update Edges and Link_edges
        for edge in edges_to_delete:
            Edges.remove(edge)
            Links_edges.pop(edge)
        for edge in edges_to_add:
            Edges.append(edge)
            Links_edges[edge] = set()

        # Pre-Update LinkConditions
        for edge in edges_to_delete: LinkConditions.pop(edge)
        for edge in edges_to_add: LinkConditions[edge] = None
        for edge in edges_to_modify: LinkConditions[edge] = None
        for edge in edges_with_v0: LinkConditions[edge] = None
            
        # Update link edges
        for edge in edges_to_add: Links_edges[edge] = GetLinkEdge(st, edge) #améliorer ici
        for edge in edges_with_v0: Links_edges[edge] = GetLinkEdge(st, edge) #améliorer ici
        for edge in edges_to_modify:
            NewLink = set()
            for simplex in Links_edges[edge]:
                simplex = list(simplex)
                if v1 in simplex:
                    simplex.remove(v1)
                    if v0 not in simplex:
                        simplex.append(v0)
                        simplex.sort()
                if simplex != []:
                    NewLink.add(tuple(simplex))
            Links_edges[edge] = NewLink

        # Link conditions
        if method=='fast': LinkConditions = ComputeLinkConditions(LinkConditions, Links_vertices, Links_edges, method='first')
        if method=='random': LinkConditions = ComputeLinkConditions(LinkConditions, Links_vertices, Links_edges, method='all')
        
        if verbose: 
            i = i+1
            elapsed_time_secs = time.time() - start_time_contract
            expected_time_secs = (len_vertices-i-1)/(i+1)*elapsed_time_secs
            msg1 = 'Vertex '+repr(i)+'/'+repr(len_vertices)+'. '
            msg2 = 'Duration %s' % timedelta(seconds=round(elapsed_time_secs))
            msg3 = '/%s.' % timedelta(seconds=round(expected_time_secs))
            sys.stdout.write('\r| ContractTriangulation  | '+method+'. '+msg1+msg2+msg3) 
    
    return st, LIST_CONTRACTED_EDGES

def SimplifyDelaunay(st, st_Y, SimplicialMap, Coordinates, verbose=True, keep_first_vertices = True):
    '''
    Simplify the SimplicialMap: st --> st_Y, where st is a Delaunay complex on the vertices given by
    'Coordinates'. We iteratively suppress vertices on which the map satisfy the simplex condition.
    
    Input:
    - st: a gudhi.SimplexTree, the domain of the simplicial map. It is a Delaunay complex.
    - st_Y: a gudhi.SimplexTree, the codomain of the simplicial map.
    - SimplicialMap: a dict {int:int}, representing the simplicial map. 
    - Coordinates: a dict {int:np.array}, representing the coordinatesof the vertices of the Delaunay
        complex st.
    - verbose: can be True or False, whether to print comments.
    - keep_first_vertices: can be True or False. If True, do not suppress the st.dimension()+2 first 
        vertices of st. This ensures that the Delaunay complex is remains a triangulation of the sphere.
        
    Output:
    - Sphere: a ClassTriangulation, representing the simplified sphere.
    - SimplicialMap: a dict {int:int}, representing the simplified simplicial map. 
    '''
    # Check if it is a simplicial map
    if verbose:
        is_simplicial = True
        Simplices = (filtr[0] for filtr in st.get_filtration())
        for simplex in Simplices:
            simplex_image = sorted(set([SimplicialMap[v] for v in simplex]))
            if not st_Y.find(simplex_image):
                is_simplicial = False
                break
#        if is_simplicial: sys.stdout.write(' The map is simplicial. ')
        if not is_simplicial: sys.stdout.write(' The map is not simplicial. Problem with '+repr(simplex)+' --> '+repr(simplex_image)+' ')

    if verbose: 
        msg = '| SimplifyDelaunay       |'
        sys.stdout.write(msg)    
    
    st = velour.CopySimplexTree(st)
    Coordinates = Coordinates.copy()
    SimplicialMap = SimplicialMap.copy()
    Vertices = list(Coordinates.keys())
    
    Continue = True
    if verbose: 
        Vertices_len = len(Vertices)
        start_time = time.time()
    while Continue:
        Neighbors = {i:[i] for i in Vertices}
        Edges = (filtr[0] for filtr in st.get_skeleton(1) if len(filtr[0])==2)
        for edge in Edges:
            Neighbors[edge[0]].append(edge[1])
            Neighbors[edge[1]].append(edge[0])

        PotentialVertices = {i:True for i in Vertices}
        AdmissibleVertices = list()
        for x in Vertices:
            if (x >= st.dimension()+2) or keep_first_vertices==False: #do not supress the base vertices
                if PotentialVertices[x]:
                    neighbors = Neighbors[x]
                    neighbors_image = sorted(list(set(SimplicialMap[v] for v in neighbors)))
                    if st_Y.find(neighbors_image): #the simplex condition
                        AdmissibleVertices.append(x)
                        for v in neighbors: PotentialVertices[v] = False
            
        if len(AdmissibleVertices)>0:
            # Update lists
            for x in AdmissibleVertices:
                Coordinates.pop(x)
                SimplicialMap.pop(x)
            OldVertices = list(Coordinates.keys())
            Vertices = range(len(Coordinates))
            Index = {i:OldVertices[i] for i in Vertices}
            Coordinates = {i:Coordinates[Index[i]] for i in Vertices}
            SimplicialMap = {i:SimplicialMap[Index[i]] for i in Vertices}

            # Compute convex hull
            points = np.array([Coordinates[v] for v in Coordinates])
            hull = scipy.spatial.ConvexHull(points)

            # Create new complex
            st = gudhi.SimplexTree()
            for simplex in hull.simplices: st.insert(simplex)
                
            if verbose:
                i = Vertices_len - len(Vertices)
                elapsed_time_secs = time.time() - start_time
                expected_time_secs = (Vertices_len-i-1)/(i+1)*elapsed_time_secs
                msg1 = 'Vertex '+repr(i+1)+'/'+repr(Vertices_len)+'. '
                msg2 = 'Duration %s' % timedelta(seconds=round(elapsed_time_secs))
                msg3 = '/%s.' % timedelta(seconds=round(expected_time_secs))
                sys.stdout.write('\r'+msg+' '+msg1+msg2+msg3)    
        else: Continue = False

    # Check if it is a simplicial map
    if verbose:
        is_simplicial = True
        Simplices = (filtr[0] for filtr in st.get_filtration())
        for simplex in Simplices:
            simplex_image = sorted(set([SimplicialMap[v] for v in simplex]))
            if not st_Y.find(simplex_image):
                is_simplicial = False
                break
#        if is_simplicial: sys.stdout.write(' The map is simplicial. ')
        if not is_simplicial: sys.stdout.write(' The map is not simplicial. Problem with '+repr(simplex)+' --> '+repr(simplex_image)+' ')
            
    # Create data structure ClassTriangulation
    Sphere = ClassTriangulation()
    Sphere.Complex = st
    Sphere.Coordinates = Coordinates
    Sphere.Tree = []
    Sphere.add_LocationMap(LocationMapSphereRadialFast)

    if verbose:
        result_str = ' Dim/Simp/Vert = '+repr(Sphere.Complex.dimension())+'/'+repr(Sphere.Complex.num_simplices())+'/'+ repr(Sphere.Complex.num_vertices())+'.\n'
        sys.stdout.write(result_str)
    return Sphere, SimplicialMap    
  
'''----------------------------------------------------------------------------
Projective Spaces
----------------------------------------------------------------------------'''

def GluingMapProjectiveSpace(vect, self):
    return velour.VectorToProjection(vect)

def InvCharacteristicMapProjectiveSpace(proj, self):
    vect = velour.ProjectionToVector(proj)
    if vect[-1] != 0: vect = vect*np.sign(vect[-1])
    vect_del = vect[:-1]
    return vect_del

def DomainProjectiveSpace(vect, self):
    if isinstance(vect, np.ndarray):
        if np.shape(vect) == tuple([self.Dimension+1, self.Dimension+1]):
            if np.linalg.norm(vect) != 0:
                return True
            else: return False
        else: return False
    else: return False

'''----------------------------------------------------------------------------
Grassmannian
----------------------------------------------------------------------------'''

def TwoBallsToBall(x1,x2):
    y = np.concatenate((x1,x2))
    if np.linalg.norm(y)==0: return y
    else: 
        c = max(np.linalg.norm(x1),np.linalg.norm(x2))/np.linalg.norm(y)
        return c*y

def BallToTwoBalls(y,n1):
    x1 = y[0:n1]; x2 = y[n1:]
    c = max(np.linalg.norm(x1),np.linalg.norm(x2))
    if c == 0: return x1, x2
    else: return x1/c, x2/c

def ReducedEchelonForm(rref, eps=1e-5):
    # Epsilon-threshold
    rref[abs(rref)<eps]=0
    rref1 = rref[0,:]
    rref2 = rref[1,:]        

    # Orthonormality
    rref1 = rref1/np.linalg.norm(rref1)         #normalize rref1
    rref2 = rref2-np.dot(rref1, rref2)*rref1    #make rref1 and rref2 orthogonal
    rref2 = rref2/np.linalg.norm(rref2)         #normalize rref2

    # Positivity
    n1 = np.max(np.nonzero(rref1)) #last nonzero entry
    if rref1[n1]<0: rref1 = -rref1
    n2 = np.max(np.nonzero(rref2)) #last nonzero entry
    if rref2[n2]<0: rref2 = -rref2    
    rref = np.array([rref1,rref2])
    
    # Permutation
    if n1>n2: 
        rref1, rref2 = rref2, rref1    #swap rows
        rref = np.array([rref1,rref2])

    # Reduce
    if n1 == n2:
        rref1 = rref1 - rref2*rref1[n1]/rref2[n2]
        rref = np.array([rref1,rref2])
        rref = ReducedEchelonForm(rref)
        
    # Epsilon-threshold
    rref[abs(rref)<eps]=0
        
    return rref

def GluingMapGrassmannian(x, self):
    x1, x2 = BallToTwoBalls(x,self.SchubertSymbol[0]-1)
    if np.linalg.norm(x1)>1: v1 = np.concatenate(( x1, [0] ))
    else : v1 = np.concatenate(( x1, [np.sqrt(1-np.linalg.norm(x1)**2)] ))
    if np.linalg.norm(x2)>1: v2 = np.concatenate(( x2[0:self.SchubertSymbol[0]-1], [0], x2[self.SchubertSymbol[0]-1:], [0] ))
    else: v2 = np.concatenate(( x2[0:self.SchubertSymbol[0]-1], [0], x2[self.SchubertSymbol[0]-1:], [np.sqrt(1-np.linalg.norm(x2)**2)] ))
    v1 = np.pad(v1, (0,4-len(v1)))
    v2 = np.pad(v2, (0,4-len(v2)))
    s = np.zeros(4); s[self.SchubertSymbol[0]-1] = 1
    R = v2 - np.dot(v1,v2)/(1 + np.dot(s,v1))*(s+v1)
    rref = np.stack((v1, R))
    rref = ReducedEchelonForm(rref)
    return rref

def InvCharacteristicMapGrassmannian(v, self):
    v1 = v[0,:]; v2 = v[1,:]
    s = np.zeros(4); s[self.SchubertSymbol[0]-1] = 1
    R = v2 - np.dot(s,v2)/(1 + np.dot(s,v1))*(s+v1)
    x1 = v1[0:self.SchubertSymbol[0]-1]
    x2 = R[0:self.SchubertSymbol[1]-1]; 
    x2 = np.delete(x2, self.SchubertSymbol[0]-1)
    x = TwoBallsToBall(x1,x2)
    return x

def DomainGrassmannian(vect, self):
    epsilon_domain = 1e-13
    if isinstance(vect, np.ndarray):
        if np.shape(vect) == tuple([2,4]):
            if all(np.abs(vect[0,self.SchubertSymbol[0]:]<=epsilon_domain)) \
                and all(np.abs(vect[1,self.SchubertSymbol[1]:]<=epsilon_domain)):
                return True
            else:
                return False
        else:
            print('Vector has wrong shape in DomainGrassmannian.')
            return False
    else:
        print('Error! Vector is wrong instance in DomainGrassmannian.')
        return False
