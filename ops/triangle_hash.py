"""Delaunay triangle-based alignment between segmented microscopy datasets of the same sample.

Helpful for aligning between datasets of the same sample with different magnification, imaging grid, etc.

1. Build hashed Delaunay triangulations of both segmented datasets using `find_triangles`
2. Perform rough alignment by finding matched tiles/sites between datasets and 
    corresponding rotation & translation transformations of points using `multistep_alignment`.
3. Perform fine alignment and merging of labeled segmentations using `merge_sbs_phenotype`.
"""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings

from . import utils


def find_triangles(df):
    """Turns a table of i,j coordinates (typically of nuclear centroids)
    into a table containing a hashed Delaunay triangulation of the 
    input points. Process each tile/site independently. The output for all
    tiles/sites within a single well is concatenated and used as input to 
    `multistep_alignment`.

    Parameters
    ----------
    df : pandas DataFrame
        Table of points with columns `i` and `j`.

    Returns
    -------
    df_dt : pandas DataFrame
        Table containing a hashed Delaunay triangulation, one line
        per simplex (triangle).

    """
    v, c = get_vectors(df[['i', 'j']].values)

    return (pd.concat([
        pd.DataFrame(v).rename(columns='V_{0}'.format), 
        pd.DataFrame(c).rename(columns='c_{0}'.format)], axis=1)
        .assign(magnitude=lambda x: x.eval('(V_0**2 + V_1**2)**0.5'))
    )


def nine_edge_hash(dt, i):
    """For triangle `i` in Delaunay triangulation `dt`, extract the vector 
    displacements of the 9 edges containing to at least one vertex in the 
    triangle.

    Raises an error if triangle `i` lies on the outer boundary of the triangulation.

    Example:
    dt = Delaunay(X_0)
    i = 0
    segments, vector = nine_edge_hash(dt, i)
    plot_nine_edges(X_0, segments)

    """
    # indices of inner three vertices
    # already in CCW order
    a,b,c = dt.simplices[i]

    # reorder so ab is the longest
    X = dt.points
    start = np.argmax((np.diff(X[[a, b, c, a]], axis=0)**2).sum(axis=1)**0.5)
    if start == 0:
        order = [0, 1, 2]
    elif start == 1:
        order = [1, 2, 0]
    elif start == 2:
        order = [2, 0, 1]
    a,b,c = np.array([a,b,c])[order]

    # outer three vertices
    a_ix, b_ix, c_ix = dt.neighbors[i]
    inner = {a,b,c}
    outer = lambda xs: [x for x in xs if x not in inner][0]
    # should be two shared, one new; if not, probably a weird edge simplex
    # that shouldn't hash (return None)
    try:
        bc = outer(dt.simplices[dt.neighbors[i, order[0]]])
        ac = outer(dt.simplices[dt.neighbors[i, order[1]]])
        ab = outer(dt.simplices[dt.neighbors[i, order[2]]])
    except IndexError:
        return None

    if any(x == -1 for x in (bc, ac, ab)):
        error = 'triangle on outer boundary, neighbors are: {0} {1} {2}'
        raise ValueError(error.format(bc, ac, ab))
    
    # segments
    segments = [
     (a, b),
     (b, c),
     (c, a),
     (a, ab),
     (b, ab),
     (b, bc),
     (c, bc),
     (c, ac),
     (a, ac),
    ]

    i = X[segments, 0]
    j = X[segments, 1]
    vector = np.hstack([np.diff(i, axis=1), np.diff(j, axis=1)])
    return segments, vector

def plot_nine_edges(X, segments):
    fig, ax = plt.subplots()
    
    [(a, b),
     (b, c),
     (c, a),
     (a, ab),
     (b, ab),
     (b, bc),
     (c, bc),
     (c, ac),
     (a, ac)] = segments
    
    for i0, i1 in segments:
        ax.plot(X[[i0, i1], 0], X[[i0, i1], 1])

    d = {'a': a, 'b': b, 'c': c, 'ab': ab, 'bc': bc, 'ac': ac}
    for k,v in d.items():
        i,j = X[v]
        ax.text(i,j,k)

    ax.scatter(X[:, 0], X[:, 1])

    s = X[np.array(segments).flatten()]
    lim0 = s.min(axis=0) - 100
    lim1 = s.max(axis=0) + 100

    ax.set_xlim([lim0[0], lim1[0]])
    ax.set_ylim([lim0[1], lim1[1]])
    return ax

def get_vectors(X):
    """Get the nine edge vectors and centers for all the faces in the 
    Delaunay triangulation of point array `X`.
    """
    dt = Delaunay(X)
    vectors, centers = [], []
    for i in range(dt.simplices.shape[0]):
        # skip triangles with an edge on the outer boundary
        if (dt.neighbors[i] == -1).any():
            continue
        result = nine_edge_hash(dt, i)
        # some rare event 
        if result is None:
            continue
        _, v = result
        c = X[dt.simplices[i], :].mean(axis=0)
        vectors.append(v)
        centers.append(c)

    return np.array(vectors).reshape(-1, 18), np.array(centers)

def nearest_neighbors(V_0, V_1):
    Y = cdist(V_0, V_1, metric='sqeuclidean')
    distances = np.sqrt(Y.min(axis=1))
    ix_0 = np.arange(V_0.shape[0])
    ix_1 = Y.argmin(axis=1)
    return ix_0, ix_1, distances

def get_vc(df, normalize=True):
    V,c = (df.filter(like='V').values, 
            df.filter(like='c').values)
    if normalize:
        V = V / df['magnitude'].values[:, None]
    return V, c

def evaluate_match(df_0, df_1, threshold_triangle=0.3, threshold_point=2):
    
    V_0, c_0 = get_vc(df_0)
    V_1, c_1 = get_vc(df_1)

    i0, i1, distances = nearest_neighbors(V_0, V_1)

    # matching triangles
    filt = distances < threshold_triangle
    X, Y = c_0[i0[filt]], c_1[i1[filt]]

    # minimum to proceed
    if sum(filt) < 5:
        return None, None, -1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # use matching triangles to define transformation
        model = RANSACRegressor()
        model.fit(X, Y)
    
    rotation = model.estimator_.coef_
    translation = model.estimator_.intercept_
    
    # score transformation based on triangle i,j centers
    distances = cdist(model.predict(c_0), c_1, metric='sqeuclidean')
    # could use a fraction of the data range or nearest neighbor 
    # distances within one point set
    threshold_region = 50
    filt = np.sqrt(distances.min(axis=0)) < threshold_region
    score = (np.sqrt(distances.min(axis=0))[filt] < threshold_point).mean()
    
    return rotation, translation, score

def build_linear_model(rotation, translation):
    m = LinearRegression()
    m.coef_ = rotation
    m.intercept_ = translation
    return m

def prioritize(df_info_0, df_info_1, matches):
    """Produces an Nx2 array of tile (site) identifiers that are predicted
    to match within a search radius, based on existing matches.
    
    Expects info tables to contain tile (site) identifier as index
    and two columns of coordinates. Matches should be supplied as an 
    Nx2 array of tile (site) identifiers.
    """
    a = df_info_0.loc[matches[:, 0]].values
    b = df_info_1.loc[matches[:, 1]].values

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = RANSACRegressor()
        model.fit(a, b)

    # rank all pairs by distance
    predicted = model.predict(df_info_0.values)
    distances = cdist(predicted, df_info_1, metric='sqeuclidean')
    ix = np.argsort(distances.flatten())
    ix_0, ix_1 = np.unravel_index(ix, distances.shape)

    candidates = list(zip(df_info_0.index[ix_0], df_info_1.index[ix_1]))

    return remove_overlap(candidates, matches)


def remove_overlap(xs, ys):
    ys = set(map(tuple, ys))
    return [tuple(x) for x in xs if tuple(x) not in ys]

def brute_force_pairs(df_0, df_1, n_jobs=-2, tqdn=True):
    if tqdn:
        from tqdm.auto import tqdm
        work = tqdm(df_1.groupby('site'),'site')
    else:
        work = df_1.groupby('site')

    arr = []
    for site, df_s in work:

        def work_on(df_t):
            rotation, translation, score = evaluate_match(df_t, df_s)
            determinant = None if rotation is None else np.linalg.det(rotation)
            result = pd.Series({'rotation': rotation, 
                                'translation': translation, 
                                'score': score, 
                                'determinant': determinant})
            return result

        (df_0
         .pipe(utils.gb_apply_parallel, 'tile', work_on, n_jobs=n_jobs)
         .assign(site=site)
         .pipe(arr.append)
        )
        
    return (pd.concat(arr).reset_index()
            .sort_values('score', ascending=False)
            )

def parallel_process(func, args_list, n_jobs, tqdn=True):
    from joblib import Parallel, delayed
    if tqdn:
        from tqdm.auto import tqdm
        work = tqdm(args_list, 'work')
    else:
        work = args_list
    return Parallel(n_jobs=n_jobs)(delayed(func)(*w) for w in work)


def merge_sbs_phenotype(df_0_, df_1_, model, threshold=2):
    """Fine alignment of one (tile,site) match found using 
    `multistep_alignment`.

    Parameters
    ----------
    df_0_ : pandas DataFrame
        Table of coordinates to align (e.g., nuclei centroids) 
        for one tile of dataset 0. Expects `i` and `j` columns.

    df_1_ : pandas DataFrame
        Table of coordinates to align (e.g., nuclei centroids) 
        for one site of dataset 1 that was identified as a match
        to the tile in df_0_ using `multistep_alignment`. Expects 
        `i` and `j` columns.

    model : sklearn.linear_model.LinearRegression
        Linear alignment model between tile of df_0_ and site of 
        df_1_. Produced using `build_linear_model` with the rotation 
        and translation matrix determined in `multistep_alignment`.

    threshold : float, default 2
        Maximum euclidean distance allowed between matching points.

    Returns
    -------
    df_merge : pandas DataFrame
        Table of merged identities of cell labels from df_0_ and 
        df_1_.
    """

    X = df_0_[['i', 'j']].values
    Y = df_1_[['i', 'j']].values
    Y_pred = model.predict(X)

    distances = cdist(Y, Y_pred, metric='sqeuclidean')
    ix = distances.argmin(axis=1)
    filt = np.sqrt(distances.min(axis=1)) < threshold
    columns_0 = {'tile': 'tile', 'cell': 'cell_0',
              'i': 'i_0', 'j': 'j_0',}
    columns_1 = {'site': 'site', 'cell': 'cell_1',
              'i': 'i_1', 'j': 'j_1',}

    cols_final = ['well', 'tile', 'cell_0', 'i_0', 'j_0', 
                  'site', 'cell_1', 'i_1', 'j_1', 'distance'] 
    target = df_0_.iloc[ix[filt]].reset_index(drop=True).rename(columns=columns_0)
    return (df_1_
     [filt].reset_index(drop=True)
     [list(columns_1.keys())]
     .rename(columns=columns_1)
     .pipe(lambda x: pd.concat([target, x], axis=1))
     .assign(distance=np.sqrt(distances.min(axis=1))[filt])
     [cols_final]
    )


def plot_alignments(df_ph, df_sbs, df_align, site):
    """Filter for one well first.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    X_0 = df_ph.query('site == @site')[['i', 'j']].values
    ax.scatter(X_0[:, 0], X_0[:, 1], s=10)
    
    it = (df_align
          .query('site == @site')
          .sort_values('score', ascending=False)
          .iterrows())
    
    for _, row in it:
        tile = row['tile']
        X = df_sbs.query('tile == @tile')[['i', 'j']].values
        model = build_linear_model(row['rotation'], row['translation'])
        Y = model.predict(X)
        ax.scatter(Y[:, 0], Y[:, 1], s=1, label=tile)
        print(tile)
    
    return ax


def multistep_alignment(df_0, df_1, df_info_0, df_info_1, 
                        det_range=(1.125,1.186),
                        initial_sites=8, batch_size=180, 
                        tqdn=True, n_jobs=None):
    """Finds tiles of two different acquisitions with matching Delaunay 
    triangulations within the same well. Cells must not have moved significantly
    between acquisitions and segmentations approximately equivalent.

    Parameters
    ----------
    df_0 : pandas DataFrame
        Hashed Delaunay triangulation for all tiles of dataset 0. Produced by 
        concatenating outputs of `find_triangles` from individual tiles of a 
        single well. Expects a `tile` column.

    df_1 : pandas DataFrame
        Hashed Delaunay triangulation for all sites of dataset 1. Produced by 
        concatenating outputs of `find_triangles` from individual sites of a 
        single well. Expects a `site` column.

    df_info_0 : pandas DataFrame
        Table of global coordinates for each tile acquisition to match tiles
        of `df_0`. Expects `tile` as index and two columns of coordinates.

    df_info_1 : pandas DataFrame
        Table of global coordinates for each site acquisition to match sites 
        of `df_1`. Expects `site` as index and two columns of coordinates.

    det_range : 2-tuple, default (1.125,1.186)
        Range of acceptable values for the determinant of the rotation matrix 
        when evaluating an alignment of a tile,site pair. Rotation matrix determinant
        is a measure of the scaling between sites, should be consistent within microscope
        acquisition settings. Calculate determinant for several known matches in a dataset 
        to determine.

    initial_sites : int or list of 2-tuples, default 8
        If int, the number of sites to sample from df_1 for initial brute force 
        matching of tiles to build an initial global alignment model. Brute force 
        can be inefficient and inaccurate. If a list of 2-tuples, these are known 
        matches of (tile,site) to initially evaluate and start building a global 
        alignment model. 5 or more intial pairs of known matching sites should be 
        sufficient.

    batch_size : int, default 180
        Number of (tile,site) matches to evaluate in a batch between updates of the global 
        alignment model.

    tqdn : boolean, default True
        Displays tqdm progress bar if True.

    n_jobs : int or None, default None
        Number of parallelized jobs to deploy using joblib.

    Returns
    -------
    df_align : pandas DataFrame
        Table of possible (tile,site) matches with corresponding rotation and translation 
        transformations. All tested matches are included here, should query based on `score` 
        and `determinant` to keep only valid matches.

    """

    if n_jobs is None:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count() - 1

    def work_on(df_t, df_s):
        rotation, translation, score = evaluate_match(df_t, df_s)
        determinant = None if rotation is None else np.linalg.det(rotation)
        result = pd.Series({'rotation': rotation, 
                            'translation': translation, 
                            'score': score, 
                            'determinant': determinant})
        return result

    if isinstance(initial_sites,list):
        arr = []
        for tile,site in initial_sites:
            result = work_on(df_0.query('tile==@tile'),df_1.query('site==@site'))
            result.at['site']=site
            result.at['tile']=tile
            arr.append(result)
            
        df_initial = pd.DataFrame(arr)
    else:
        sites = (pd.Series(df_info_1.index)
            .sample(initial_sites, replace=False, random_state=0)
            .pipe(list))

        df_initial = brute_force_pairs(df_0, df_1.query('site == @sites'), tqdn=tqdn, n_jobs=n_jobs)

    # dets = df_initial.query('score > 0.3')['determinant']
    # d0, d1 = dets.min(), dets.max()
    # delta = (d1 - d0)
    # d0 -= delta * 1.5
    # d1 += delta * 1.5

    d0, d1 = det_range

    gate = '@d0 <= determinant <= @d1 & score > 0.1'

    alignments = [df_initial.query(gate)]

    #### iteration

    while True:
        df_align = (pd.concat(alignments, sort=True)
                    .drop_duplicates(['tile', 'site']))

        tested = df_align.reset_index()[['tile', 'site']].values
        matches = (df_align.query(gate).reset_index()[['tile', 'site']].values)
        candidates = prioritize(df_info_0, df_info_1, matches)
        candidates = remove_overlap(candidates, tested)

        print('matches so far: {0} / {1}'.format(
            len(matches), df_align.shape[0]))

        work = []
        d_0 = dict(list(df_0.groupby('tile')))
        d_1 = dict(list(df_1.groupby('site')))
        for ix_0, ix_1 in candidates[:batch_size]:
            work += [[d_0[ix_0], d_1[ix_1]]]    

        df_align_new = (pd.concat(parallel_process(work_on, work, n_jobs=n_jobs, tqdn=tqdn), axis=1).T
         .assign(tile=[t for t, _ in candidates[:batch_size]], 
                 site=[s for _, s in candidates[:batch_size]])
        )

        alignments += [df_align_new]
        if len(df_align_new.query(gate)) == 0:
            break
            
    return df_align