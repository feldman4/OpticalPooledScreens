import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression

import ops.utils

def find_triangles(df):
    v, c = get_vectors(df[['i', 'j']].values)

    return (pd.concat([
        pd.DataFrame(v).rename(columns='V_{0}'.format), 
        pd.DataFrame(c).rename(columns='c_{0}'.format)], axis=1)
        .assign(magnitude=lambda x: x.eval('(V_0**2 + V_1**2)**0.5'))
    )

    return df_

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

def brute_force_pairs(df_0, df_1):
    from tqdm import tqdm_notebook as tqdn
    arr = []
    for site, df_s in tqdn(df_1.groupby('site'), 'site'):

        def work_on(df_t):
            rotation, translation, score = evaluate_match(df_t, df_s)
            determinant = None if rotation is None else np.linalg.det(rotation)
            result = pd.Series({'rotation': rotation, 
                                'translation': translation, 
                                'score': score, 
                                'determinant': determinant})
            return result

        (df_0
         .pipe(ops.utils.gb_apply_parallel, 'tile', work_on)
         .assign(site=site)
         .pipe(arr.append)
        )
        
    return (pd.concat(arr).reset_index()
            .sort_values('score', ascending=False)
            )

def parallel_process(func, args_list, n_jobs, tqdn=True):
    from joblib import Parallel, delayed
    work = args_list
    if tqdn:
        from tqdm import tqdm_notebook 
        work = tqdm_notebook(work, 'work')
    return Parallel(n_jobs=n_jobs)(delayed(func)(*w) for w in work)


def merge_sbs_phenotype(df_sbs_, df_ph_, model):

    X = df_sbs_[['i', 'j']].values
    Y = df_ph_[['i', 'j']].values
    Y_pred = model.predict(X)

    threshold = 2

    distances = cdist(Y, Y_pred, metric='sqeuclidean')
    ix = distances.argmin(axis=1)
    filt = np.sqrt(distances.min(axis=1)) < threshold
    columns = {'site': 'site', 'cell_ph': 'cell_ph',
              'i': 'i_ph', 'j': 'j_ph',}

    cols_final = ['well', 'tile', 'cell', 'i', 'j', 
                  'site', 'cell_ph', 'i_ph', 'j_ph', 'distance'] 
    sbs = df_sbs_.iloc[ix[filt]].reset_index(drop=True)
    return (df_ph_
     [filt].reset_index(drop=True)
     [list(columns.keys())]
     .rename(columns=columns)
     .pipe(lambda x: pd.concat([sbs, x], axis=1))
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

    ax.set_xlim([-50, 1550])
    ax.set_ylim([-50, 1550])
    
    return ax


def multistep_alignment(df_0, df_1, df_info_0, df_info_1, 
                        initial_sites=8, batch_size=180):
    """Provide triangles from one well only.
    """
    sites = (pd.Series(df_info_1.index)
        .sample(initial_sites, replace=False, random_state=0)
        .pipe(list))

    df_initial = brute_force_pairs(df_0, df_1.query('site == @sites'))

    # dets = df_initial.query('score > 0.3')['determinant']
    # d0, d1 = dets.min(), dets.max()
    # delta = (d1 - d0)
    # d0 -= delta * 1.5
    # d1 += delta * 1.5

    d0, d1 = 1.125, 1.186
    gate = '@d0 <= determinant <= @d1 & score > 0.1'

    alignments = [df_initial.query(gate)]

    #### iteration

    def work_on(df_t, df_s):
        rotation, translation, score = evaluate_match(df_t, df_s)
        determinant = None if rotation is None else np.linalg.det(rotation)
        result = pd.Series({'rotation': rotation, 
                            'translation': translation, 
                            'score': score, 
                            'determinant': determinant})
        return result

    batch_size = 180

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

        df_align_new = (pd.concat(parallel_process(work_on, work, 18), axis=1).T
         .assign(tile=[t for t, _ in candidates[:batch_size]], 
                 site=[s for _, s in candidates[:batch_size]])
        )

        alignments += [df_align_new]
        if len(df_align_new.query(gate)) == 0:
            break
            
    return df_align