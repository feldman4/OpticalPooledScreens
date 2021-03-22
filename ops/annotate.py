import numpy as np
import pandas as pd
import skimage.morphology
import warnings
import os
import PIL.Image
import PIL.ImageFont

from .constants import *
import ops.filenames
import ops
import ops.io


# load font
VISITOR_PATH = os.path.join(os.path.dirname(ops.__file__), 'visitor1.ttf')
try:
    VISITOR_FONT = PIL.ImageFont.truetype(VISITOR_PATH)
except OSError as e:
    warnings.warn('visitor font not found at {0}'.format(VISITOR_PATH))


def annotate_labels(df, label, value, label_mask=None, tag='cells', outline=False, 
                    return_key=False):
    """Transfer `value` from dataframe `df` to a saved integer image mask, using 
    `label` as an index. 

    The dataframe should contain data from a single image, which is loaded from
    `label_mask` if provided, or else guessed based on descriptors in the first 
    row of `df` and `tag`. If the values are non-numeric (strings, etc), label them with
    categorical codes (casting to categorical if necessary). Codes are sequential integers 
    starting from 1 since the background is 0. If `return_key` is true, also return dictionary 
    from labels to values.
    """
    if df[label].duplicated().any():
        raise ValueError('duplicate rows present')

    label_to_value = df.set_index(label, drop=False)[value]
    label_key = {}
    index_dtype = label_to_value.index.dtype
    value_dtype = label_to_value.dtype
    if not np.issubdtype(index_dtype, np.integer):
        raise ValueError(f'label column {label} is not integer type')

    if not np.issubdtype(value_dtype, np.number) or isinstance(value_dtype, pd.CategoricalDtype):
        label_to_value = label_to_value.astype('category')
        value_dtype = pd.CategoricalDtype
        # warnings.warn(f'converting value column {value} to categorical')

    if value_dtype == pd.CategoricalDtype:
        label_key = {i + 1: v for i,v in enumerate(label_to_value.cat.categories)}
        # offset by one since background is zero
        label_to_value = label_to_value.cat.codes + 1
        
    if label_to_value.index.duplicated().any():
        raise ValueError('duplicate index')

    top_row = df.iloc[0]
    if label_mask is None:
        filename = ops.filenames.guess_filename(top_row, tag)
        labels = ops.io.read_stack(filename)
    elif isinstance(label_mask, str):
        labels = ops.io.read_stack(label_mask)
    else:
        labels = label_mask
    
    if outline == 'label':
        labels = outline_mask(labels, 'inner')
    
    labeled_by_value = relabel_array(labels, label_to_value)

    if outline == 'value':
        labeled_by_value = outline_mask(labeled_by_value, 'inner')
    
    if all([x == int(x) for x in label_to_value.values]):
        labeled_by_value = labeled_by_value.astype(int)

    if return_key:
        return labeled_by_value, label_key
    else:
        return labeled_by_value


def annotate_points(df, value, ij=('i', 'j'), width=3, shape=(1024, 1024), selem=None):
    """Create a mask with pixels at coordinates `ij` set to `value` from 
    dataframe `df`. Dilation is performed with `selem` if provided, or else a square of
    `width`.
    """
    ij = df[list(ij)].values.astype(int)
    n = ij.shape[0]
    mask = np.zeros(shape, dtype=df[value].dtype)
    mask[ij[:, 0], ij[:, 1]] = df[value]

    if selem is None:
        selem = np.ones((width, width))
        
    mask = skimage.morphology.dilation(mask, selem)

    return mask


def relabel_array(arr, new_label_dict):
    """Map values in integer array based on `new_labels`, a dictionary from
    old to new values.
    """
    n = arr.max()
    arr_ = np.zeros(n+1)
    for old_val, new_val in new_label_dict.items():
        if old_val <= n:
            arr_[old_val] = new_val
    return arr_[arr]


def outline_mask(arr, direction='outer'):
    """Remove interior of label mask in `arr`.
    """
    arr = arr.copy()
    if direction == 'outer':
        mask = skimage.morphology.erosion(arr)
        arr[mask > 0] = 0
        return arr
    elif direction == 'inner':
        mask1 = skimage.morphology.erosion(arr) == arr
        mask2 = skimage.morphology.dilation(arr) == arr
        arr[mask1 & mask2] = 0
        return arr
    else:
        raise ValueError(direction)
    

def bitmap_label(labels, positions, colors=None):
    positions = np.array(positions).astype(int)
    if colors is None:
        colors = [1] * len(labels)
    i_all, j_all, c_all = [], [], []
    for label, (i, j), color in zip(labels, positions, colors):
        if label == '':
            continue
        i_px, j_px = np.where(bitmap_line(label))
        i_all += list(i_px + i)
        j_all += list(j_px + j)
        c_all += [color] * len(i_px)
        
    shape = max(i_all) + 1, max(j_all) + 1
    arr = np.zeros(shape, dtype=int)
    arr[i_all, j_all] = c_all
    return arr


def build_discrete_lut(colors):
    """Build ImageJ lookup table for list of discrete colors. 

    If the values to  label are in the range 0..N, N + 1 colors should be 
    provided (zero value is usually black). Color values should be understood 
    by `sns.color_palette` (e.g., "blue", (1, 0, 0), or "#0000ff").
    """
    try:
        import seaborn as sns
        colors = sns.color_palette(colors)
    except:
        pass
    colors = 255 * np.array(colors)

    # try to match ImageJ LUT rounding convention
    m = len(colors)
    n = int(256 / m)
    p = m - (256 - n * m)
    color_index_1 = list(np.repeat(range(0, p), n))
    color_index_2 = list(np.repeat(range(p, m), n + 1))
    color_index = color_index_1 + color_index_2
    return colors_to_imagej_lut(colors[color_index, :])


def bitmap_line(s):
    """Draw text using Visitor font (characters are 5x5 pixels).
    """
    import PIL.Image
    import PIL.ImageDraw
    img = PIL.Image.new("RGBA", (len(s) * 8, 10), (0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    draw.text((0, 0), s, (255, 255, 255), font=VISITOR_FONT)
    draw = PIL.ImageDraw.Draw(img)

    n = np.array(img)[2:7, :, 0]
    if n.sum() == 0:
        return n
    return (n[:, :np.where(n.any(axis=0))[0][-1] + 1] > 0).astype(int)


def bitmap_lines(lines, spacing=1):
    """Draw multiple lines of text from a list of strings.
    """
    bitmaps = [bitmap_line(x) for x in lines]
    height = 5
    shapes = np.array([x.shape for x in bitmaps])
    shape = (height + 1) * len(bitmaps), shapes[:, 1].max()

    output = np.zeros(shape, dtype=int)
    for i, bitmap in enumerate(bitmaps):
        start, end = i * (height + 1), (i + 1) * (height + 1) - 1
        output[start:end, :bitmap.shape[1]] = bitmap

    return output[:-1, :]


def colors_to_imagej_lut(lut_values):
    """ImageJ header expects 256 red values, then 256 green values, then 
    256 blue values.
    """
    return tuple(np.array(lut_values).T.flatten().astype(int))


def build_GRMC():
    import seaborn as sns
    colors = (0, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1)
    lut = []
    for color in colors:
        lut.append([0, 0, 0, 1])
        lut.extend(sns.dark_palette(color, n_colors=64 - 1))
    lut = np.array(lut)[:, :3]
    RGCM = np.zeros((256, 3), dtype=int)
    RGCM[:len(lut)] = (lut * 255).astype(int)
    return tuple(RGCM.T.flatten())


def add_rect_bounds(df, width=10, ij='ij', bounds_col='bounds'):
    arr = []
    for i,j in df[list(ij)].values.astype(int):
        arr.append((i - width, j - width, i + width, j + width))
    return df.assign(**{bounds_col: arr})


# BASE LABELING

colors = (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1)
GRMC = build_discrete_lut(colors)


def add_base_codes(df_reads, bases, offset, col):
    n = len(df_reads[col].iloc[0])
    df = (df_reads[col].str.extract('(.)'*n)
          .applymap(bases.index)
          .rename(columns=lambda x: 'c{0}'.format(x+1))
         )
    return pd.concat([df_reads, df + offset], axis=1)


def annotate_bases(df_reads, col='barcode', bases='GTAC', offset=1, **kwargs):
    """
    from ops.annotate import add_base_codes, GRMC
    labels = annotate_bases(df_reads)
    # labels = annotate_bases(df_cells, col='cell_barcode_0')

    data = read('process/10X_A1_Tile-7.log.tif')
    labeled = join_stacks(data, (labels[:, None], '.a'))

    luts = GRAY, GREEN, RED, MAGENTA, CYAN, GRMC 
    save('test/labeled', labeled, luts=luts)
    """
    df_reads = add_base_codes(df_reads, bases, offset, col)
    n = len(df_reads[col].iloc[0])
    cycles = ['c{0}'.format(i+1) for i in range(n)]
    labels = np.array([annotate_points(df_reads, c, **kwargs) for c in cycles])
    return labels


