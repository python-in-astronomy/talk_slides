import numpy as np
from pyasdf import AsdfFile
from matplotlib import image as mplimage
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting
from gwcs import wcs, selector
import copy


def read_model(filename):
    f = AsdfFile.read(filename)
    return f.tree['model']


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """
    import matplotlib
    from matplotlib import pyplot as plt
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


def show(mask, n_regions=3):
    from matplotlib import pyplot as plt
    c = cmap_discretize('jet', n_regions)
    labels=np.arange(n_regions)
    loc= labels
    f = plt.figure()
    #f.set_size_inches(12, 8, forward=True)
    plt.imshow(mask, interpolation='nearest', cmap=c, aspect='auto')
    cb = plt.colorbar()
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    plt.show()


def miri_mask():
    f = fits.open('MIRI_FM_LW_A_D2C_01.00.00.fits')
    slices = f[3].data
    nslices = np.zeros((slices.shape))
    nslices[:, :500] = slices[:, :500]
    sl1 = slices[:, 500:]
    sl2 = np.where(sl1> 0, sl1+12, sl1)
    #sl2 = np.where(sl1>100, sl1-100+21, sl1)
    nslices[:,500:] = sl2
    return nslices
    
def make_channel_models(channel):
    f = fits.open('MIRI_FM_LW_A_D2C_01.00.00.fits')
    if channel == 3:
        slice_mask = f[3].data[:,:500]
        b1 = f[0].header['B_DEL3']
        b0 = f[0].header['B_MIN3']

    elif channel == 4:
        slice_mask = f[3].data[:, 500:]
        b1 = f[0].header['B_DEL4']
        b0 = f[0].header['B_MIN4']

    slices = np.unique(slice_mask)
    slices = np.asarray(slices, dtype=np.int16).tolist()
    slices.remove(0)
    
    #read alpha and lambda planes from pixel maps file
    lam = f[1].data
    alpha = f[2].data
    # create a model to fit to each slice in alpha plane
    amodel = models.Chebyshev2D(x_degree=2, y_degree=1)
    # a model to be fitted to each slice in lambda plane
    lmodel = models.Chebyshev2D(x_degree=1, y_degree=1)
    lmodel.c0_1.fixed = True
    lmodel.c1_1.fixed = True
    fitter = fitting.LinearLSQFitter()
    reg_models = {}
    for sl in slices:
        #print('sl', sl)
        ind = (slice_mask==sl).nonzero()#(slice_mask[:, :500] ==sl).nonzero()
        x0 = ind[0].min()
        x1 = ind[0].max()
        y0 = ind[1].min()
        y1 = ind[1].max()
        if channel ==4:
            y0 += 500
            y1 += 500
        x, y = np.mgrid[x0:x1, y0:y1]
        sllam = lam[x0:x1, y0:y1]
        slalpha = alpha[x0:x1, y0:y1]
        lfitted = fitter(lmodel, x, y, sllam)
        afitted = fitter(amodel, x, y, slalpha)

        if channel == 4:
            beta_model = models.Const1D(b0 + b1 * (sl+12))
            reg_models[sl+12] = lfitted, afitted, beta_model, (x0, x1, y0, y1)
        else:
            beta_model = models.Const1D(b0 + b1*sl)
            reg_models[sl] = lfitted, afitted, beta_model, (x0, x1, y0, y1)

    return reg_models


def miri_models():
    # Use channels 3 and 4 for this example
    reg_models3 = make_channel_models(3)
    reg_models4 = make_channel_models(4)
    reg_models = {}

    for reg in reg_models3:
        lam_model, alpha_model, beta_model, _ = reg_models3[reg]
        model = models.Mapping((0, 1, 0, 0, 1)) | alpha_model & beta_model & lam_model
        reg_models[reg] = model
    for reg in reg_models4:
        lam_model, alpha_model, beta_model, _ = reg_models4[reg]
        model = models.Mapping((0, 1, 0, 0, 1)) | alpha_model & beta_model & lam_model
        reg_models[reg] = model
    return reg_models


