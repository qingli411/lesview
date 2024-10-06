#--------------------------------
# Plots
#--------------------------------
import numpy as np
import matplotlib.pyplot as plt

def plot_box_field(da, itime=-1, view='top', zdist=1, **kwargs):
    """Plot a field in a 3D box

    :da: (xarray.DataArray) array of a variable
    :var: (str) variable name
    :itime: (int, optional) index for time
    :view: (str, optional) view of the 3D box plot, "top" (default), "bottom", or "both"
    :zdist: (float, optional) distance from the surface and bottom boundaries in m
    :**kwargs: keyword arguments in pairs passed to plotting functions

    """

    # select time slice
    dat = da.isel(time=itime)

    # check coordinates
    if 'x' in dat.coords:
        xmin, xmax = dat.x.min(), dat.x.max()
        xc = dat.x
        isxi = False
    elif 'xi' in dat.coords:
        xmin, xmax = dat.xi.min(), dat.xi.max()
        xc = dat.xi
        isxi = True
    else:
        raise ValueError('Invalid x coordinate')
    if 'y' in dat.coords:
        ymin, ymax = dat.y.min(), dat.y.max()
        yc = dat.y
        isyi = False
    elif 'yi' in dat.coords:
        ymin, ymax = dat.yi.min(), dat.yi.max()
        yc = dat.yi
        isyi = True
    else:
        raise ValueError('Invalid y coordinate')
    if 'z' in dat.coords:
        zmin, zmax = dat.z.min(), dat.z.max()
        zc = dat.z
        iszi = False
    elif 'zi' in dat.coords:
        zmin, zmax = dat.zi.min(), dat.zi.max()
        zc = dat.zi
        iszi = True
    else:
        raise ValueError('Invalid z coordinate')

    # check view
    if view == 'top':
        ix = -1
        iy = 0
    elif view == 'bottom':
        ix = 0
        iy = -1
    elif view == 'both':
        if isxi:
            ix = int(da.coords['xi'].size/2)
        else:
            ix = int(da.coords['x'].size/2)
        iy = -1
    else:
        raise ValueError('Invalid view, should be \"top\", \"bottom\", or \"both\"')

    # select slices
    if iszi:
        dxy  = dat.sel(zi=zmax-zdist, method= "nearest")
        dxy2 = dat.sel(zi=zmin+zdist, method= "nearest")
        zoffset  = dxy.zi
        zoffset2 = dxy2.zi
    else:
        dxy  = dat.sel(z=zmax-zdist, method= "nearest")
        dxy2 = dat.sel(z=zmin+zdist, method= "nearest")
        zoffset  = dxy.z
        zoffset2 = dxy2.z
    if isyi:
        dxz = dat.isel(yi=iy)
    else:
        dxz = dat.isel(y=iy)
    if isxi:
        dyz = dat.isel(xi=ix)
    else:
        dyz = dat.isel(x=ix)

    # create plot
    ax = plt.figure(figsize=[8,6]).add_subplot(projection='3d',computed_zorder=False)
    edges_kw = dict(color='0.4', linewidth=1, zorder=10)
    if view == 'both':
        im = ax.contourf(np.tile(xc[:ix], [zc.size, 1]), dxz.data[:,:ix], np.tile(zc, [xc[:ix].size, 1]).transpose(),  zdir='y', offset=yc[0], extend='both', **kwargs)
        ax.contourf(xc[ix:], yc, dxy2.data[:,ix:], zdir='z', offset=zoffset2, extend='both', zorder=3, **kwargs)
        ax.contourf(np.tile(xc, [zc.size, 1]), dxz.data, np.tile(zc, [xc.size, 1]).transpose(), zdir='y', offset=yc[iy], extend='both', zorder=1, **kwargs)
        ax.contourf(dyz.data, np.tile(yc, [zc.size, 1]), np.tile(zc, [yc.size, 1]).transpose(), zdir='x', offset=xc[ix], extend='both', zorder=2, **kwargs)
        ax.contourf(xc[:ix], yc, dxy.data[:,:ix], zdir='z', offset=zoffset, extend='both', zorder=1,  **kwargs)
        # reference lines
        ax.plot([xmin, xc[ix]], [ymin, ymin], zmax, **edges_kw) # h
        ax.plot([xc[ix], xc[ix]], [ymin, ymax], zmax, **edges_kw) # h
        ax.plot([xc[ix], xc[ix]], [ymin, ymin], [zmin, zmax], **edges_kw) # v
        ax.plot([xc[ix], xc[ix]], [ymin, ymax], zoffset2, linestyle=':', **edges_kw) # h
        ax.plot([xc[ix], xc[ix]], [ymax, ymax], [zoffset2, zmax], linestyle=':', **edges_kw) # v
        ax.plot([xc[ix], xmax], [ymax, ymax], zoffset2, linestyle=':', **edges_kw) # h
    else:
        if view == 'top':
            im = ax.contourf(xc, yc, dxy.data, zdir='z', offset=zoffset, extend='both', **kwargs)
            # reference lines
            ax.plot([xmax, xmax], [ymin, ymax], zmax, **edges_kw) # h
            ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw) # h
            ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw) # v
        else:
            im = ax.contourf(xc, yc, dxy2.data, zdir='z', offset=zoffset2, extend='both', **kwargs)
            # reference lines
            ax.plot([xmin, xmin], [ymin, ymax], zmin, linestyle=':', **edges_kw) # h
            ax.plot([xmin, xmax], [ymax, ymax], zmin, linestyle=':', **edges_kw) # h
            ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], linestyle=':', **edges_kw) # v
        ax.contourf(np.tile(xc, [zc.size, 1]), dxz.data, np.tile(zc, [xc.size, 1]).transpose(), zdir='y', offset=yc[iy], extend='both', **kwargs)
        ax.contourf(dyz.data, np.tile(yc, [zc.size, 1]), np.tile(zc, [yc.size, 1]).transpose(), zdir='x', offset=xc[ix], extend='both', zorder=1, **kwargs)
    # other settings
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_zlabel('$z$ [m]')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    ax.set_box_aspect((1,1,0.3), zoom=1)
    ax.view_init(elev=40, azim=-40)
    ax_inset = ax.inset_axes([0.75, 0.85, 0.3, 0.03])
    cb = plt.colorbar(im, cax=ax_inset, orientation='horizontal', label='${:s}$ [{:s}]'.format(dxy.attrs['long_name'], dxy.attrs['units']), ticks=[im.levels[0], im.levels[int(im.levels.size/2)], im.levels[-1]])
    ax.text2D(0.05, 0.85, da.time.dt.strftime('%m-%d %H:%M:%S').data[itime], transform=ax.transAxes, fontsize=12, va='top', ha='left')

    return plt.gcf()
