#--------------------------------
# Plots
#--------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def _check_x_coordinate(dat):

    if 'x' in dat.coords:
        return dat.x.min(), dat.x.max(), dat.x, False
    elif 'xi' in dat.coords:
        return dat.xi.min(), dat.xi.max(), dat.xi, True
    else:
        raise ValueError('Invalid x coordinate')

def _check_y_coordinate(dat):

    if 'y' in dat.coords:
        return dat.y.min(), dat.y.max(), dat.y, False
    elif 'yi' in dat.coords:
        return dat.yi.min(), dat.yi.max(), dat.yi, True
    else:
        raise ValueError('Invalid y coordinate')

def _check_z_coordinate(dat):

    if 'z' in dat.coords:
        return dat.z.min(), dat.z.max(), dat.z, False
    elif 'zi' in dat.coords:
        return dat.zi.min(), dat.zi.max(), dat.zi, True
    else:
        raise ValueError('Invalid z coordinate')

def _check_view(view, nx):

    if view == 'top':
        return -1, 0
    elif view == 'bottom':
        return 0, -1
    elif view == 'both':
        return int(nx/2), -1
    else:
        raise ValueError('Invalid view, should be \"top\", \"bottom\", or \"both\"')

def _plot_box(dxy, dxy2, dxz, dyz, view,
              xc, xmin, xmax, ix, yc, ymin, ymax, iy,
              zc, zmin, zmax, zoffset, zoffset2, timestr,
              ax, add_colorbar, add_timestamp, **kwargs):

    if ax is None:
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
    if add_colorbar:
        ax_inset = ax.inset_axes([0.75, 0.85, 0.3, 0.03])
        cb = plt.colorbar(im, cax=ax_inset, orientation='horizontal', label='{:s} [{:s}]'.format(dxz.attrs['long_name'], dxz.attrs['units']), ticks=[im.levels[0], im.levels[int(im.levels.size/2)], im.levels[-1]])
    if add_timestamp:
        ax.text2D(0.05, 0.85, timestr, transform=ax.transAxes, fontsize=12, va='top', ha='left')

    return im

def plot_box_field(
        da,
        itime=-1,
        view='top',
        zdist=1,
        ax=None,
        add_colorbar=True,
        add_timestamp=True,
        **kwargs):
    """Plot a field in a 3D box

    :da: (xarray.DataArray) array of a variable
    :var: (str) variable name
    :itime: (int, optional) index for time
    :view: (str, optional) view of the 3D box plot, "top" (default), "bottom", or "both"
    :zdist: (float, optional) distance from the surface and bottom boundaries in m
    :ax: (matplotlib.axes, optional) axis to plot figure on
    :add_colorbar: (bool, optional) flag to turn on colorbar
    :add_timestamp: (bool, optional) flag to add timestamp
    :**kwargs: keyword arguments in pairs passed to plotting functions

    """
    # select time slice
    dat = da.isel(time=itime)
    timestr = da.time.dt.strftime('%m-%d %H:%M:%S').data[itime]

    # check coordinates
    xmin, xmax, xc, isxi = _check_x_coordinate(dat)
    ymin, ymax, yc, isyi = _check_y_coordinate(dat)
    zmin, zmax, zc, iszi = _check_z_coordinate(dat)

    # check view
    ix, iy = _check_view(view, xc.size)

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

    return _plot_box(dxy, dxy2, dxz, dyz, view,
                  xc, xmin, xmax, ix, yc, ymin, ymax, iy,
                  zc, zmin, zmax, zoffset, zoffset2, timestr,
                  ax=ax, add_colorbar=add_colorbar, add_timestamp=add_timestamp, **kwargs)

def plot_box_slice(
        da_xz,
        da_yz,
        da_xy,
        iz,
        da_xy2=None,
        iz2=None,
        itime=-1,
        view='top',
        ax=None,
        add_colorbar=True,
        add_timestamp=True,
        **kwargs):
    """Plot slices in a 3D box

    :da_xz: (xarray.DataArray) xz slice of a variable
    :da_yz: (xarray.DataArray) yz slice of a variable
    :da_xy: (xarray.DataArray) xy slice of a variable
    :iz: (int) z index for the xy slice
    :da_xy2: (xarray.DataArray, optional) additional xy slice of a variable
    :iz2: (int, optional) z index for the additional xy slice
    :itime: (int, optional) index for time
    :view: (str, optional) view of the 3D box plot, "top" (default), "bottom", or "both"
    :ax: (matplotlib.axes, optional) axis to plot figure on
    :add_colorbar: (bool, optional) flag to turn on colorbar
    :add_timestamp: (bool, optional) flag to add timestamp
    :**kwargs: keyword arguments in pairs passed to plotting functions

    """
    # select time slice
    if view == 'both':
        if da_xy2 is None or iz2 is None:
            raise ValueError('Both \"da_xy2\" and \"iz2\" are required when view=\"both\".')
        else:
            dxy  = da_xy.isel(time=itime)
            dxy2 = da_xy2.isel(time=itime)
            izoffset  = iz
            izoffset2 = iz2
    elif view == 'top':
        dxy  = da_xy.isel(time=itime)
        dxy2 = None
        izoffset  = iz
        izoffset2 = -1
    elif view == 'bottom':
        dxy2 = da_xy.isel(time=itime)
        dxy  = None
        izoffset2 = iz
        izoffset  = -1
    else:
        raise ValueError('Invalid view, should be \"top\", \"bottom\", or \"both\"')

    dxz  = da_xz.isel(time=itime)
    dyz  = da_yz.isel(time=itime)
    timestr = da_xz.time.dt.strftime('%m-%d %H:%M:%S').data[itime]

    # check coordinates
    xmin, xmax, xc, isxi = _check_x_coordinate(dxz)
    ymin, ymax, yc, isyi = _check_y_coordinate(dyz)
    zmin, zmax, zc, iszi = _check_z_coordinate(dxz)

    # check view
    ix, iy = _check_view(view, xc.size)

    # zoffset
    zoffset  = zc[izoffset]
    zoffset2 = zc[izoffset2]

    # create plot

    return _plot_box(dxy, dxy2, dxz, dyz, view,
                  xc, xmin, xmax, ix, yc, ymin, ymax, iy,
                  zc, zmin, zmax, zoffset, zoffset2, timestr,
                  ax=ax, add_colorbar=add_colorbar, add_timestamp=add_timestamp, **kwargs)

def plot_overview_average(
    das,
    labels,
    levels,
    tavgs):
    """Plot an overview of the horizontally averaged variables

    :das: (dict) data array
    :labels: (dict) labels for each variable in the data array
    :levels: (dict) levels for each variable in the data array
    :tavgs: (dict) time slices for average profiles

    Example:

    tavg1 = dict(starttime='2000-01-01T07:00:00', endtime='2000-01-01T09:00:00', line_kw=dict(color='k', linestyle='--'))
    tavg2 = dict(starttime='2000-01-01T12:00:00', endtime='2000-01-01T14:00:00', line_kw=dict(color='k', linestyle='-'))
    das = dict(
        u = ds.data_vars['uxym']/ustar,
        v = ds.data_vars['vxym']/ustar,
        dudz = ds.data_vars['uxym'].differentiate(coord='z')*hb/ustar,
        dvdz = ds.data_vars['vxym'].differentiate(coord='z')*hb/ustar,
    )
    labels = dict(
        u = '$u/u_*$',
        v = '$v/u_*$',
        dudz = '$\partial_z u h_b/u_*$',
        dvdz = '$\partial_z v h_b/u_*$',
    )
    levels = dict(
        u = np.linspace(-10, 10, 41),
        v = np.linspace(-10, 10, 41),
        dudz = np.linspace(-10, 10, 41),
        dvdz = np.linspace(-10, 10, 41),
    )
    fig = plot_overview(das, labels, levels, tavgs)

    """
    nv = len(das)
    fig, axarr = plt.subplots(nv, 2, gridspec_kw={'width_ratios': [1, 5]})
    fig.set_size_inches([8, 0.4+2*nv])
    rlcolor = {'RdBu_r': 'k', 'viridis': 'w'}
    date_form = DateFormatter("%d-%H")
    for i, var in enumerate(das.keys()):
        ax = np.ravel(axarr)[i*2+1]
        cf = das[var].plot(ax=ax, levels=levels[var], cbar_kwargs={'label': labels[var]})
        cmap = cf.get_cmap().name
        for j, tag in enumerate(tavgs.keys()):
            ax.axvline(x=pd.Timestamp(tavgs[tag]['starttime']), linestyle=':', color=rlcolor[cmap])
            ax.axvline(x=pd.Timestamp(tavgs[tag]['endtime']), linestyle=':', color=rlcolor[cmap])
            ax.text(pd.Timestamp(tavgs[tag]['starttime']), 0, tag, va='bottom', ha='left')
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.xaxis.set_major_formatter(date_form)
        for lb in ax.get_xticklabels(which='major'):
            lb.set(rotation=30, horizontalalignment='right')
        ax = np.ravel(axarr)[i*2+0]
        for j, tag in enumerate(tavgs.keys()):
            tslice = slice(tavgs[tag]['starttime'], tavgs[tag]['endtime'])
            das[var].sel(time=tslice).mean(dim='time').plot(ax=ax, y=das[var].dims[0], label=tag, **tavgs[tag]['line_kw'])
        ax.set_xlabel(labels[var])
        ax.set_ylabel('Depth [m]')
    np.ravel(axarr)[0].legend(loc='lower right')

    plt.tight_layout()
    return fig
