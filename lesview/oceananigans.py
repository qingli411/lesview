#--------------------------------
# Data type for Oceananigans
#--------------------------------

from .data import LESData
import h5py
import numpy as np
import xarray as xr
import pandas as pd

#--------------------------------
# Shared functions
#--------------------------------

def get_iters_sorted(
        data,
        ):
    """Get sorted iteration labels

    :data:   (h5py.File) input file
    :return: (list of str) sorted iteration labels

    """
    davar = data['timeseries']['t']
    iters = []
    for it in davar.keys():
        iters.append(int(it))
    iters.sort()
    iters_out = ['{}'.format(s) for s in iters]
    return iters_out

def get_time(
        data,
        iters=None,
        origin='2000-01-01T00:00:00',
        ):
    """Get the time dimension

    :data:   (h5py.File) input file
    :iters:  (list of str) sorted iteration labels
    :origin: (scalar) reference date passed to pandas.to_datetime()
    :return: (datetime64) time

    """
    if iters is None:
        iters = get_iters_sorted(data)
    davar = data['timeseries']['t']
    nt = len(iters)
    time_arr = np.zeros(nt)
    for i, it in enumerate(iters):
        time_arr[i] = davar[it][()]
    time = pd.to_datetime(time_arr, unit='s', origin=origin)
    return time

def get_grid(
        data,
        in_name,
        halo_name,
        out_name,
        out_units,
        ):
    """Get the spatial dimension

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) dim

    """
    H = data['grid'][halo_name][()]
    d = data['grid'][in_name][()][H:-H]
    d = xr.DataArray(
        d,
        dims=(out_name),
        coords={out_name: np.arange(d.size)},
        attrs={'long_name': out_name, 'units': out_units})
    return d

def get_x(
        data,
        ):
    """Get the x dimension (grid centers)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) x

    """
    return get_grid(data, 'xᶜᵃᵃ', 'Hx', 'x', 'm')

def get_xi(
        data,
        ):
    """Get the x dimension (grid interfaces)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) x

    """
    return get_grid(data, 'xᶠᵃᵃ', 'Hx', 'xi', 'm')

def get_y(
        data,
        ):
    """Get the y dimension (grid centers)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) y

    """
    return get_grid(data, 'yᵃᶜᵃ', 'Hy', 'y', 'm')

def get_yi(
        data,
        ):
    """Get the y dimension (grid interfaces)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) y

    """
    return get_grid(data, 'yᵃᶠᵃ', 'Hy', 'yi', 'm')

def get_z(
        data,
        ):
    """Get the z dimension (grid centers)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) z

    """
    return get_grid(data, 'zᵃᵃᶜ', 'Hz', 'z', 'm')

def get_zi(
        data,
        ):
    """Get the z dimension (grid interfaces)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) z

    """
    return get_grid(data, 'zᵃᵃᶠ', 'Hz', 'zi', 'm')

def var_units(
        varname,
        ):
    """Return the unit of a variable in Oceananigans

    :varname: (str) variable name
    :return:  (str) unit

    """
    var_units = {
            'T':    '$^\circ$C',
            'S':    'psu',
            'b':    'm/s$^2$',
            'bb':   'm$^2$/s$^4$',
            'e':    'm$^2$/s$^2$',
            'p':    'm$^2$/s$^2$',
            'u':    'm/s',
            'ub':   'm$^2$/s$^3$',
            'uu':   'm$^2$/s$^2$',
            'uv':   'm$^2$/s$^2$',
            'v':    'm/s',
            'vb':   'm$^2$/s$^3$',
            'vv':   'm$^2$/s$^2$',
            'w':    'm/s',
            'wb':   'm$^2$/s$^3$',
            'wt':   'm/s $^\circ$C',
            'ws':   'm/s psu',
            'wu':   'm$^2$/s$^2$',
            'wv':   'm$^2$/s$^2$',
            'ww':   'm$^2$/s$^2$',
            'wu':   'm$^2$/s$^2$',
            'νₑ':   'm$^2$/s',
            'tke_advective_flux':   'm$^3$/s$^3$',
            'tke_buoyancy_flux':    'm$^2$/s$^3$',
            'tke_dissipation':      'm$^2$/s$^3$',
            'tke_pressure_flux':    'm$^3$/s$^3$',
            'tke_shear_production': 'm$^2$/s$^3$',
            }
    if varname in var_units.keys():
        return var_units[varname]
    else:
        print('The unit of \'{:s}\' is not defined. Set to \'none\'.'.format(varname))
        return 'none'

#--------------------------------
# OceananigansDataProfile
#--------------------------------

class OceananigansDataProfile(LESData):

    """A data type for Oceananigans vertical profile data

    """

    def __init__(
            self,
            filepath = '',
            datetime_origin = '2000-01-01T00:00:00',
            ):
        """Initialization

        :filepath:        (str) path of the Oceananigans profile data file
        :datetime_origin: (scalar) reference date passed to pandas.to_datetime()

        """
        super(OceananigansDataProfile, self).__init__(filepath)
        self._datetime_origin = datetime_origin
        self.dataset = self._load_dataset()

    def _load_dataset(
            self,
            ):
        """Load data set

        :return: (xarray.Dataset) data set

        """
        # function for loading one variable
        def get_vertical_profile(data, varname, time, z, iters):
            davar = data['timeseries'][varname]
            nt = len(iters)
            nz = z.size
            var_arr = np.zeros([nz, nt])
            for i, it in enumerate(iters):
                var_arr[:,i] = davar[it][()].flatten()
            var = xr.DataArray(
                var_arr,
                dims=(z.dims[0], 'time'),
                coords={z.dims[0]: z, 'time': time},
                attrs={'long_name': varname, 'units': var_units(varname)})
            return var
        # load all variables into an xarray.Dataset
        with h5py.File(self._filepath, 'r') as fdata:
            iters_sorted = get_iters_sorted(fdata)
            time = get_time(fdata, iters=iters_sorted, origin=self._datetime_origin)
            gz   = get_z(fdata)
            gzi  = get_zi(fdata)
            gnz  = gz.size
            gnzi = gzi.size
            # define output dataset
            out = xr.Dataset()
            for varname in fdata['timeseries'].keys():
                ndvar = fdata['timeseries'][varname]['0'][()].size
                if ndvar == gnz:
                    z = gz
                elif ndvar == gnzi:
                    z = gzi
                else:
                    print('Variable \'{:}\' has dimension {:d}. Skipping.'.format(varname, ndvar))
                    continue
                out[varname] = get_vertical_profile(fdata, varname, time, z, iters_sorted)
        return out

#--------------------------------
# OceananigansDataVolume
#--------------------------------

class OceananigansDataVolume(LESData):

    """A data type for Oceananigans volume data

    """

    def __init__(
            self,
            filepath = '',
            datetime_origin = '2000-01-01T00:00:00',
            ):
        """Initialization

        :filepath:        (str) path of the Oceananigans volume data file
        :datetime_origin: (scalar) reference date passed to pandas.to_datetime()

        """
        super(OceananigansDataVolume, self).__init__(filepath)
        self._datetime_origin = datetime_origin
        self.dataset = self._load_dataset()

    def _load_dataset(
            self,
            ):
        """Load data set

        :return: (xarray.Dataset) data set

        """
        # function for loading one variable
        def get_volume(data, varname, time, x, y, z, iters):
            davar = data['timeseries'][varname]
            nt = len(iters)
            nz, ny, nx = davar['0'][()].shape
            var_arr = np.zeros([nz, ny, nx, nt])
            for i, it in enumerate(iters):
                var_arr[:,:,:,i] = davar[it][()]
            var = xr.DataArray(
                var_arr,
                dims=(z.dims[0], y.dims[0], x.dims[0], 'time'),
                coords={z.dims[0]: z, y.dims[0]: y, x.dims[0]: x, 'time': time},
                attrs={'long_name': varname, 'units': var_units(varname)})
            return var
        # define the dimension for a slice
        def get_dim_slice(dat, dimname, dimunits):
            d = xr.DataArray(
                dat,
                dims=(dimname),
                coords={dimname: dat},
                attrs={'long_name': dimname, 'units': dimunits})
            return d
        # load all variables into an xarray.Dataset
        with h5py.File(self._filepath, 'r') as fdata:
            iters_sorted = get_iters_sorted(fdata)
            time = get_time(fdata, iters=iters_sorted, origin=self._datetime_origin)
            gx   = get_x(fdata)
            gxi  = get_xi(fdata)
            gnx  = gx.size
            gnxi = gxi.size
            gy   = get_y(fdata)
            gyi  = get_yi(fdata)
            gny  = gy.size
            gnyi = gyi.size
            gz   = get_z(fdata)
            gzi  = get_zi(fdata)
            gnz  = gz.size
            gnzi = gzi.size
            # define output dataset
            out = xr.Dataset()
            for varname in fdata['timeseries'].keys():
                ndvar = fdata['timeseries'][varname]['0'][()].shape
                if len(ndvar) != 3:
                    print('Variable \'{:}\' has dimension {}. Skipping.'.format(varname, ndvar))
                    continue
                nz, ny, nx = ndvar
                if nz == 1:
                    z = get_dim_slice(np.array([0]), 'zslice', 'm' )
                elif nz == gnz:
                    z = gz
                elif nz == gnzi:
                    z = gzi
                else:
                    raise ValueError('Invalid z coordinate')
                if ny == 1:
                    y = get_dim_slice(np.array([0]), 'yslice', 'm' )
                elif ny == gny:
                    if varname in ['v']:
                        y = gyi
                    else:
                        y = gy
                else:
                    raise ValueError('Invalid y coordinate')
                if nx == 1:
                    x = get_dim_slice(np.array([0]), 'xslice', 'm' )
                elif nx == gnx:
                    if varname in ['u']:
                        x = gxi
                    else:
                        x = gx
                else:
                    raise ValueError('Invalid x coordinate')
                da = get_volume(fdata, varname, time, x, y, z, iters_sorted)
                for dname in ['zslice', 'yslice', 'xslice']:
                    if dname in da.coords:
                        da = da.drop(labels=dname)
                out[varname] = da
        return out


