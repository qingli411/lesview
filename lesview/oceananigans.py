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

def get_zu(
        data,
        ):
    """Get the z dimension (grid centers)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) z

    """
    Lz = data['grid']['Lz'][()]
    z = data['grid']['zC'][()]
    z = z[ (z>=-Lz) & (z<=0.) ]
    z = xr.DataArray(
        z,
        dims=('z'),
        coords={'z': np.arange(z.size)},
        attrs={'long_name': 'z', 'units': 'm'})
    return z

def get_zw(
        data,
        ):
    """Get the z dimension (grid interfaces)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) z

    """
    Lz = data['grid']['Lz'][()]
    z = data['grid']['zF'][()]
    z = z[ (z>=-Lz) & (z<=0.) ]
    z = xr.DataArray(
        z,
        dims=('zi'),
        coords={'zi': np.arange(z.size)},
        attrs={'long_name': 'zi', 'units': 'm'})
    return z

def var_units(
        varname,
        ):
    """Return the unit of a variable in Oceananigans

    :varname: (str) variable name
    :return:  (str) unit

    """
    var_units = {
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
            'wb':   'm$^2$/s$^3$',
            'wu':   'm$^2$/s$^2$',
            'wv':   'm$^2$/s$^2$',
            'ww':   'm$^2$/s$^2$',
            'tke_advective_flux':   'm$^3$/s$^3$',
            'tke_buoyancy_flux':    'm$^2$/s$^3$',
            'tke_dissipation':      'm$^2$/s$^3$',
            'tke_pressure_flux':    'm$^3$/s$^3$',
            'tke_shear_production': 'm$^2$/s$^3$',
            }
    return var_units[varname]

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
            zu = get_zu(fdata)
            zw = get_zw(fdata)
            nzu = zu.size
            nzw = zw.size
            print(zu)
            print(zw)
            # define output dataset
            out = xr.Dataset()
            for varname in fdata['timeseries'].keys():
                ndvar = fdata['timeseries'][varname]['0'][()].size
                if ndvar == nzu:
                    out[varname] = get_vertical_profile(fdata, varname, time, zu, iters_sorted)
                elif ndvar == nzw:
                    out[varname] = get_vertical_profile(fdata, varname, time, zw, iters_sorted)
                else:
                    print('Variable \'{:}\' has dimension {:d}. Skipping.'.format(varname, ndvar))
        return out

