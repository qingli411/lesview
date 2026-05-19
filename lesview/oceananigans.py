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
        timeindex=None,
        ):
    """Get sorted iteration labels

    :data:      (h5py.File) input file
    :timeindex: (int) time index
    :return:    (list of str) sorted iteration labels

    """
    davar = data['timeseries']['t']
    iters = []
    for it in davar.keys():
        iters.append(int(it))
    iters.sort()
    iters_out = ['{}'.format(s) for s in iters]
    if timeindex is None:
        return iters_out
    else:
        return [iters_out[timeindex]]
    return iters_out

def get_time(
        data,
        iters=None,
        origin='2000-01-01T00:00:00',
        hours=None,

        ):
    """Get the time dimension

    :data:   (h5py.File) input file
    :iters:  (list of str) sorted iteration labels
    :origin: (scalar) reference date passed to pandas.to_datetime()
    :hours:  (bool or scalar) use hours since some reference time as the time coordinate
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
    if hours is None or hours is False:
        return time
    else:
        if hours is True:
            reftime = origin
        else:
            reftime = hours
        timediff = time - pd.to_datetime(reftime)
        d = timediff/pd.Timedelta('1 hour')
        tnew = xr.DataArray(
                d,
                dims=('time'),
                coords={'time': np.arange(d.size)},
                attrs={'long_name': 'time', 'units': 'hours since {:s}'.format(reftime)})
        return tnew

def get_grid(
        data,
        in_name,
        halo_name,
        out_name,
        out_units,
        in_subname=None,
        ):
    """Get the spatial dimension

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) dim

    """
    H = data['grid'][halo_name][()]
    if in_subname is None:
        d = data['grid'][in_name][()][H:-H]
    else:
        d = data['grid'][in_name][in_subname][()][H:-H]
    d = xr.DataArray(
        d,
        dims=(out_name),
        coords={out_name: np.arange(d.size)},
        attrs={'long_name': out_name, 'units': out_units})
    return d, H

def get_x(
        data,
        latlon=False,
        ):
    """Get the x dimension (grid centers)

    :data:   (h5py.File) input file
    :latlon: (bool) Latitude-Longitude grid
    :return: (xarray.DataArray) x

    """
    if latlon:
        return get_grid(data, 'λᶜᵃᵃ', 'Hx', 'lon', 'degree_east')
    else:
        return get_grid(data, 'xᶜᵃᵃ', 'Hx', 'x', 'm')

def get_xi(
        data,
        latlon=False,
        ):
    """Get the x dimension (grid interfaces)

    :data:   (h5py.File) input file
    :latlon: (bool) Latitude-Longitude grid
    :return: (xarray.DataArray) x

    """
    if latlon:
        return get_grid(data, 'λᶠᵃᵃ', 'Hx', 'loni', 'degree_east')
    else:
        return get_grid(data, 'xᶠᵃᵃ', 'Hx', 'xi', 'm')

def get_y(
        data,
        latlon=False,
        ):
    """Get the y dimension (grid centers)

    :data:   (h5py.File) input file
    :latlon: (bool) Latitude-Longitude grid
    :return: (xarray.DataArray) y

    """
    if latlon:
        return get_grid(data, 'φᵃᶜᵃ', 'Hy', 'lat', 'degree_north')
    else:
        return get_grid(data, 'yᵃᶜᵃ', 'Hy', 'y', 'm')

def get_yi(
        data,
        latlon=False,
        ):
    """Get the y dimension (grid interfaces)

    :data:   (h5py.File) input file
    :latlon: (bool) Latitude-Longitude grid
    :return: (xarray.DataArray) y

    """
    if latlon:
        return get_grid(data, 'φᵃᶠᵃ', 'Hy', 'lati', 'degree_north')
    else:
        return get_grid(data, 'yᵃᶠᵃ', 'Hy', 'yi', 'm')

def get_z(
        data,
        ):
    """Get the z dimension (grid centers)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) z

    """
    return get_grid(data, 'z', 'Hz', 'z', 'm', 'cᵃᵃᶜ')

def get_zi(
        data,
        ):
    """Get the z dimension (grid interfaces)

    :data:   (h5py.File) input file
    :return: (xarray.DataArray) z

    """
    return get_grid(data, 'z', 'Hz', 'zi', 'm', 'cᵃᵃᶠ')

def var_longname(
        varname,
        ):
    """Return the long name of a variable in Oceananigans

    :varname: (str) variable name
    :return:  (str) unit

    """
    var_longname = {
            'T':    r'$T$',
            'S':    r'psu',
            'η':    r'$\eta$',
            'b':    r'$b$',
            'B':    r'$\overline{b}$',
            'bb':   r'$\overline{{b^\prime}^2}$',
            'tt':   r'$\overline{{T^\prime}^2}$',
            'e':    r'$e$',
            'p':    r'$p$',
            'u':    r'$u$',
            'U':    r'$\overline{u}$',
            'ub':   r'$\overline{u^\prime b^\prime}$',
            'uu':   r'$\overline{{u^\prime}^2}$',
            'uv':   r'$\overline{u^\prime v^\prime}$',
            'v':    r'$v$',
            'V':    r'$\overline{v}$',
            'vb':   r'$\overline{v^prime b^\prime}$',
            'vv':   r'$\overline{{v^\prime}^2}$',
            'w':    r'$w$',
            'W':    r'$\overline{w}$',
            'wb':   r'$\overline{w^\prime b^\prime}$',
            'wt':   r'$\overline{w^\prime T^\prime}$',
            'ws':   r'$\overline{w^\prime S^\prime}$',
            'wu':   r'$\overline{u^\prime w^\prime}$',
            'wv':   r'$\overline{v^\prime w^\prime}$',
            'ww':   r'$\overline{{w^\prime}^2}$',
            'ubsb': r'$\overline{u^\prime b^\prime}_{sgs}$',
            'vbsb': r'$\overline{v^\prime b^\prime}_{sgs}$',
            'wbsb': r'$\overline{w^\prime b^\prime}_{sgs}$',
            'wtsb': r'$\overline{w^\prime T^\prime}_{sgs}$',
            'wssb': r'$\overline{w^\prime S^\prime}_{sgs}$',
            'uvsb': r'$\overline{u^\prime v^\prime}_{sgs}$',
            'wusb': r'$\overline{u^\prime w^\prime}_{sgs}$',
            'wvsb': r'$\overline{v^\prime w^\prime}_{sgs}$',
            'w3':   r'$\overline{{w^\prime}^3}$',
            'νₑ':   r'$\\nu_e$',
            'tke_advective_flux':   'TKE advective flux',
            'tke_buoyancy_flux':    'TKE buoyancy flux',
            'tke_dissipation':      'TKE dissipation',
            'tke_pressure_flux':    'TKE pressure flux',
            'tke_shear_production': 'TKE shear producton',
            'wbsc': r'$\overline{w^\prime b^\prime}_{sc}$',
            'uusc': r'$\overline{{u^\prime}^2}_{sc}$',
            'vvsc': r'$\overline{{u^\prime}^2}_{sc}$',
            'wwsc': r'$\overline{{u^\prime}^2}_{sc}$',
            'uvsc': r'$\overline{u^\prime v^\prime}_{sc}$',
            'wusc': r'$\overline{u^\prime w^\prime}_{sc}$',
            'wvsc': r'$\overline{v^\prime w^\prime}_{sc}$',
            'w3sc': r'$\overline{{u^\prime}^3}_{sc}$',
            }
    if varname in var_longname.keys():
        return var_longname[varname]
    else:
        print('The long name of \'{:s}\' is not defined. Set to \'none\'.'.format(varname))
        return 'none'

def var_units(
        varname,
        ):
    """Return the unit of a variable in Oceananigans

    :varname: (str) variable name
    :return:  (str) unit

    """
    var_units = {
            'T':    r'$^\circ$C',
            'S':    r'psu',
            'η':    r'm',
            'b':    r'm/s$^2$',
            'B':    r'm/s$^2$',
            'bb':   r'm$^2$/s$^4$',
            'tt':   r'$^\circ$C$^2$',
            'e':    r'm$^2$/s$^2$',
            'p':    r'm$^2$/s$^2$',
            'u':    r'm/s',
            'U':    r'm/s',
            'ub':   r'm$^2$/s$^3$',
            'uu':   r'm$^2$/s$^2$',
            'uv':   r'm$^2$/s$^2$',
            'v':    r'm/s',
            'V':    r'm/s',
            'vb':   r'm$^2$/s$^3$',
            'vv':   r'm$^2$/s$^2$',
            'w':    r'm/s',
            'W':    r'm/s',
            'wb':   r'm$^2$/s$^3$',
            'wt':   r'm/s $^\circ$C',
            'ws':   r'm/s psu',
            'wu':   r'm$^2$/s$^2$',
            'wv':   r'm$^2$/s$^2$',
            'ww':   r'm$^2$/s$^2$',
            'ubsb': r'm$^2$/s$^3$',
            'vbsb': r'm$^2$/s$^3$',
            'wbsb': r'm$^2$/s$^3$',
            'wtsb': r'm/s $^\circ$C',
            'wssb': r'm/s psu',
            'uvsb': r'm$^2$/s$^2$',
            'wusb': r'm$^2$/s$^2$',
            'wvsb': r'm$^2$/s$^2$',
            'w3':   r'm$^3$/s$^3$',
            'νₑ':   r'm$^2$/s',
            'tke_advective_flux':   r'm$^3$/s$^3$',
            'tke_buoyancy_flux':    r'm$^2$/s$^3$',
            'tke_dissipation':      r'm$^2$/s$^3$',
            'tke_pressure_flux':    r'm$^3$/s$^3$',
            'tke_shear_production': r'm$^2$/s$^3$',
            'wbsc': r'm$^2$/s$^3$',
            'uusc': r'm$^2$/s$^2$',
            'vvsc': r'm$^2$/s$^2$',
            'wwsc': r'm$^2$/s$^2$',
            'uvsc': r'm$^2$/s$^2$',
            'wusc': r'm$^2$/s$^2$',
            'wvsc': r'm$^2$/s$^2$',
            'w3sc': r'm$^3$/s$^3$',
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
            hoursince = None,
            ):
        """Initialization

        :filepath:        (str) path of the Oceananigans profile data file
        :datetime_origin: (scalar) reference date passed to pandas.to_datetime()
        :hoursince:       (bool or scalar) use hours since some reference time as the time coordinate

        """
        super(OceananigansDataProfile, self).__init__(filepath)
        self._datetime_origin = datetime_origin
        self._hoursince = hoursince
        self.dataset = self._load_dataset()

    def _load_dataset(
            self,
            ):
        """Load data set

        :return: (xarray.Dataset) data set

        """
        # function for loading one variable
        def get_vertical_profile(data, varname, time, z, Hz, iters):
            davar = data['timeseries'][varname]
            nt = len(iters)
            nz = z.size
            var_arr = np.zeros([nz, nt])
            for i, it in enumerate(iters):
                if it in davar.keys():
                    var_arr[:,i] = davar[it][()][Hz:-Hz].flatten()
                else:
                    var_arr[:,i] = np.nan
            var = xr.DataArray(
                var_arr,
                dims=(z.dims[0], 'time'),
                coords={z.dims[0]: z, 'time': time},
                attrs={'long_name': var_longname(varname), 'units': var_units(varname)})
            return var
        # load all variables into an xarray.Dataset
        with h5py.File(self._filepath, 'r') as fdata:
            iters_sorted = get_iters_sorted(fdata)
            time = get_time(fdata, iters=iters_sorted, origin=self._datetime_origin, hours=self._hoursince)
            gz, Hz = get_z(fdata)
            gzi, _ = get_zi(fdata)
            gnz  = gz.size + 2*Hz
            gnzi = gzi.size + 2*Hz
            # define output dataset
            out = xr.Dataset()
            for varname in fdata['timeseries'].keys():
                davar = fdata['timeseries'][varname]
                tlist = list(davar.keys())
                ndvar = davar[tlist[0]][()].size
                if ndvar == gnz:
                    z = gz
                elif ndvar == gnzi:
                    z = gzi
                else:
                    print('Variable \'{:}\' has dimension {:d}. Skipping.'.format(varname, ndvar))
                    continue
                out[varname] = get_vertical_profile(fdata, varname, time, z, Hz, iters_sorted)
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
            hoursince = None,
            latlon = False,
            fieldname = None,
            timeindex = None,
            ):
        """Initialization

        :filepath:        (str) path of the Oceananigans volume data file
        :datetime_origin: (scalar) reference date passed to pandas.to_datetime()
        :hoursince:       (bool or scalar) use hours since some reference time as the time coordinate
        :latlon:          (bool) Latitude-Longitude grid
        :fieldname:       (str) name of field to load if given
        :timeindex:       (int) time index to load if given

        """
        super(OceananigansDataVolume, self).__init__(filepath)
        self._datetime_origin = datetime_origin
        self._hoursince = hoursince
        self._latlon = latlon
        self.dataset = self._load_dataset(fieldname, timeindex)

    def _load_dataset(
            self,
            fieldname,
            timeindex
            ):
        """Load data set

        :fieldname: (str) name of field to load if given
        :timeindex: (int) time index to load if given
        :return: (xarray.Dataset) data set

        """
        # function for loading one variable
        def get_volume(data, varname, time, x, y, z, Hx, Hy, Hz, iters):
            davar = data['timeseries'][varname]
            nt = len(iters)
            tlist = list(davar.keys())
            nz, ny, nx = davar[tlist[0]][()].shape
            izs = iys = ixs = 0
            ize = iye = ixe = 1
            if nz != 1:
                izs = Hz
                ize = -Hz
                nz -= 2*Hz
            if ny != 1:
                iys = Hy
                iye = -Hy
                ny -= 2*Hy
            if nx != 1:
                ixs = Hx
                ixe = -Hx
                nx -= 2*Hx
            var_arr = np.zeros([nz, ny, nx, nt])
            for i, it in enumerate(iters):
                var_arr[:,:,:,i] = davar[it][()][izs:ize, iys:iye, ixs:ixe]
            var = xr.DataArray(
                var_arr,
                dims=(z.dims[0], y.dims[0], x.dims[0], 'time'),
                coords={z.dims[0]: z, y.dims[0]: y, x.dims[0]: x, 'time': time},
                attrs={'long_name': var_longname(varname), 'units': var_units(varname)})
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
            iters_sorted = get_iters_sorted(fdata, timeindex=timeindex)
            time = get_time(fdata, iters=iters_sorted, origin=self._datetime_origin, hours=self._hoursince)
            gx, Hx = get_x(fdata, latlon=self._latlon)
            gxi, _ = get_xi(fdata, latlon=self._latlon)
            gnx  = gx.size
            gnxi = gxi.size
            gy, Hy = get_y(fdata, latlon=self._latlon)
            gyi, _ = get_yi(fdata, latlon=self._latlon)
            gny  = gy.size
            gnyi = gyi.size
            gz, Hz = get_z(fdata)
            gzi, _ = get_zi(fdata)
            gnz  = gz.size
            gnzi = gzi.size
            if gnx != 1:
                gnx  += 2*Hx
                gnxi += 2*Hx
            if gny != 1:
                gny  += 2*Hy
                gnyi +=+ 2*Hy
            if gnz != 1:
                gnz  += 2*Hz
                gnzi += 2*Hz
            # define output dataset
            out = xr.Dataset()
            if fieldname is None:
                vlist = fdata['timeseries'].keys()
            else:
                vlist = list(fieldname)
            for varname in vlist:
                davar = fdata['timeseries'][varname]
                tlist = list(davar.keys())
                ndvar = davar[tlist[0]][()].shape
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
                elif ny == gny or ny == gny+1:
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
                da = get_volume(fdata, varname, time, x, y, z, Hx, Hy, Hz, iters_sorted)
                for dname in ['zslice', 'yslice', 'xslice']:
                    if dname in da.coords:
                        da = da.squeeze(dname, drop=True)
                out[varname] = da
        return out


