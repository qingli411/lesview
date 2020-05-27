#--------------------------------
# Data types
#--------------------------------

import xarray as xr
import numpy as np
import pandas as pd

#--------------------------------
# LESData
#--------------------------------

class LESData:

    """The common data type

    """

    def __init__(
            self,
            filepath = '',
            name = '',
            ):
        """Initialization

        :filepath:   (str) path of the LES data file
        :name:       (str) name of the LES data

        """
        self._filepath = filepath
        self._name = name

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+':']
        summary.append('{:>12s}: {:s}'.format('name', self._name))
        summary.append('{:>12s}: {:s}'.format('data path', self._filepath))
        return '\n'.join(summary)

#--------------------------------
# PALMDataProfile
#--------------------------------

class PALMDataProfile(LESData):

    """A data type for PALM vertical profile data
       See https://palm.muk.uni-hannover.de/trac/wiki/doc/app/iofiles#DATA_1D_PR_NETCDF

    """

    def __init__(
            self,
            filepath = '',
            year_ref = 2000,
            ):
        """Initialization

        :filepath:  (str) path of the PALM profile data file
        :year_ref:  (int) reference year for time

        """
        super(PALMDataProfile, self).__init__(filepath)
        self._year_ref = year_ref
        self.dataset = self._load_dataset()
        self._name = self.dataset.attrs['title']

    def _load_dataset(
            self,
            ):
        """Load data set

        """
        with xr.open_dataset(self._filepath) as fdata:
            # time dimension, use reference time
            time = np.datetime64('{:0d}-01-01T00:00:00'.format(self._year_ref)) + fdata.time
            # depth
            zu = fdata.coords['zpt']
            zw = fdata.coords['zw*pt*']
            z = xr.DataArray(
                zu.data[1:-1],
                dims=('z'),
                coords={'z': zu.data[1:-1]},
                attrs={'long_name': 'z', 'units': zu.units},
                )
            zi = xr.DataArray(
                zw.data[:-1],
                dims=('zi'),
                coords={'zi': zw.data[:-1]},
                attrs={'long_name': 'zi', 'units': zw.units},
                )
            # define output dataset
            out = xr.Dataset()
            for varname in fdata.data_vars:
                var = fdata.data_vars[varname]
                zvar0 = fdata.coords['z'+varname].data[-2]
                if zvar0 == z.data[-1]:
                    # variables at cell centers
                    out[varname] = xr.DataArray(
                        var.data[:,1:-1],
                        dims=('time', 'z'),
                        coords={'time': time, 'z':z},
                        attrs={'long_name': var.long_name, 'units': var.units},
                    )
                elif zvar0 == zi.data[-1]:
                    # variables at cell interfaces
                    out[varname] = xr.DataArray(
                        var.data[:,:-1],
                        dims=('time', 'zi'),
                        coords={'time': time, 'zi':zi},
                        attrs={'long_name': var.long_name, 'units': var.units},
                    )
                else:
                    raise IOError('Invalid z coordinate')
                # attributes
                out.attrs['title'] = fdata.attrs['title']
        return out.transpose()

#--------------------------------
# PALMDataVolume
#--------------------------------

class PALMDataVolume(LESData):

    """A data type for PALM volumn data
       See https://palm.muk.uni-hannover.de/trac/wiki/doc/app/iofiles#DATA_3D_NETCDF

    """

    def __init__(
            self,
            filepath = '',
            year_ref = 2000,
            ):
        """Initialization

        :filepath:  (str) path of the PALM profile data file
        :year_ref:  (int) reference year for time

        """
        super(PALMDataVolume, self).__init__(filepath)
        self._year_ref = year_ref
        self.dataset = self._load_dataset()
        self._name = self.dataset.attrs['title']

    def _load_dataset(
            self,
            ):
        """Load data set

        """
        with xr.open_dataset(self._filepath) as fdata:
            # time dimension, use reference time
            time = np.datetime64('{:0d}-01-01T00:00:00'.format(self._year_ref)) + fdata.time
            # coordinates
            zu = fdata.coords['zu_3d']
            zw = fdata.coords['zw_3d']
            z = xr.DataArray(
                zu.data[1:-1],
                dims=('z'),
                coords={'z': zu.data[1:-1]},
                attrs={'long_name': 'z', 'units': zu.units},
                )
            zi = xr.DataArray(
                zw.data[:-1],
                dims=('zi'),
                coords={'zi': zw.data[:-1]},
                attrs={'long_name': 'zi', 'units': zw.units},
                )
            x  = fdata.coords['x']
            x.attrs['long_name'] = 'x'
            xu = fdata.coords['xu']
            xi = xr.DataArray(
                xu.data,
                dims=('xi'),
                coords={'xi': xu.data},
                attrs={'long_name': 'xi', 'units': xu.units},
                )
            y  = fdata.coords['y']
            y.attrs['long_name'] = 'y'
            yv = fdata.coords['yv']
            yi = xr.DataArray(
                yv.data,
                dims=('yi'),
                coords={'yi': yv.data},
                attrs={'long_name': 'yi', 'units': yv.units},
                )
            # define output dataset
            out = xr.Dataset()
            for varname in fdata.data_vars:
                var = fdata.data_vars[varname]
                # special case
                if 'xu' in var.coords:
                    # variables at the u location
                    out[varname] = xr.DataArray(
                        var.data[:,1:-1,:,:],
                        dims=('time', 'z', 'y', 'xi'),
                        coords={'time': time, 'z':z, 'y':y, 'xi':xi},
                        attrs={'long_name': var.long_name, 'units': var.units},
                    )
                elif 'yv' in var.coords:
                    # variables at the v location
                    out[varname] = xr.DataArray(
                        var.data[:,1:-1,:,:],
                        dims=('time', 'z', 'yi', 'x'),
                        coords={'time': time, 'z':z, 'yi':yi, 'x':x},
                        attrs={'long_name': var.long_name, 'units': var.units},
                    )
                elif 'zu_3d' in var.coords:
                    # variables at cell centers
                    out[varname] = xr.DataArray(
                        var.data[:,1:-1,:,:],
                        dims=('time', 'z', 'y', 'x'),
                        coords={'time': time, 'z':z, 'y':y, 'x':x},
                        attrs={'long_name': var.long_name, 'units': var.units},
                    )
                elif 'zw_3d' in var.coords:
                    # variables at cell interfaces
                    out[varname] = xr.DataArray(
                        var.data[:,:-1,:,:],
                        dims=('time', 'zi', 'y', 'x'),
                        coords={'time': time, 'zi':zi, 'y':y, 'x':x},
                        attrs={'long_name': var.long_name, 'units': var.units},
                    )
                else:
                    raise IOError('Invalid coordinates')
                # attributes
                out.attrs['title'] = fdata.attrs['title']
        return out

#--------------------------------
# NCARLESDataProfile
#--------------------------------

class NCARLESDataProfile(LESData):

    """A data type for NCAR LES profile data

    """

    def __init__(
            self,
            filepath = '',
            year_ref = 2000,
            ):
        """Initialization

        :filepath:  (str) path of the PALM profile data file
        :year_ref:  (int) reference year for time

        """
        super(NCARLESDataProfile, self).__init__(filepath)
        self._year_ref = year_ref
        self.dataset = self._load_dataset()
        self._name = self.dataset.attrs['title']

    def _var_units(
            self,
            varname,
            ):
        """Return the unit of a variable in NCAR LES

        """
        var_units = {
                'stokes':   'm/s',
                'engsbz':   'm$^2$/s$^2$',
                'uxym':     'm/s',
                'vxym':     'm/s',
                'ups':      'm$^2$/s$^2$',
                'vps':      'm$^2$/s$^2$',
                'uvle':     'm$^2$/s$^2$',
                'wcube':    'm$^3$/s$^3$',
                'wfour':    'm$^4$/s$^4$',
                'uwle':     'm$^2$/s$^2$',
                'vwle':     'm$^2$/s$^2$',
                'englez':   'm$^2$/s$^2$',
                'engz':     'm$^2$/s$^2$',
                'uwsb':     'm$^2$/s$^2$',
                'vwsb':     'm$^2$/s$^2$',
                't_rprod':  'm$^2$/s$^3$',
                't_sprod':  'm$^2$/s$^3$',
                't_buoy':   'm$^2$/s$^3$',
                't_wq':     'm$^2$/s$^3$',
                't_wp':     'm$^2$/s$^3$',
                't_tau':    'm$^2$/s$^3$',
                't_tran':   'm$^2$/s$^3$',
                't_diss':   'm$^2$/s$^3$',
                't_dsle':   'm$^2$/s$^3$',
                't_stokes': 'm$^2$/s$^3$',
                'shrz':     'm$^2$/s$^3$',
                'triz':     'm$^2$/s$^3$',
                'dudz':     '1/s',
                'dvdz':     '1/s',
                'wxym':     'm/s',
                'wps':      'm$^2$/s$^2$',
                'tps':      'K$^2$',
                'txym':     'K',
                'tcube':    'K$^3$',
                'utle':     'K m/s',
                'vtle':     'K m/s',
                'wtle':     'K m/s',
                'utsb':     'K m/s',
                'vtsb':     'K m/s',
                'wtsb':     'K m/s',
                }
        return var_units[varname]

    def _load_dataset(
            self,
            ):
        """Load data set

        """
        with xr.open_dataset(self._filepath) as fdata:
            # time dimension, use reference time
            time_ref_str = '{:0d}-01-01T00:00:00'.format(self._year_ref)
            time = pd.to_datetime(fdata.time.data, unit='s', origin=time_ref_str)
            # depth
            zu = fdata.coords['z_u']
            zw = fdata.coords['z_w']
            z = xr.DataArray(
                zu.data,
                dims=('z'),
                coords={'z': zu.data},
                attrs={'long_name': 'z', 'units': 'm'},
                )
            zi = xr.DataArray(
                zw.data,
                dims=('zi'),
                coords={'zi': zw.data},
                attrs={'long_name': 'zi', 'units': 'm'},
                )
            # define output dataset
            out = xr.Dataset()
            for varname in fdata.data_vars:
                var = fdata.data_vars[varname]
                if var.ndim == 2:
                    if 'z_u' in var.coords:
                        # variables at cell centers
                        out[varname] = xr.DataArray(
                            var.data,
                            dims=('time', 'z'),
                            coords={'time': time, 'z':z},
                            attrs={'long_name': var.long_name, 'units': self._var_units(varname)},
                        )
                    elif 'z_w' in var.coords:
                        # variables at cell interfaces
                        out[varname] = xr.DataArray(
                            var.data,
                            dims=('time', 'zi'),
                            coords={'time': time, 'zi':zi},
                            attrs={'long_name': var.long_name, 'units': self._var_units(varname)},
                        )
                    else:
                        raise IOError('Invalid z coordinate')
                elif var.ndim == 3:
                    iscl = 0
                    if 'z_u' in var.coords:
                        # variables at cell centers
                        out[varname] = xr.DataArray(
                            var.data[:,iscl,:],
                            dims=('time', 'z'),
                            coords={'time': time, 'z':z},
                            attrs={'long_name': var.long_name, 'units': self._var_units(varname)},
                        )
                    elif 'z_w' in var.coords:
                        # variables at cell interfaces
                        out[varname] = xr.DataArray(
                            var.data[:,iscl,:],
                            dims=('time', 'zi'),
                            coords={'time': time, 'zi':zi},
                            attrs={'long_name': var.long_name, 'units': self._var_units(varname)},
                        )
                    else:
                        raise IOError('Invalid z coordinate')
                # attributes
                out.attrs['title'] = fdata.attrs['title']
        return out.transpose()

