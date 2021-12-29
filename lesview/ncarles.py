#--------------------------------
# Data type for NCAR LES
#--------------------------------

from .data import LESData
import numpy as np
import xarray as xr
import pandas as pd

#--------------------------------
# Shared functions
#--------------------------------

def var_units(
        varname,
        ):
    """Return the unit of a variable in NCAR LES

    :varname: (str) variable name
    :return:  (str) unit

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
            'uvle':     'm$^2$/s$^2$',
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
            'ttau11':   'm$^2$/s$^3$',
            'ttau12':   'm$^2$/s$^3$',
            'ttau13':   'm$^2$/s$^3$',
            'ttau22':   'm$^2$/s$^3$',
            'ttau23':   'm$^2$/s$^3$',
            'ttau33':   'm$^2$/s$^3$',
            'dsle11':   'm$^2$/s$^3$',
            'dsle12':   'm$^2$/s$^3$',
            'dsle13':   'm$^2$/s$^3$',
            'dsle22':   'm$^2$/s$^3$',
            'dsle23':   'm$^2$/s$^3$',
            'dsle33':   'm$^2$/s$^3$',
            'udpdx':    'm$^2$/s$^3$',
            'udpdy':    'm$^2$/s$^3$',
            'udpdz':    'm$^2$/s$^3$',
            'vdpdx':    'm$^2$/s$^3$',
            'vdpdy':    'm$^2$/s$^3$',
            'vdpdz':    'm$^2$/s$^3$',
            'wdpdx':    'm$^2$/s$^3$',
            'wdpdy':    'm$^2$/s$^3$',
            'wdpdz':    'm$^2$/s$^3$',
            'uuwle':    'm$^3$/s$^3$',
            'uvwle':    'm$^3$/s$^3$',
            'uwwle':    'm$^3$/s$^3$',
            'vvwle':    'm$^3$/s$^3$',
            'vwwle':    'm$^3$/s$^3$',
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

#--------------------------------
# NCARLESDataProfile
#--------------------------------

class NCARLESDataProfile(LESData):

    """A data type for NCAR LES profile data

    """

    def __init__(
            self,
            filepath = '',
            datetime_origin = '2000-01-01T00:00:00',
            ):
        """Initialization

        :filepath:        (str) path of the PALM profile data file
        :datetime_origin: (scalar) reference date passed to pandas.to_datetime()

        """
        super(NCARLESDataProfile, self).__init__(filepath)
        self._datetime_origin = datetime_origin
        self.dataset = self._load_dataset()
        self._name = self.dataset.attrs['title']

    def _load_dataset(
            self,
            ):
        """Load data set

        """
        with xr.open_dataset(self._filepath) as fdata:
            # time dimension, use reference time
            time = pd.to_datetime(fdata.time.data, unit='s', origin=self._datetime_origin)
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
                            attrs={'long_name': var.long_name, 'units': var_units(varname)},
                        )
                    elif 'z_w' in var.coords:
                        # variables at cell interfaces
                        out[varname] = xr.DataArray(
                            var.data,
                            dims=('time', 'zi'),
                            coords={'time': time, 'zi':zi},
                            attrs={'long_name': var.long_name, 'units': var_units(varname)},
                        )
                    else:
                        raise ValueError('Invalid z coordinate')
                elif var.ndim == 3:
                    iscl = 0
                    if 'z_u' in var.coords:
                        # variables at cell centers
                        out[varname] = xr.DataArray(
                            var.data[:,iscl,:],
                            dims=('time', 'z'),
                            coords={'time': time, 'z':z},
                            attrs={'long_name': var.long_name, 'units': var_units(varname)},
                        )
                    elif 'z_w' in var.coords:
                        # variables at cell interfaces
                        out[varname] = xr.DataArray(
                            var.data[:,iscl,:],
                            dims=('time', 'zi'),
                            coords={'time': time, 'zi':zi},
                            attrs={'long_name': var.long_name, 'units': var_units(varname)},
                        )
                    else:
                        raise ValueError('Invalid z coordinate')
                # attributes
                out.attrs['title'] = fdata.attrs['title']
        return out.transpose()
