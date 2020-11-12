#--------------------------------
# Data type for PALM
#--------------------------------

from .data import LESData
import numpy as np
import xarray as xr
import pandas as pd

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
            datetime_origin = '2000-01-01T00:00:00',
            ):
        """Initialization

        :filepath:        (str) path of the PALM profile data file
        :datetime_origin: (scalar) reference date passed to pandas.to_datetime()

        """
        super(PALMDataProfile, self).__init__(filepath)
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
            time = pd.to_datetime(fdata.time.data/np.timedelta64(1, 's'), unit='s', origin=self._datetime_origin)
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
                if varname.startswith('NORM'):
                    break
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
            datetime_origin = '2000-01-01T00:00:00',
            ):
        """Initialization

        :filepath:        (str) path of the PALM profile data file
        :datetime_origin: (scalar) reference date passed to pandas.to_datetime()

        """
        super(PALMDataVolume, self).__init__(filepath)
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
            time = pd.to_datetime(fdata.time.data/np.timedelta64(1, 's'), unit='s', origin=self._datetime_origin)
            # coordinates
            if 'zu_3d' in fdata.coords:
                zu = fdata.coords['zu_3d']
                z = xr.DataArray(
                    zu.data[1:-1],
                    dims=('z'),
                    coords={'z': zu.data[1:-1]},
                    attrs={'long_name': 'z', 'units': zu.units},
                    )
            if 'zw_3d' in fdata.coords:
                zw = fdata.coords['zw_3d']
                zi = xr.DataArray(
                    zw.data[:-1],
                    dims=('zi'),
                    coords={'zi': zw.data[:-1]},
                    attrs={'long_name': 'zi', 'units': zw.units},
                    )
            if 'x' in fdata.coords:
                x  = fdata.coords['x']
                x.attrs['long_name'] = 'x'
            if 'xu' in fdata.coords:
                xu = fdata.coords['xu']
                xi = xr.DataArray(
                    xu.data,
                    dims=('xi'),
                    coords={'xi': xu.data},
                    attrs={'long_name': 'xi', 'units': xu.units},
                    )
            if 'y' in fdata.coords:
                y  = fdata.coords['y']
                y.attrs['long_name'] = 'y'
            if 'yv' in fdata.coords:
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

