#--------------------------------
# Data types
#--------------------------------

import xarray as xr
import numpy as np

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
