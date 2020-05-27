#--------------------------------
# Data class
#--------------------------------

import xarray as xr
import numpy as np

class LESData:

    """The common data type

    """

    def __init__(
            self,
            filepath = '',
            ):
        """Initialization

        :filepath:   (str) path of the LES data file

        """
        self._filepath = filepath

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+':']
        summary.append('{:>12s}: {:s}'.format('data path', self._filepath))
        return '\n'.join(summary)

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
                zvar0 = fdata.coords['z'+varname].data[-2]
                if zvar0 == z.data[-1]:
                    # variables at cell centers
                    var = fdata.data_vars[varname]
                    out[varname] = xr.DataArray(
                        var.data[:,1:-1],
                        dims=('time', 'z'),
                        coords={'time': time, 'z':z},
                        attrs={'long_name': var.long_name, 'units': var.units},
                    )
                elif zvar0 == zi.data[-1]:
                    # variables at cell interfaces
                    var = fdata.data_vars[varname]
                    out[varname] = xr.DataArray(
                        var.data[:,:-1],
                        dims=('time', 'zi'),
                        coords={'time': time, 'zi':zi},
                        attrs={'long_name': var.long_name, 'units': var.units},
                    )
                else:
                    raise IOError('Invalid z coordinate')
        return out.transpose()
