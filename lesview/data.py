#--------------------------------
# Data types
#--------------------------------

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

