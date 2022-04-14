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

