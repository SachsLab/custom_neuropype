import logging
import numpy as np
from neuropype.engine import *
from neuropype.utilities import cache
import tempfile
import h5pickle as h5py
import weakref
import lazy_ops


logger = logging.getLogger(__name__)


class AsType(Node):
    """Change data type."""

    # --- Input/output ports ---
    data = DataPort(Packet, "Data to process.")

    # --- Properties ---
    dtype = EnumPort('none', domain=['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'none'],
                     help="""The new dtype. Use 'none' for no change.""")
    data_class = EnumPort('none', domain=['DatasetView', 'ndarray', 'none'])
    use_caching = BoolPort(False, """Enable caching.""", expert=True)

    def __init__(self,
                 dtype: Union[str, None, Type[Keep]] = Keep,
                 data_class: Union[str, None, Type[Keep]] = Keep,
                 use_caching: Union[bool, None, Type[Keep]] = Keep,
                 **kwargs):
        super().__init__(dtype=dtype, data_class=data_class, use_caching=use_caching, **kwargs)

    @classmethod
    def description(cls):
        """Declare descriptive information about the node."""
        return Description(name='As Type',
                           description="""\
                           Change the dtype of the chunk data,
                           and/or change the block._data class.
                           """,
                           version='0.1.0', status=DevStatus.alpha)

    @data.setter
    def data(self, v):
        # try to read from cache
        record = cache.try_lookup(context=self, enabled=self.use_caching,
                                  verbose=True, data=v, state=None)

        if record.success():
            self._data = record.data
            return

        for n, chunk in enumerate_chunks(v, nonempty=True):

            dtype = {'float64': np.float64, 'float32': np.float32, 'float16': np.float16,
                     'int64': np.int64, 'int32': np.int32, 'int16': np.int16, 'int8': np.int8,
                     'none': chunk.block._data.dtype}[self.dtype]

            if self.data_class == 'DatasetView' or\
                    (isinstance(chunk.block._data, lazy_ops.DatasetView) and self.data_class == 'none'):
                # Create a tempfile and close it.
                tf = tempfile.NamedTemporaryFile(delete=False)
                tf.close()
                # Open again with h5py/h5pickle
                f = h5py.File(tf.name, mode='w', libver='latest')
                dset_name_in_parent = chunk.block._data._dataset.name.split('/')[-1]
                chunk.block._data._dataset = f.create_dataset(dset_name_in_parent,
                                                              data=chunk.block._data, dtype=dtype,
                                                              chunks=True, compression="gzip")
                f.swmr_mode = True
                # Setup an automatic deleter for the new tempfile
                chunk.block._data._finalizer = weakref.finalize(
                    chunk.block._data, chunk.block._data.on_finalize, f, f.filename)

            elif self.data_class == 'ndarray' or\
                    (isinstance(chunk.block._data, np.ndarray) and self.data_class == 'none'):
                chunk.block._data = np.array(chunk.block._data).astype(dtype)

        record.writeback(data=v)
        self._data = v

    def on_port_assigned(self):
        """Callback to reset internal state when a value was assigned to a
        port (unless the port's setter has been overridden)."""
        self.signal_changed(True)
