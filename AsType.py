import logging
import numpy as np
from neuropype.engine import *
from neuropype.utilities import cache


logger = logging.getLogger(__name__)


class AsType(Node):
    """Change data type."""

    # --- Input/output ports ---
    data = DataPort(Packet, "Data to process.", mutating=False)

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
    def data(self, pkt):
        import lazy_ops

        # try to read from cache
        record = cache.try_lookup(context=self, enabled=self.use_caching,
                                  verbose=True, data=pkt, state=None)

        if record.success():
            self._data = record.data
            return

        out_chunks = {}
        for n, chunk in enumerate_chunks(pkt, nonempty=True):

            out_axes = deepcopy_most(chunk.block.axes)

            dtype = {'float64': np.float64, 'float32': np.float32, 'float16': np.float16,
                     'int64': np.int64, 'int32': np.int32, 'int16': np.int16, 'int8': np.int8,
                     'none': chunk.block._data.dtype}[self.dtype]

            if self.data_class == 'DatasetView' or\
                    (isinstance(chunk.block._data, lazy_ops.DatasetView) and self.data_class == 'none'):
                # Create new DatasetView backed by tempfile
                cache_settings = chunk.block._data._dataset.file.id.get_access_plist().get_cache()
                file_kwargs = {'rdcc_nbytes': cache_settings[2],
                               'rdcc_nslots': cache_settings[1]}
                data = lazy_ops.create_with_tempfile(chunk.block.shape, dtype=dtype,
                                                     chunks=chunk.block._data._dataset.chunks,
                                                     **file_kwargs)
                data[:] = chunk.block._data

            elif self.data_class == 'ndarray' or\
                    (isinstance(chunk.block._data, np.ndarray) and self.data_class == 'none'):
                data = np.array(chunk.block._data).astype(dtype)

            out_chunks[n] = Chunk(block=Block(data=data, axes=out_axes),
                                  props=deepcopy_most(chunk.props))

        pkt = Packet(chunks=out_chunks)
        record.writeback(data=pkt)
        self._data = pkt

    def on_port_assigned(self):
        """Callback to reset internal state when a value was assigned to a
        port (unless the port's setter has been overridden)."""
        self.signal_changed(True)
