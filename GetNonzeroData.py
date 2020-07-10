import logging
import numpy as np

from neuropype.engine import *
from neuropype.utilities import cache


logger = logging.getLogger(__name__)


class GetNonzeroData(Node):
    """Get a copy of continuous timeseries including only samples where all channels were zero."""

    # --- Input/output ports ---
    data = DataPort(Packet, "Data to process.", mutating=False)

    waveform_window = ListPort([0, 0], float, """Time window (seconds) around
        each event that is assumed to have non-zero waveform data.
            """, verbose_name='waveform window')
    use_caching = BoolPort(False, """Enable caching.""", expert=True)

    def __init__(self,
                 waveform_window: Union[List[float], None, Type[Keep]] = Keep,
                 use_caching: Union[bool, None, Type[Keep]] = Keep,
                 **kwargs):
        super().__init__(waveform_window=waveform_window, use_caching=use_caching, **kwargs)

    @classmethod
    def description(cls):
        """Declare descriptive information about the node."""
        return Description(name='Drop Blank Times',
                           description="""\
                           Drop samples (time axis) where all channels were value == 0
                           """,
                           version='0.1.0', status=DevStatus.alpha)

    @data.setter
    def data(self, pkt):

        record = cache.try_lookup(context=self, enabled=self.use_caching,
                                  verbose=True, data=pkt, state=None)
        if record.success():
            self._data = record.data
            return

        # Get the event train, if present.
        evt_name, evt_chunk = find_first_chunk(pkt, with_axes=(space, time),
                                               with_flags=Flags.is_sparse,
                                               allow_markers=False)

        # Get the signals chunk, if present.
        sig_name, sig_chunk = find_first_chunk(pkt, with_axes=(space, time),
                                               without_flags=Flags.is_sparse)

        if sig_name is None:
            return

        b_keep = np.zeros((len(sig_chunk.block.axes['time']),), dtype=bool)
        if evt_name is not None:
            # if evt_chunk is present then we can use that to identify which samples were covered by a waveform
            spk_blk = evt_chunk.block
            spike_inds = np.sort(np.unique(spk_blk._data.indices))
            wf_samps = [int(_ * sig_chunk.block.axes[time].nominal_rate) for _ in self.waveform_window]
            spike_inds = spike_inds[np.logical_and(spike_inds > wf_samps[0], spike_inds < (len(b_keep) - wf_samps[1]))]
            dat_inds = np.unique(spike_inds[:, None] + np.arange(wf_samps[0], wf_samps[1], dtype=int)[None, :])
            b_keep[dat_inds] = True
        else:
            # else, scan the data. This is probably slower than above.
            for ch_ix in range(len(sig_chunk.block.axes[space])):
                b_keep = np.logical_or(b_keep,
                                       sig_chunk.block[space[ch_ix], ...].data[0] != 0)

        logger.info(f"Copying {np.sum(b_keep)} / {len(b_keep)} samples ({100.*np.sum(b_keep)/len(b_keep):.2f} %)...")

        # Create output block that is copy of input
        out_axes = list(sig_chunk.block[space, time].axes)
        out_axes[-1] = TimeAxis(times=out_axes[-1].times[b_keep], nominal_rate=out_axes[-1].nominal_rate)
        out_blk = Block(data=sig_chunk.block._data, axes=out_axes, data_only_for_type=True)
        for ch_ix in range(len(sig_chunk.block.axes[space])):
            # Non-slice and non-scalar indexing of long block axes is quite slow, so get full time then slice that.
            out_blk[ch_ix:ch_ix+1, :].data = sig_chunk.block[ch_ix:ch_ix+1, :].data[:, b_keep]

        # Create a new packet using only nonzero samples. Note this uses a ndarray, not DatasetView
        self._data = Packet(chunks={sig_name: Chunk(
            block=out_blk,
            props=deepcopy_most(sig_chunk.props)
        )})

        record.writeback(data=self._data)

    def on_port_assigned(self):
        """Callback to reset internal state when a value was assigned to a
        port (unless the port's setter has been overridden)."""
        self.signal_changed(True)
