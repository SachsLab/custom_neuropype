import logging
import numpy as np
from neuropype.engine import *


logger = logging.getLogger(__name__)


class ReconContFromEvents(Node):
    """Reconstitute Continuous Signal From Sparse Events"""

    # --- Input/output ports ---
    data = DataPort(Packet, "Data to process.", mutating=False)

    # --- Properties ---
    add_noise = BoolPort(False, """Add white noise to reconstituted data.""")
    add_lfps = BoolPort(False, """If present, upsample analog signals and add to reconstituted data.""")

    def __init__(self,
                 add_noise: Union[bool, None, Type[Keep]] = Keep,
                 add_lfps: Union[bool, None, Type[Keep]] = Keep,
                 **kwargs):
        super().__init__(add_noise=add_noise, add_lfps=add_lfps, **kwargs)

    @classmethod
    def description(cls):
        """Declare descriptive information about the node."""
        return Description(name='Reconstitute Continuous Signal From (Sparse) Events',
                           description="""\
                           Use sparse event waveforms (and optionally sparse event train) to reconstitute
                           a continuous signal on a zeros background. Optionally add noise and lfps (if provided)
                           to the output.
                           Note that sorting is lost.
                           """,
                           version='0.1.0', status=DevStatus.alpha)

    @data.setter
    def data(self, pkt):

        # Get the waveforms chunk, if present.
        wf_name, wf_chunk = find_first_chunk(pkt, with_axes=(instance, time),
                                             without_flags=(Flags.is_signal, Flags.is_sparse),
                                             allow_markers=False)

        # Get the event train. Only useful for
        evt_name, evt_chunk = find_first_chunk(pkt, with_axes=(space, time),
                                               with_flags=Flags.is_sparse,
                                               allow_markers=False)

        # Get the signals chunk, if present.
        sig_name, sig_chunk = find_first_chunk(pkt, with_axes=(space, time),
                                               without_flags=Flags.is_sparse)

        # Make TimeAxis new continuous data
        wf_blk = wf_chunk.block
        t0 = 0.0  # np.min(wf_blk.axes['ne.instance'].times) + wf_blk.axes['time'].times[0]
        t_end = np.max(wf_blk.axes[instance].times) + wf_blk.axes[time].times[-1]
        step_size = 1 / wf_blk.axes[time].nominal_rate
        t_vec = np.arange(t0, t_end + step_size, step_size)
        new_time_ax = TimeAxis(times=t_vec, nominal_rate=wf_blk.axes[time].nominal_rate)

        # Make SpaceAxis for new continuous data
        # Get channel labels, sort so Ch10 is after Ch9, etc.
        chan_labels = np.unique(wf_blk.axes[instance].data['chan_label'])
        sort_ix = np.argsort([int(_[2:]) for _ in chan_labels])
        chan_labels = chan_labels[sort_ix]
        # Get channel positions if available
        if evt_name:
            sp_ax = evt_chunk.block.axes[space]
            new_pos = np.array([sp_ax.positions[sp_ax.names == _][0] for _ in chan_labels])
        else:
            new_pos = None
        new_space_ax = SpaceAxis(names=chan_labels, positions=new_pos)

        # Calculate index offsets for each waveform
        wf_idx_off = (wf_blk.axes[time].times * wf_blk.axes[time].nominal_rate).astype(int)

        # Initialize null continuous data
        dat = np.zeros((len(chan_labels), len(t_vec)), dtype=wf_blk.data.dtype)

        # Superimpose spikes on zeros, one channel at a time
        for ch_ix, ch_label in enumerate(chan_labels):
            ch_blk = wf_blk[..., instance[wf_blk.axes[instance].data['chan_label'] == ch_label], ...]
            if ch_blk.size:
                ch_wf_dat = np.copy(ch_blk[instance, time].data).flatten()
                ch_idx = np.searchsorted(t_vec, ch_blk.axes[instance].times)[:, None] + wf_idx_off[None, :]
                ch_idx = ch_idx.flatten()
                # Only take the first occurrence of any particular sample to avoid
                #  samples that may be over-represented if there are overlapping waveforms.
                uq_ch_idx, uq_wf_idx = np.unique(ch_idx, return_index=True)
                dat[ch_ix, uq_ch_idx] = ch_wf_dat[uq_wf_idx]

        raw_blk = Block(data=dat, axes=(new_space_ax, new_time_ax))

        # Add white noise to samples that weren't written with waveforms.
        if self.add_noise:
            # Noise should very rarely cross threshold.
            # Set 4 STDs (=99.96% of samples) to be less than threshold of -54 uV:
            noise_std = 54/4
            for ch_dat in raw_blk.data:
                b_zero = ch_dat == 0.
                ch_dat[b_zero] = (noise_std * np.random.randn(np.sum(b_zero))).astype(np.int16)

        # Superimpose interpolated LFPs if available
        if self.add_lfps and sig_name is not None:
            lfp_blk = sig_chunk.block[space[raw_blk.axes[space].names.tolist()], ..., time]
            # Manual interpolation, one channel at a time, to save memory
            from scipy.interpolate import interp1d
            new_times = raw_blk.axes[time].times
            old_times = lfp_blk.axes[time].times
            for chan_ix, chan_label in enumerate(lfp_blk.axes[space].names):
                if chan_label in chan_labels:
                    f = interp1d(old_times, lfp_blk.data[chan_ix], kind='cubic', axis=-1,
                                 assume_sorted=True, fill_value='extrapolate')
                    lfp_upsamp = f(new_times)
                    full_ch_ix = np.where(chan_labels == chan_label)[0][0]
                    raw_blk.data[full_ch_ix] = raw_blk.data[full_ch_ix] + lfp_upsamp.astype(raw_blk.data.dtype)

        self._data = Packet(chunks={'recon_raw': raw_blk})

    def on_port_assigned(self):
        """Callback to reset internal state when a value was assigned to a
        port (unless the port's setter has been overridden)."""
        self.signal_changed(True)
