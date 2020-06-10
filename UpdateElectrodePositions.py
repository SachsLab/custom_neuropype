import logging
import scipy.io
import numpy as np
from neuropype.engine import *
from neuropype.utilities.cloud import storage


logger = logging.getLogger(__name__)


class UpdateElectrodePositions(Node):
    # --- Input/output ports ---
    data = Port(None, Packet, "Packet with Blackrock packet output from ImportNSX", required=True,
                editable=False, mutating=True)
    filename = StringPort("", """Path to the map file.
                    """, is_filename=True)

    banks = ListPort(['A', 'B', 'C', 'D'], domain=str, help="""
        List of single-character-strings ['A', 'B', 'C', 'D'] to indicate which banks were
        recorded in the input data packet.
    """)

    # options for cloud-hosted files
    cloud_host = EnumPort("Default", ["Default", "Azure", "S3", "Google",
                                      "Local", "None"], """Cloud storage host to
            use (if any). You can override this option to select from what kind of
            cloud storage service data should be downloaded. On some environments
            (e.g., on NeuroScale), the value Default will be map to the default
            storage provider on that environment.""")
    cloud_account = StringPort("", """Cloud account name on storage provider
            (use default if omitted). You can override this to choose a non-default
            account name for some storage provider (e.g., Azure or S3.). On some
            environments (e.g., on NeuroScale), this value will be
            default-initialized to your account.""")
    cloud_bucket = StringPort("", """Cloud bucket to read from (use default if
            omitted). This is the bucket or container on the cloud storage provider
            that the file would be read from. On some environments (e.g., on
            NeuroScale), this value will be default-initialized to a bucket
            that has been created for you.""")
    cloud_credentials = StringPort("", """Secure credential to access cloud data
            (use default if omitted). These are the security credentials (e.g.,
            password or access token) for the the cloud storage provider. On some
            environments (e.g., on NeuroScale), this value will be
            default-initialized to the right credentials for you.""")

    @classmethod
    def description(cls):
        return Description(name='Update Electrode Positions for Utah Array',
                           description="""
                           The Martinez-Trujillo lab constructs map files (.cmp) to describe the
                           Utah electrode array channel mapping. Here we process the map file
                           and update the channel positions in the data packet.
                           """,
                           version='0.1',
                           license=Licenses.MIT)

    @data.setter
    def data(self, pkt):
        if pkt is not None:
            import pandas as pd
            filename = storage.cloud_get(self.filename, host=self.cloud_host,
                                         account=self.cloud_account,
                                         bucket=self.cloud_bucket,
                                         credentials=self.cloud_credentials)

            logger.info("Replacing electrode positions with positions loaded from %s..." % filename)
            map_info = {}
            with open(filename, 'r') as f:
                _ = f.readline()
                line_ix = 1
                map_start = None
                while True:
                    line = f.readline()
                    if not line:
                        # nothing returned
                        break
                    words = line.strip().split()
                    if not len(words):
                        # empty line after stripping
                        continue
                    if words[0].lower() in ['subject', 'hemisphere']:
                        map_info[words[0].lower()] = words[1]
                    elif words[0].lower() in ['wireorientation', 'implantorientation', 'electrodespacing']:
                        # according to notes:
                        # wire pointing right and array on left hemi
                        map_info[words[0].lower()] = int(words[1])
                    elif words[0] == 'Cerebus':
                        # Reached map
                        map_start = line_ix + 1
                        break
                    line_ix += 1
            df = None
            if map_start is not None:
                df = pd.read_csv(filename, sep='\t', header=None, names=['X', 'Y', 'Bank', 'ChInBank'],
                                 skiprows=map_start + 1)
                df = df.infer_objects()
                # The Matlab code says to flip ud, but not flipping is the better way to align with the diagrams I
                # received from members of Julio's lab.
                # df['Y'] = max(df['Y']) - df['Y']
                # Convert X and Y into um
                spacing = map_info['electrodespacing'] if 'electrodespacing' in map_info else 400
                df['X'] *= spacing
                df['Y'] *= spacing
                # Add a column of channel indices
                ch_offset = np.array([32 * (ord(_) - 65) for _ in df['Bank']])
                df['ChIdx'] = df['ChInBank'] + ch_offset
                df = df.sort_values('ChIdx')
                # Trim out the rows from banks not in self.banks
                df = df[df['Bank'].isin(self.banks)]

            # For each chunk, replace the electrode positions with positions from the map file stored in the df.
            # - The chunk>Block>SpaceAxis names are created by python-neo and do not correspond to
            #   any channel names we have in our df. We create channel names for our df from ch0 to chN
            # - When the data have fewer channels than exist in the df, we exhaust the banks in order.
            #   (Another approach not used is to get equal numbers of channels from each bank).
            if df is not None:
                positions = df[['X', 'Y']].to_numpy().astype(float) / 10e6  # um to m
                positions = np.hstack((positions, np.zeros_like(positions[:, 0][:, None])))  # Add on z dimension
                ch_names = [f'ch{_:d}' for _ in range(1, 1+len(positions))]
                for n, chnk in enumerate_chunks(pkt, with_axes=(space,)):
                    sp_idx_in_df = [ch_names.index(_) for _ in chnk.block.axes[space].names]
                    chnk.block.axes[space].positions[:] = positions[sp_idx_in_df]

                    if False:
                        new_space_ax = chnk.block.axes['space']
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(1, 1)
                        ax.set_xlim([-0.2, 4.0])
                        ax.set_ylim([-0.2, 4.0])
                        for name, xy in zip(new_space_ax.names, new_space_ax.positions[:, :2]):
                            ax.text(xy[0], xy[1], name, ha="center", va="center")
                        plt.show()

        self._data = pkt
