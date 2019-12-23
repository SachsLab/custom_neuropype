# First download and run AudioCapture application from here:
# https://github.com/labstreaminglayer/App-AudioCapture/releases
# Allow it on your network if required.

# Your PyCharm environment should have Neuropype cpe attached as a dependency.

# Then run this script to visualize a real-time plot of the computer's microphone input.

# import cProfile
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import time as time_
from neuropype.engine import Graph, Scheduler
import neuropype.nodes as nn

inlet = nn.LSLInput(query="type='Audio'")
timeseriesplot = nn.TimeSeriesPlot(scale=0.25, time_range=3.0, initial_dims=[0, 100, 800, 600])

patch = Graph()
patch.chain(inlet, timeseriesplot)
scheduler = Scheduler(patch)
tlast = time_.time()


def update():
    global tlast, scheduler
    tstart = time_.time()
    scheduler.advance()
    tnow = time_.time()
    # print("{0:8.3f}, {1:8.3f}, {2:8.3f}".format(1000 * (tnow - tstart), 1000 * (tstart - tlast), 1000 * (tnow - tlast)))
    tlast = tnow


def main():
    qapp = QtGui.QApplication(sys.argv)
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    # Delay not needed because scheduler.advance will block while waiting for data as LSLInput has block_wait=True
    timer.start(1)
    # print("scheduler.advance(), outside, total")
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(QtGui.QApplication.instance().exec_())

# cProfile.run("main()", filename="vistest.profile")
main()
