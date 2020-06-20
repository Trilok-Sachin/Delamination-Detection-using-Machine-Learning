import pyansys
import numpy as np

ansys = pyansys.Mapdl(loglevel="INFO")

ITER_SIZE = 20

BLOCK_X = np.linspace(0.75,2,ITER_SIZE)
BLOCK_Y = np.linspace(0.75,2,ITER_SIZE)
BLOCK_THICK = np.linspace(0.01,0.04,ITER_SIZE) 
FREQs = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20"]
DATA_ARRAY = np.zeros((20))

ansys.run("/PREP7")

ansys.et(1, "SOLID185")

ansys.run("MPTEMP,,,,,,,")
ansys.mptemp(1, 0)
ansys.mpdata("DENS", 1, "", 1600)
ansys.mpdata("EX", 1, "", 2e9)
ansys.mpdata("PRXY", 1, "", 0.3)

ansys.block(0, 1, 0, 1, 0, 0.01 )
ansys.block(0, 1, 0, 1, 0.01, 0.02)
ansys.block(0, 1, 0, 1, 0.02, 0.03)
ansys.block(0, 1, 0, 1, 0.03, 0.04)


ansys.run("/VIEW,1,1,2,3")

#ansys.run("/ANG,1")
#ansys.run("/REP,FAST")

ansys.mshape(0, "3D")
ansys.mshkey(1)

ansys.flst(5, 2, 6, "ORDE", 2)
ansys.fitem(5, 1)
ansys.fitem(5, -2)

ansys.cm("_Y", "VOLU")
ansys.vsel("", "", "", "P51X")
ansys.cm("_Y1", "VOLU")
ansys.chkmsh("'VOLU'")
ansys.cmsel("S", "_Y")
ansys.vmesh("_Y1")

ansys.cmdele("_Y")
ansys.cmdele("_Y1")
ansys.cmdele("_Y2")

ansys.run("NSEL,ALL")
ansys.nsel("A", "LOC", "Z", 0.01)
ansys.nsel("A", "LOC", "Z", 0.02)
ansys.nsel("A", "LOC", "Z", 0.03)
ansys.run("/REPLOT")
ansys.nplot()

ansys.nummrg("NODE", "", "", "", "LOW")
ansys.eplot()

ansys.finish()

ansys.run("/SOL")

ansys.antype(2)
ansys.modopt("LANB", 20)
ansys.eqslv("SPAR")
ansys.mxpand(0, "", "", 0)
ansys.lumpm(0)
ansys.pstres(0)

ansys.flst(2, 8, 5, "ORDE", 4)
ansys.fitem(2, 3)
ansys.fitem(2, -6)
ansys.fitem(2, 9)
ansys.fitem(2, -12)

ansys.run("/GO")
ansys.da("P51X", "ALL", 0)

ansys.run("/STATUS,SOLU")
ansys.allsel("ALL")
ansys.solve()
ansys.finish()

for j in range(len(FREQs)):
	ansys.get(FREQs[j],"MODE",j+1,"FREQ")

ansys.load_parameters()
x = ansys.parameters

for j in range(len(FREQs)):
	DATA_ARRAY[j] = x[FREQs[j]]

ansys.cmdele("_Y1")

ansys.cmdele("_Y2")

# *

#ansys.run("/POST1")

#ansys.set("LIST")

ansys.exit()

np.save('Uncracked_4layers.npy',DATA_ARRAY)
print(x)

