""" Script generated by pyansys version 0.39.17 """

import pyansys
import numpy as np
import pandas as pd
import random


ansys = pyansys.Mapdl(loglevel="INFO")

ITER_SIZE = 2000
FEATURES = 25

BLOCK_X = np.linspace(0.75,2,20)
BLOCK_Y = np.linspace(0.75,2,20)
BLOCK_THICK = np.linspace(0.01,0.04,20) 

CRACK_X1 = np.linspace(0, 0.9, 20)
CRACK_Y1 = np.linspace(0, 0.9, 20)
LAYERS = [0.01, 0.02]
#CRACK_X2 = np.linspace(0, 0.9, 19)
#CRACK_Y2 = np.linspace(0, 0.9, 19)

DATA_ARRAY = np.zeros((ITER_SIZE,FEATURES))

FREQs = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20"]

for i in range(ITER_SIZE):

	try:
	
		rand_index_x1 = np.random.randint(20, size=1)
		rand_index_x2 = np.random.randint(20, size=1)
		X2 = np.linspace(0.05, 1 - CRACK_X1[rand_index_x1], 20)
		CRACK_X2_ = CRACK_X1[rand_index_x1] + X2[rand_index_x2]

		rand_index_y1 = np.random.randint(20, size=1)
		rand_index_y2 = np.random.randint(20, size=1)
		Y2 = np.linspace(0.05, 1 - CRACK_Y1[rand_index_y1], 20)
		CRACK_Y2_ = CRACK_Y1[rand_index_y1] + Y2[rand_index_y2]
			
		rand_layer = random.sample(range(0, 2), 2)
		#ansys.run('/CLEAR')
		ansys.run("/CLE    ,NOSTART")	
		ansys.run("/PREP7")

		ansys.et(1, "SOLID185")

		ansys.run("MPTEMP,,,,,,,")
		ansys.mptemp(1, 0)
		ansys.mpdata("DENS", 1, "", 1600)

		ansys.mpdata("EX", 1, "", 2e9)
		ansys.mpdata("PRXY", 1, "", 0.3)

		ansys.block(0, 1, 0, 1, 0, 0.01)
		ansys.block(0, 1, 0, 1, 0.01, 0.02)
		ansys.block(0, 1, 0, 1, 0.02, 0.03)
		ansys.block(0, 1, 0, 1, 0.03, 0.04)

		#ansys.run("/VIEW,1,1,2,3")
		#ansys.run("/ANG,1")
		#ansys.run("/REP,FAST")

		ansys.mshape(0, "3D")
		ansys.mshkey(1)

		# *

		ansys.flst(5, 2, 6, "ORDE", 2)
		ansys.fitem(5, 1)
		ansys.fitem(5, -2)

		ansys.cm("_Y", "VOLU")
		ansys.vsel("", "", "", "P51X")
		ansys.cm("_Y1", "VOLU")
		#ansys.chkmsh("'VOLU'")
		#ansys.run("CHKMESH,'VOLU'")
		ansys.cmsel("S", "_Y")

		# *

		ansys.vmesh("_Y1")

		# *

		ansys.cmdele("_Y")
		ansys.cmdele("_Y1")
		ansys.cmdele("_Y2")

		# *
		

		ansys.run("NSEL,ALL")
		ansys.nplot()
		ansys.nsel("R", "LOC", "X", 0, CRACK_X1[rand_index_x1][0])
		#ansys.run("/REPLOT")
		ansys.nsel("A", "LOC", "X", CRACK_X2_[0][0], 1)
		#ansys.run("/REPLOT")
		ansys.nsel("A", "LOC", "Y", 0, CRACK_Y1[rand_index_y1][0])
		#ansys.run("/REPLOT")
		ansys.nsel("A", "LOC", "Y", CRACK_Y2_[0][0], 1)
		#ansys.run("/REPLOT")
		ansys.nsel("R", "LOC", "Z", LAYERS[rand_layer[0]])
		ansys.nsel("A", "LOC", "Z", LAYERS[rand_layer[1]])
		ansys.nsel("A", "LOC", "Z", 0.03)
		#ansys.run("/REPLOT")
		#ansys.run("/REPLOT")

		ansys.nummrg("NODE", "", "", "", "LOW")

		#ansys.eplot()

		ansys.finish()

		ansys.run("/SOL")

		# *

		ansys.antype(2)

		# *

		ansys.modopt("LANB", 20)
		ansys.eqslv("SPAR")
		ansys.mxpand(0, "", "", 0)
		ansys.lumpm(0)
		ansys.pstres(0)

		# *

		ansys.modopt("LANB", 20, 0, 0, "", "OFF")
		ansys.flst(2, 8, 5, "ORDE", 4)
		ansys.fitem(2, 3)
		ansys.fitem(2, -6)
		ansys.fitem(2, 9)
		ansys.fitem(2, -12)

		# *

		ansys.run("/GO")
		ansys.da("P51X", "ALL", 0)
		ansys.run("/STATUS,SOLU")
		ansys.allsel("ALL")
		ansys.solve()

		for j in range(len(FREQs)):
			ansys.get(FREQs[j],"MODE",j+1,"FREQ")

		ansys.load_parameters()
		x = ansys.parameters

		for j in range(len(FREQs)):
			DATA_ARRAY[i][j] = x[FREQs[j]]
		DATA_ARRAY[i][20] = CRACK_X1[rand_index_x1]
		DATA_ARRAY[i][21] = CRACK_X2_
		DATA_ARRAY[i][22] = CRACK_Y1[rand_index_y1]
		DATA_ARRAY[i][23] = CRACK_Y2_
		DATA_ARRAY[i][24] = LAYERS[rand_layer[0]]

		ansys.finish()
	
	except Exception as e:
		pass
		print('Exception', e)

#ansys.run("/POST1")

#ansys.set("LIST")

ansys.exit()

print(DATA_ARRAY)

my_df = pd.DataFrame(DATA_ARRAY)
np.save('Delamination_4_data1.npy', DATA_ARRAY)

my_df.to_csv('Delamination_4_layers1.csv', index=False)


