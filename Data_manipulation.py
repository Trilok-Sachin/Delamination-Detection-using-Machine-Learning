import pandas as pd
import numpy as np

Delam_Data = np.load('Delamination_4_data1.npy')
Uncracked_data = np.load('Uncracked_4layers.npy')

Delam_Data1 = Delam_Data[:,0]-5
print(Delam_Data1)

for i in range(20):

    Delam_Data[:,i] = Uncracked_data[i] - Delam_Data[:, i] 

Delam_Data = np.round(Delam_Data, 6)
print(Delam_Data)

my_df = pd.DataFrame(Delam_Data)
my_df.to_csv('Delam_data_diff_4layers1.csv', index=False)




