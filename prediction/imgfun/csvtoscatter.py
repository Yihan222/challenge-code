import pandas as pd
import matplotlib.pyplot as plt
path = 'data_pyg/prediction/past/CO2/CO2_raw.csv'
points = pd.read_csv(path)
x,y = points['SMILES'],points['CO2']
#plt.subplot(2,2,1)
plt.scatter(x,y,s=5)
plt.title("CO2 Property Dataset")
plt.xticks([])
'''
plt.subplot(2,2,2)
plt.scatter(x,y,s=5)
plt.ylim((0,2500))
plt.title("0-2500")
plt.xticks([])


plt.subplot(2,2,3)
plt.scatter(x,y,s=5)
plt.ylim((0,500))
plt.title("0-500")
plt.xticks([])


plt.subplot(2,2,4)
plt.scatter(x,y,s=5)
plt.ylim((0,100))
plt.title("0-100")
plt.xticks([])
plt.suptitle("N2 Property Dataset")

'''
plt.ylabel("CO2 Permeability")
plt.savefig('res_img/CO2_data.jpg')