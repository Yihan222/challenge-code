from matplotlib import pyplot as plt
import numpy as np
def strtolist(s: str):
    ss = s.split('\t')
    ab =[]
    abb = []
    for i in ss:
        ii = i.split("±")
        ab.append(float(ii[0]))
        abb.append(float(ii[1]))
    return ab,abb

if __name__=='__main__':
    N2_prop_GIN,yegin = strtolist("424.8062±40.7246	395.2071±5.0481	385.1268±17.4162	365.1596±34.1004	364.8618±35.3398")
    N2_prop_GCN_Virtual,yegcnv = strtolist("394.5562±139.4885	393.9599±108.0602	388.6633±108.0091	332.3778±71.9094	387.4223±110.4740")
    
    Repeat_times = [1,2,3,5,10]
    fig,ax=plt.subplots()
    ax.errorbar(Repeat_times,N2_prop_GCN_Virtual,yerr=yegcnv,fmt='o-',label='CO2_prop_GCN_Virtual',capsize=1)
    ax.errorbar(Repeat_times,N2_prop_GIN,yerr=yegin,fmt='s-',label='CO2_prop_GIN',capsize=1)

    plt.legend()
    plt.title("Statistic for CO2 property prediction")
    plt.xlabel("Repeat Times")
    plt.ylabel("MAE")
    #plt.show()
    plt.savefig('CO2prop_mae.jpg')