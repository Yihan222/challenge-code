import numpy as np
import pandas as pd
from rdkit import Chem
import datamol as dm
from CombineMols.CombineMols import CombineMols
from rdkit.Chem import Draw  
import matplotlib.pyplot as plt
import torch

tasks = ['tg']
metrics = ['test_mae','test_lgmae']
param_csv_name = "hyperparameters_summary.csv"

def sep(s):
    point,rang = [],[]
    for val in s:
        vals = val.split("±")
        point.append(float(vals[0]))
        rang.append(float(vals[1]))
    return point,rang

def scatter(path = 'data_pyg/prediction/past/CO2/CO2_raw.csv'):
    points = pd.read_csv(path)
    x,y = points['SMILES'],points['CO2']
    #plt.subplot(2,2,1)
    plt.scatter(x,y,s=5)
    plt.title("CO2 Property Dataset")
    plt.xticks([])
    plt.ylabel("CO2 Permeability")
    plt.savefig('res_img/CO2_data.jpg')

def linechart(open_file = 'result_summary_tg.csv'):
    df = pd.read_csv(open_file)
    group_df = df.groupby('repeat_time')
    num_of_repeat_time = 2
    for t in range(num_of_repeat_time):
        #t = tasks[i]
        save_fig = 'gin_res_{}.jpg'.format(t)
    
        m1 = 'test_mae'
        m2 = 'test_lgmae'
        gp,number_of_layers = group_df.get_group(t)[m1].to_list(), group_df.get_group(t)['number_of_layers'].to_list()
        gp_lg= group_df.get_group(t)[m2].to_list()

        gp_point,gp_range = sep(gp)
        gp_lgmae_point,gp_lgmae_range = sep(gp_lg)
        fig, (ax1, ax2) = plt.subplots(2,figsize=(5,10))
        plt.xlabel("Number of Layers")
        plt.ylabel("Metrics")
        fig.suptitle("Statistic for tg property prediction with {} repeat times".format(t))

        ax1.errorbar(number_of_layers,gp_point,yerr=gp_range,fmt='o-',capsize=1)
        ax1.set_ylabel(m1)
        #ax1.set_title('{}_prop_predic_{}'.format(t,m1))
        ax2.errorbar(number_of_layers,gp_lgmae_point,yerr=gp_lgmae_range,fmt='s-',c='orange',capsize=1)
        ax2.set_ylabel(m2)
        #ax2.set_title('{}_prop_lgmae'.format(t,m2),)
        #plt.legend()
        
        #plt.show()

        
        plt.savefig(save_fig)
        
def strtoimg(smiles):
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    # Use direct edit
    #Draw.MolToFile(mol, newpath+'/d_%s.png'%i)
    Draw.MolToFile(mol, '{}.png'.format(smiles[:6]))   

def getpoints(open_file,train_repeat_time,withsd):
    df = pd.read_csv(open_file)
    df_group = df.groupby('train_repeat_time')
    axs = []
    for t in train_repeat_time:
        df = df_group.get_group(t)['test_r2'].to_list()
        # if on dataset with standard deviation
        if withsd:
            dfpoint,_ = sep(df)
        else:
            dfpoint = df
        dfpoint = [0.5 if x < 0.5 else x for x in dfpoint]
        axs.append(dfpoint)
    return axs

def distribution(singlefile, single_train_repeat_time, mergefile, merge_train_repeat_time,concatname):
    single_axs = getpoints(singlefile,single_train_repeat_time,True) 
    merge_axs = getpoints(mergefile,merge_train_repeat_time,True)
    save_fig = 'r2_dist_imgs/tran/merge_{}.png'.format(concatname)
    # the number of axes need to be specify
    # num_of_axes = 2*(len(single_train_repeat_time)+len(merge_train_repeat_time))
    fig, ax = plt.subplots()
    test_repeat_time = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    

    plt.xlabel("Repeat times of test set")
    plt.ylabel("r2")
    fig.suptitle("r2 distribution comparison between merge {} model with single models".format(concatname))
    for i in range(len(single_train_repeat_time)):
        a = single_axs[i]
        ax.fill_between(test_repeat_time, 0.5, a, alpha=0.2, label='single_{}'.format(single_train_repeat_time[i]))
        ax.grid(True)
    for i in range(len(merge_train_repeat_time)):
        a = merge_axs[i]
        ax.fill_between(test_repeat_time, 0.5, a, alpha=0.1, label='merge_{}'.format(concatname))
        ax.grid(True)
    ax.legend()

    #auxiliary line for GIN best performance around 0.87/0.88
    plt.xticks([0,2,4,6,8,10,12,14,16])
    plt.yticks([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9])
    #plt.axhline(y=0.88, color='gray', linestyle='--',linewidth=0.8)
    #plt.text(-2, 0.88, 'r2=0.88',fontsize=8)
    plt.legend(loc=4)
    plt.savefig(save_fig)


 
if __name__=='__main__':
    singlefile = '../res/tran_res/result.csv'
    mergefile = '../res/tran_res/concat/result.csv'
    #concatname = merge_train_repeat_time[0]
    concatname = '1-8'
    single_train_repeat_time = [1,2,3,4,5,6,7,8]
    merge_train_repeat_time = [18]
    distribution(singlefile, single_train_repeat_time, mergefile, merge_train_repeat_time,concatname)
    
