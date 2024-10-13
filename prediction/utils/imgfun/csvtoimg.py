from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

open_file = 'result_summary_tg.csv'
#tasks = ['O2','N2','CO2','H2','He']
tasks = ['tg']
metrics = ['test_mae','test_lgmae']
def sep(s):
    point,range = [],[]
    for val in s:
        vals = val.split("±")
        point.append(float(vals[0]))
        range.append(float(vals[1]))
    return point,range


if __name__=='__main__':
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

    