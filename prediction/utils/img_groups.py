from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
import matplotlib.mlab as mlab
import seaborn as sns 
import scipy.stats as stats


task = 'tg'
open_file = './rnn/res.csv'
def check_dis(row):
    gt = float(row[1])
    min_dis = float('inf')
    pos = [1]
    res = float(row[2])
    for i in range(2,12):
        cur = float(row[i])
        dis = abs(gt-cur)
        if dis < min_dis:
            min_dis = dis
            pos = [i-1]
            res = cur
        elif dis == min_dis:
            pos.append(i-1)

    return pos,res
def create_multi_bars(labels, datas, tick_step, group_gap, bar_gap, save_fig):
    '''
    labels : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    fig = plt.figure(figsize=(12,4)) 
    # ticks为x轴刻度
    ticks = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可
    baseline_x = ticks - (group_width - bar_span) / 2
    print(baseline_x)
    for index, y in enumerate(datas):
        
        if index == 0:
            plt.bar(baseline_x + index*bar_span, y,bar_width,label = '1 RU')
        if index == 1:
            plt.bar(baseline_x + index*bar_span, y,bar_width,label = '2-4 RUs')
        if index == 2:
            plt.bar(baseline_x + index*bar_span, y,bar_width,label = '5-7 RUs')
        if index == 3:
            plt.bar(baseline_x + index*bar_span, y,bar_width,label = '8-10 RUs')
        
        #plt.bar(baseline_x + index*bar_span, y,bar_width,label = '{} RU'.format(index+1))
    # x轴刻度标签位置与x轴刻度一致
    plt.xticks(ticks, labels,rotation=-15)
    plt.tick_params(axis='x', labelsize=8) 
    plt.ylabel('Freq')
    plt.title('Frequencies of best performance on density property with different RUs')          
    
    plt.legend()
    plt.savefig(save_fig)
    

def his(x,save_fig,title):
    mu =np.mean(x) #计算均值
    sigma =np.std(x)
    fig = plt.figure(figsize=(8,4)) 
    n, bins, patches = plt.hist(x, bins=40, edgecolor='whitesmoke',
                            density = 1, 
                            color='steelblue',
                            alpha = 0.5)
    my_x_ticks = np.arange(0.8, 2.1, 0.1)
    #plt.contour(plt.hist(x, bins=30)[1], mu, colors='red')


    # 去除图形顶部边界和右边界的刻度
    plt.tick_params(top=False, right=False)
    plt.xlabel('Glass Transition Temperature(℃)')
    plt.xticks(my_x_ticks)
    plt.ylabel('Freq')
 
    
    plt.title(title,fontweight = "bold")
    
    plt.savefig(save_fig)


if __name__=='__main__':
    r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r234,r567,r8910 = [],[],[],[],[],[],[],[],[],[],[],[],[]
    # Open file  
    
    idx = 0
    with open(open_file) as file_obj: 
        
        # Create reader object by passing the file  
        # object to reader method 
        reader_obj = csv.reader(file_obj)
        header = next(reader_obj)
        #print(reader_obj)
        # Iterate over each row in the csv  
        # file using reader object 
        for row in reader_obj: 
            idx += 1
            min_pos,minres = check_dis(row)
            for pos in min_pos:
                if pos == 1:
                    r1.append(minres)
                elif pos == 2:
                    r2.append(minres)
                    r234.append(minres)
                elif pos == 3:
                    r3.append(minres)
                    r234.append(minres)
                elif pos == 4:
                    r4.append(minres)
                    r234.append(minres)
                elif pos == 5:
                    r5.append(minres)
                    r567.append(minres)
                elif pos == 6:
                    r6.append(minres)
                    r567.append(minres)
                elif pos == 7:
                    r7.append(minres)
                    r567.append(minres)
                elif pos == 8:
                    r8.append(minres)
                    r8910.append(minres)
                elif pos == 9:
                    r9.append(minres)
                    r8910.append(minres)
                else:
                    r10.append(minres)
                    r8910.append(minres)
        group10 = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10]
        print(group10[0])
        group4 = [r1,r234,r567,r8910]
        for i in range(10):
            save_fig = './rnn/img/tg_group10/{}_{}sns.jpg'.format(task,i+1)
            title = 'Frequencies of best performance on tg property with {} repeat unit'.format(i+1)
            #title = 'Frequencies of best performance on tg property of group {}'.format(i+1)
            sns.set(font_scale=2)
            try:
                    
                displot = sns.displot(data=pd.DataFrame(group10[i]),  col_wrap=5, kind='hist', kde=True,  legend=False)#, palette="light:m_r")
                displot.save(save_fig)
            except:
                pass
            #his(group4[i],save_fig,title)
            