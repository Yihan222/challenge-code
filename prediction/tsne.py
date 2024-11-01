from gnn import MainGNN
import os
import os.path as osp
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import seaborn as sns



#from rnn import MainRNN
task_name = 'tg'
test_index = 500
colors = ['maroon','tan','lightcoral','cyan','lightseagreen','slategray','steelblue','violet','navy','mediumpurple']

def produce_embedding(MainGNN):
    # test_similarity: uncomment the 'self.similarity(h_node,h_graph)' line in model.py
    # obtain graph representation: uncomment #self.save_h_graph(h_graph)
    MainGNN.add_arg()
    model,device,_,_ = MainGNN.model_seed(1)
    print("Start calculating...")
    MainGNN.test(model,device,test_index) #the index here refers to the test data size you want
    print('********** Embeddings Saved ! **********')

def get_embedding(r,tr):
    if isinstance(r,list):
        rname = int('{}{}'.format(r[0],r[-1]))
    else:
        rname = r
    emb_path = osp.join('./embeddings/h_graph/gin-virtual/ptrain',str(rname))
    if not osp.exists(emb_path):
        os.makedirs(emb_path)
    emb_name = osp.join(emb_path, '{}.npy'.format(tr))
    if not osp.exists(emb_name):
        # have to produce embedding first
        g = MainGNN(train_rep=r, task_name = task_name, model_type = 'gin-virtual',test_rep = [tr])
        produce_embedding(g)
    else:
        print('********** Embeddings Loaded ! **********')
    embedding = np.load(emb_name)

    return embedding[:test_index]

def paint_seperate(rep_times,tr):
    for i in range(len(rep_times)):
        r = rep_times[i]
        color = colors[i]
        if isinstance(r,list):
            rname = 'merge_{}_{}'.format(r[0],r[-1])
        else:
            rname = 'single_{}'.format(r)
        embedding = get_embedding(r,tr) #diagonal has best performance for single dataset

        print("start fitting t-SNE")
        tSNE = TSNE(
            perplexity=50,
            metric='euclidean',
            init='pca'
        )
        tSNE = tSNE.fit_transform(embedding)
        print("finish t-SNE")
        fig, ax = plt.subplots(figsize=(15, 15))

        ax.scatter(embedding[:, 0], embedding[:, 1], s=100, edgecolors='None', linewidths=0.4, c=color)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text('Graph representations of test {} in Model trained on {} dataset(s)'.format(tr, rname))
        ax.axis('off')
        tsne_save_path = './embeddings/h_graph/gin-virtual/train/{}/{}_{}/tsne_{}.png'.format(tr,test_index[0],test_index[1],rname)
        plt.savefig(tsne_save_path, bbox_inches='tight')

def new_test(r,test_rep_time):
    tSNE = TSNE(
        n_components=2,       
        perplexity = 30, 
        random_state=0
        )
    tsne_embeddings = []
    embeddings = []
    
    print("start fitting t-SNE")
    for tr in test_rep_times:
        embedding = get_embedding(r,tr)
        embeddings.append(embedding)
        if tr == 1:
            tsne_embeddings = embedding
        else:
            tsne_embeddings = np.append(tsne_embeddings,embedding,axis = 0)

    tSNE.fit(tsne_embeddings)
    print("finish t-SNE fitting!")

    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(10):
        tr = test_rep_times[i]
        color = colors[i]
        cur_tsne = tSNE.fit_transform(embeddings[i])
        ax.scatter(cur_tsne[:, 0], cur_tsne[:, 1], s=100, edgecolors='None', linewidths=0.4, c=color, label='{}'.format(tr))

    if isinstance(r,list):
        title = "Merge {}_{} Model".format(r[0],r[-1])
        tsne_save_path = './embeddings/h_graph/gin-virtual/test/tsne_{}_{}.png'.format(r[0],r[-1])
    else:
        title = 'Single {} Model'.format(r)
        tsne_save_path = './embeddings/h_graph/gin-virtual/test/tsne_{}.png'.format(r)
    ax.title.set_text('Graph representations of train {} on different test sets'.format(title))
    ax.axis('off')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(tsne_save_path) 

    
def main_test(r,test_rep_times):
    tSNE = TSNE(
        n_components=2,       
        perplexity = 40, 
        random_state=0,
        init = 'pca'
    )
    embeddings = []
    
    print("start fitting t-SNE")
    for tr in test_rep_times:
        embedding = get_embedding(r,tr)
        embeddings.append(embedding)
        if tr == 1:
            tsne_embeddings = embedding
        else:
            tsne_embeddings = np.append(tsne_embeddings,embedding,axis = 0)

    tSNE.fit(tsne_embeddings)
    print("finish t-SNE fitting!")

    fig, ax = plt.subplots(figsize=(15, 15))
    results = {
        'tsne_1':[],
        'tsne_2': [], 
        'label': [],
        }
    df=pd.DataFrame(results)
    #for i in range(len(test_rep_times)):
    for i in range(10):
        tr = test_rep_times[i]
        tsne_result = tSNE.fit_transform(embeddings[i])
        new_df = pd.DataFrame({'tsne_1':list(tsne_result[:,0]),'tsne_2':list(tsne_result[:,1]),'label':['test_{}'.format(tr)]*len(tsne_result[:,0])})
        df = pd.concat([df, new_df], ignore_index=True)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',palette='rocket_r',data=df, ax=ax,s=100)
    if isinstance(r,list):
        title = "Merge {}_{} Model".format(r[0],r[-1])
        tsne_save_path = './embeddings/h_graph/gin-virtual/test/tsne_{}_{}.png'.format(r[0],r[-1])
    else:
        title = 'Single {} Model'.format(r)
        tsne_save_path = './embeddings/h_graph/gin-virtual/test/tsne_{}.png'.format(r)
    ax.title.set_text('Graph representations of train {} on different test sets'.format(title))
    ax.axis('off')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(tsne_save_path) 

if __name__ == "__main__":
   

    # train_rep can be a list of ints/lists, 
    # int numbers means use single dataset, 
    # list means use mergeenate datasets of different repeating times

    rep_times = [3,4,5,6,7,8,9,10,[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10]]
    test_rep_times = [1,2,3,4,5,6,7,8,9,10]

    for r in rep_times:
        main_test(r,test_rep_times)



            