#from gnn import MainGNN
from tsfm_main import MainTransformer
#from rnn import MainRNN

def get_embedding(MainGNN):
    # test_similarity: uncomment the 'self.similarity(h_node,h_graph)' line in model.py
    # obtain graph representation: uncomment #self.save_h_graph(h_graph)
    MainGNN.add_arg()
    model,device,_,_ = MainGNN.model_seed(1)
    print("Start calculating...")
    MainGNN.test(model,device,[0,101])
    
    
if __name__ == "__main__":
   
    tasks = ['tg']
    for task_name in tasks:
        # train_rep can be a list of ints/lists, 
        # int numbers means use single dataset, 
        # list means use concatenate datasets of different repeating times
        rep_times = [7,8]
             
        for r in rep_times:
            #r = MainRNN(train_rep=r, task_name = 'tg', test_rep = test_rep, seed_num = 10, pad_size =15, _use_ck = True)
            #g = MainGNN(train_rep=r, task_name = task_name, model_type = model_type,test_rep = test_rep)
            t = MainTransformer(r, task_name = 'tg', seed_num = 5, _use_ck = False)

            t.main()