from gnn import MainGNN
import torch
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
        rep_times = [[1,3,10]]
        
        model_types = ["gin-virtual"]
        test_rep = [16,17,18,19,20]
        for model_type in model_types:        
            for r in rep_times:
                #r = MainRNN(train_rep=r, task_name = 'tg', test_rep = test_rep, seed_num = 10, pad_size =15, _use_ck = True)
                g = MainGNN(train_rep=r, task_name = 'tg', test_rep = test_rep, model_type = 'gin-virtual',seed_num = 5,_use_ck = True, _use_param = True)
                #g.main()
                #t = MainTransformer(r, task_name = 'tg', seed_num = 5, _use_ck = False)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                get_embedding(g)
                #t.main()