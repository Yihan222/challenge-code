from rnn import MainRNN

def get_embedding(MainGNN):
    # test_similarity: uncomment the 'self.similarity(h_node,h_graph)' line in model.py
    # obtain graph representation: uncomment #self.save_h_graph(h_graph)
    MainGNN.add_arg()
    model,device,_,_ = MainGNN.model_seed(1)
    print("Start calculating...")
    MainGNN.test(model,device,[0,101])
    
    
if __name__ == "__main__":
   
    tasks = ['tg']
    test_rep = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    for task_name in tasks:
        # train_rep can be a list of ints/lists, 
        # int numbers means use single dataset, 
        # list means use concatenate datasets of different repeating times
        rep_times = [[3,5],[3,8]]
        
        model_types = ["rnn"]

        for model_type in model_types:        
            for rep in rep_times:
                r = MainRNN(train_rep=rep, task_name = 'tg', test_rep = test_rep, seed_num=3,pad_size=15, _use_ck = True)
                #g = MainGNN(train_rep=r, task_name = task_name, model_type = model_type,test_rep = test_rep)

                r.main()