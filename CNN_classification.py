from utils import load_vocab, read_corpus,load_model, save_model
import training_args as args
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert import BertConfig
from new_bert_model import CNN_Classification
import os
from sklearn import metrics
import csv
import numpy as np
import random
from torch.optim import Adam
from optimizer import Lookahead,RAdam
print("cls classification!")
import argparse
parser=argparse.ArgumentParser(description="chip-2019")
parser.add_argument("--model",type=str,default="CLS",help="chose a model:cls, max, mean")
parser.add_argument("--cuda",type=str,default="0",help="chose a cuda numbering")
parser.add_argument("--seed",type=int,default=42,help="set model seed")
parser.add_argument("--submit",type=bool,default=False,help="set model seed")
parser.add_argument("--lr",type=float,default=1e-5,help="set train learing rate")
parser.add_argument("--train_batch_size",type=int,default=16,help="set train batch size")
parser.add_argument("--start",type=int,default=0,help="select cross validation start point")
parser.add_argument("--end",type=int,default=4000,help="select cross validation end point")
arg=parser.parse_args()
model_name=arg.model
cuda=arg.cuda
seed=arg.seed
submit=arg.submit
start=arg.start
end=arg.end
lr=arg.lr
train_batch_size=arg.train_batch_size
print(model_name,cuda,seed,submit,train_batch_size,lr,start,end)
os.environ["CUDA_VISIBLE_DEVICES"]=cuda
torch.manual_seed(seed) # cpu
torch.cuda.manual_seed(seed) #gpu
np.random.seed(seed) #numpy
random.seed(seed) #random and transforms
torch.backends.cudnn.deterministic=True # cu
#torch.backends.cudnn.deterministic=True # cu

def dev(model, dev_loader, config):
    model.eval()
    with torch.no_grad():
        length = 0
        true_labels=[]
        predict_labels=[]
        dev_loss=[]
        for i, batch in enumerate(dev_loader):
            input, mask ,tags = batch
            input, mask, tags = Variable(input), Variable(mask),Variable(tags)
            if args.use_cuda:
               input, mask, tags = input.cuda(), mask.cuda(), tags.cuda()
            loss,preds = model(input, mask,'CLS',tags)
            dev_loss.append(loss.item())
            predicts=torch.argmax(preds,dim=1)
            true_labels+=tags.cpu().numpy().tolist()
            predict_labels+=predicts.cpu().numpy().tolist()
    #print(true_labels)
    #print(predict_labels)
    m_acc=metrics.accuracy_score(true_labels,predict_labels)
    m_precision=metrics.precision_score(true_labels,predict_labels,average="macro")
    m_recall=metrics.recall_score(true_labels,predict_labels,average="macro")
    m_f1=metrics.f1_score(true_labels,predict_labels,average="macro")
    print(sum(dev_loss)/len(dev_loss),m_acc,m_precision,m_recall,m_f1)
    return m_f1
def test(model, dev_loader, use_cuda,ids):
    model.eval()
    with torch.no_grad():
        predict_labels=[]
        for i, batch in enumerate(dev_loader):
            input, mask= batch
            input, mask= Variable(input), Variable(mask)
            if use_cuda:
               input, mask= input.cuda(), mask.cuda()
            preds = model(input, mask,'CLS')
            predicts=torch.argmax(preds,dim=1)
            predict_labels+=predicts.cpu().numpy().tolist()
    with open(model_name+"_"+str(seed)+"_"+str(train_batch_size)+"_"+str(lr)+"/"+model_name+'.csv','w',encoding='utf-8',newline='') as f:
        f_csv=csv.writer(f)
        f_csv.writerow(['id','label'])
        for i,data in enumerate(predict_labels):
            f_csv.writerow([ids[i],predict_labels[i]])
        f_csv.writerow([])
def main():
    vocab = load_vocab(args.vocab_file)
    label_dic = load_vocab(args.label_file)

    index2label = {v: k for k, v in label_dic.items()}
    tagset_size = len(label_dic)

    train_data= read_corpus(args.train_path, max_length=args.max_seq_length, label_dic=label_dic, vocab=vocab)
    #dev_data= read_corpus(args.dev_path, max_length=args.max_seq_length, label_dic=label_dic, vocab=vocab)
    print(len(train_data))
    data=train_data
    train_data=data[:start]+data[end:]
    dev_data=data[start:end]
    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])
    train_dataset = TensorDataset(train_ids, train_masks,train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)


    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])
    dev_dataset = TensorDataset(dev_ids, dev_masks,dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)
    num_train_optimization_steps = int(
        len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.epochs

    config=BertConfig(args.bert_config_json)
    model=CNN_Classification(config,tagset_size)
    if args.use_cuda:
        model = model.cuda()
    if submit:
        assert args.submit_path is not None
        submit_data,ids= read_corpus(args.submit_path, max_length=args.max_seq_length,label_dic=None, vocab=vocab)
        submit_ids = torch.LongTensor([temp.input_id for temp in submit_data])
        submit_masks = torch.LongTensor([temp.input_mask for temp in submit_data])
        submit_dataset = TensorDataset(submit_ids, submit_masks)
        submit_loader = DataLoader(submit_dataset, shuffle=False, batch_size=args.eval_batch_size)
        model = load_model(model, path=model_name+"_"+str(seed)+"_"+str(train_batch_size)+"_"+str(lr))
        test(model, submit_loader, args.use_cuda,ids)
    else:
        model.train()
        bert_param_optimizer = list(model.word_embeds.named_parameters())
        cnn_param_optimizer=list(model.conv_layers.named_parameters())
        classification_param_optimizer = list(model.classification.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0001, 'lr': lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': lr},
            {'params': [p for n, p in cnn_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': lr*5},
            {'params': [p for n, p in cnn_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': lr*5},
            {'params': [p for n, p in classification_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': lr*5},
            {'params': [p for n, p in classification_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': lr*5},
        ]
        optimizer = Adam(optimizer_grouped_parameters)
        #optimizer = Lookahead(base_optimizer=base_optimizer,k=5,alpha=0.5)
        eval_f1 = 0
        step=0
        count=0
        total_step=args.epochs*len(train_loader)
        warm_up_step=int(total_step*0.1)
        print(total_step,warm_up_step)
        for epoch in range(args.epochs):
            #print("epoch=",epoch)
            for i in optimizer.param_groups:
                print(i['lr'])
            train_loss=[]
            for i, batch in enumerate(train_loader):
                step += 1
                if step<warm_up_step:
                #i['lr']=(i['lr']*step)/warm_up_step
                    index=0
                    for i in optimizer.param_groups:
                        
                        i['lr']=(lr/2*step)/warm_up_step+(lr/2)
                        index+=1
                elif step==warm_up_step:
                    index=0
                    for i in optimizer.param_groups:

                        i['lr']=lr
                        index+=1
                    '''
                    for i in optimizer.param_groups:
                        if index<=1:
                            i['lr']=(lr/2*step)/warm_up_step+(lr/2)
                        else:
                            i['lr']=5*(lr/2*step)/warm_up_step+(lr/2)
                        index+=1
                    '''
                model.zero_grad()
                input1, mask1, tags = batch
                input1, mask1, tags = Variable(input1),Variable(mask1),Variable(tags)
                if args.use_cuda:
                    input1, mask1, tags = input1.cuda(), mask1.cuda(), tags.cuda()

                loss,_= model(input1, mask1,'CLS',tags)
                train_loss.append(loss.item())
                # loss = model.loss(feats, masks, tags)
                loss.backward()
                optimizer.step()
               # print(step)
                if step % 100 == 0:
                    print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, sum(train_loss)/len(train_loss)))
                train_loss.clear()
            f1 = dev(model, dev_loader, args.use_cuda)
            model.train()
            if eval_f1 < f1:
                save_model(model, step,f1,path=model_name+"_"+str(seed)+"_"+str(train_batch_size)+"_"+str(lr),name=model_name)
                eval_f1=f1
                count=0
            else:
                count+=1
                if count>=2:
                    index=0
                    for i in optimizer.param_groups:
                        i['lr']=lr*(0.5**count)
                        index+=1
                if count>=5:
                    exit(1)
if __name__=='__main__':
    main()
