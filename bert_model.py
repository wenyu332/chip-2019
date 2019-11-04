import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
import torch
from torch import nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertPooler
class Bert_Classification(nn.Module):
    def __init__(self,config,output_size):
        super(Bert_Classification, self).__init__()
        self.word_embeds = BertModel(config)
        self.word_embeds.load_state_dict(torch.load('/home/fayan/lxl/chip_2019/weights/robert/RoBertLarge_weight.bin'))
        #self.pooler = BertPooler(config)
        #self.dropout=nn.Dropout(0.5) 
        self.classification=nn.Linear(config.hidden_size,output_size)
    def forward(self, sentences,attention_mask,flag,labels=None):
        _, pooled_output= self.word_embeds(sentences, attention_mask=attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.pooler(pooled_output)
        #print('model_shape:',_.size())
        #exit(1)
        '''
        if flag=='CLS':
            pooled_output=pooled_output[:,0]
        elif flag=='MAX':
            pooled_output=pooled_output.max(1)[0]
        elif flag=='MEAN':
            pooled_output=pooled_output.mean(1)
        # print(pooled_output.size())
        '''
        #pooled_output=self.dropout(pooled_output)
        logits=self.classification(pooled_output)
        # print('logits:',logits.size())
        # print('labels:', labels.size(),labels)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class LSTM_Classification(nn.Module):
    def __init__(self,config,output_size):
        super(LSTM_Classification, self).__init__()
        self.word_embeds = BertModel(config)
        self.word_embeds.load_state_dict(torch.load('/home/fayan/lxl/chip_2019/weights/robert/RoBertLarge_weight.bin'))
        #self.pooler = BertPooler(config)
        self.lstm=nn.LSTM(config.hidden_size,128,bidirectional=True,batch_first=True)
        #self.dropout=nn.Dropout(0.5) 
        self.classification=nn.Linear(config.hidden_size+128*6,output_size)
    def forward(self, sentences,attention_mask,flag,labels=None):
        _, pooled_output= self.word_embeds(sentences, attention_mask=attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.pooler(pooled_output)
        h_lstm,(hidden_state,cell_state)=self.lstm(_)
        hh_lstm=torch.cat((hidden_state[0],hidden_state[1]),dim=1)
        avg_pool=torch.mean(h_lstm,1)
        max_pool,_=torch.max(h_lstm,1)
        #print(avg_pool.size(),max_pool.size(),hh_lstm.size(),pooled_output.size())
        pooled_output=torch.cat((avg_pool,max_pool,pooled_output,hh_lstm),1)
        logits=self.classification(pooled_output)
        # print('logits:',logits.size())
        # print('labels:', labels.size(),labels)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
class SpatialDropout1D(nn.Module):
    def __init__(self, p=0.5):
        super(SpatialDropout1D, self).__init__()
        self.p = p
        self.dropout2d = nn.Dropout2d(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, maxlen, 1, embed_size)
        x = x.permute(0, 3, 2, 1)  # (N, embed_size, 1, maxlen)
        x = self.dropout2d(x)  # (N, embed_size, 1, maxlen)
        x = x.permute(0, 3, 2, 1)  # (N, maxlen, 1, embed_size)
        x = x.squeeze(2)  # (N, maxlen, embed_size)

        return x
LSTM_UNITS = 128
CHANNEL_UNITS = 128
class CNN_Classification(nn.Module):
    def __init__(self,config,output_size):
        super(CNN_Classification, self).__init__()
        self.word_embeds = BertModel(config)
        self.word_embeds.load_state_dict(torch.load('/home/fayan/lxl/chip_2019/weights/robert/RoBertLarge_weight.bin'))
        #self.pooler = BertPooler(config)
        self.embedding_dropout = SpatialDropout1D(config.hidden_dropout_prob)
        filters = [3, 4, 5]
        self.conv_layers = nn.ModuleList()
        for filter_size in filters:
            conv_block = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size,
                    CHANNEL_UNITS,
                    kernel_size=filter_size,
                    padding=1,
                ),
                # nn.BatchNorm1d(CHANNEL_UNITS),
                # nn.ReLU(inplace=True),
            )
            self.conv_layers.append(conv_block)
        #self.dropout=nn.Dropout(0.5) 
        self.classification=nn.Linear(config.hidden_size+CHANNEL_UNITS*6,output_size)
    def forward(self, sentences,attention_mask,flag,labels=None):
        _, pooled_output= self.word_embeds(sentences, attention_mask=attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.pooler(pooled_output)
        h_embedding = _.permute(0, 2, 1)
        feature_maps= []
        for layer in self.conv_layers:
            h_x= layer(h_embedding)
            feature_maps.append(
                F.max_pool1d(h_x, kernel_size=h_x.size(2)).squeeze()
            )
            feature_maps.append(
                F.avg_pool1d(h_x, kernel_size=h_x.size(2)).squeeze()
            )
        conv_features= torch.cat(feature_maps, 1)
        #print(conv_features.size())
        pooled_output= torch.cat((conv_features, pooled_output), 1)
        #print(pooled_output.size())
        logits=self.classification(pooled_output)
        # print('logits:',logits.size())
        # print('labels:', labels.size(),labels)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class LSTMGRU_Classification(nn.Module):
    def __init__(self,config,output_size):
        super(LSTMGRU_Classification, self).__init__()
        self.word_embeds = BertModel(config)
        self.word_embeds.load_state_dict(torch.load('/home/fayan/lxl/chip_2019/weights/robert/RoBertLarge_weight.bin'))
        #self.pooler = BertPooler(config)
        self.lstm=nn.LSTM(config.hidden_size,LSTM_UNITS,bidirectional=True,batch_first=True)
        self.gru=nn.GRU(LSTM_UNITS*2,LSTM_UNITS,bidirectional=True,batch_first=True)
        #self.dropout=nn.Dropout(0.5) 
        self.classification=nn.Linear(config.hidden_size+LSTM_UNITS*6,output_size)
    def forward(self, sentences,attention_mask,flag,labels=None):
        _, pooled_output= self.word_embeds(sentences, attention_mask=attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.pooler(pooled_output)
        h_lstm,(hidden_state,cell_state)=self.lstm(_)
        h_gru,hh_gru=self.gru(h_lstm)
        hh_gru=hh_gru.view(-1,2*LSTM_UNITS)
        avg_pool=torch.mean(h_gru,1)
        max_pool,_=torch.max(h_gru,1)
        #print(avg_pool.size(),max_pool.size(),hh_gru.size(),pooled_output.size())
        pooled_output=torch.cat((avg_pool,max_pool,pooled_output,hh_gru),1)
        #print(pooled_output.size())
        logits=self.classification(pooled_output)
        # print('logits:',logits.size())
        # print('labels:', labels.size(),labels)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class Dense_Classification(nn.Module):
    def __init__(self,config,output_size):
        super(Dense_Classification, self).__init__()
        self.word_embeds = BertModel(config)
        self.word_embeds.load_state_dict(torch.load('/home/fayan/lxl/chip_2019/weights/robert/RoBertLarge_weight.bin'))
        #self.pooler = BertPooler(config)
        self.linear=nn.Linear(config.hidden_size,128)
        #self.dropout=nn.Dropout(0.5) 
        self.classification=nn.Linear(config.hidden_size+128,output_size)
    def forward(self, sentences,attention_mask,flag,labels=None):
        _, pooled_output= self.word_embeds(sentences, attention_mask=attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.pooler(pooled_output)
        linear=self.linear(pooled_output)
        #print(avg_pool.size(),max_pool.size(),hh_gru.size(),pooled_output.size())
        pooled_output=torch.cat((linear,pooled_output),1)
        #print(pooled_output.size())
        logits=self.classification(pooled_output)
        # print('logits:',logits.size())
        # print('labels:', labels.size(),labels)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
