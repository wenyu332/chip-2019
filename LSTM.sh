python LSTM_classification.py --cuda 1 --start 0 --end 4000 --model LSTM_bert_weight_0.0001_fold_1
python LSTM_classification.py --cuda 1 --start 4000 --end 8000 --model LSTM_bert_weight_0.0001_fold_2
python LSTM_classification.py --cuda 1 --start 8000 --end 12000 --model LSTM_bert_weight_0.0001_fold_3
python LSTM_classification.py --cuda 1 --start 12000 --end 16000 --model LSTM_bert_weight_0.0001_fold_4
python LSTM_classification.py --cuda 1 --start 16000 --end 20000 --model LSTM_bert_weight_0.0001_fold_5
python LSTM_classification.py --cuda 1 --start 0 --end 4000 --model LSTM_bert_weight_0.0001_fold_1 --submit True
python LSTM_classification.py --cuda 1 --start 4000 --end 8000 --model LSTM_bert_weight_0.0001_fold_2 --submit True
python LSTM_classification.py --cuda 1 --start 8000 --end 12000 --model LSTM_bert_weight_0.0001_fold_3 --submit True
python LSTM_classification.py --cuda 1 --start 12000 --end 16000 --model LSTM_bert_weight_0.0001_fold_4 --submit True
python LSTM_classification.py --cuda 1 --start 16000 --end 20000 --model LSTM_bert_weight_0.0001_fold_5 --submit True
