python Dense_classification.py --cuda 3 --start 0 --end 4000 --model Dense_bert_weight_0.0001_fold_1
python Dense_classification.py --cuda 3 --start 4000 --end 8000 --model Dense_bert_weight_0.0001_fold_2
python Dense_classification.py --cuda 3 --start 8000 --end 12000 --model Dense_bert_weight_0.0001_fold_3
python Dense_classification.py --cuda 3 --start 12000 --end 16000 --model Dense_bert_weight_0.0001_fold_4
python Dense_classification.py --cuda 3 --start 16000 --end 20000 --model Dense_bert_weight_0.0001_fold_5
python Dense_classification.py --cuda 3 --start 0 --end 4000 --model Dense_bert_weight_0.0001_fold_1 --submit True
python Dense_classification.py --cuda 3 --start 4000 --end 8000 --model Dense_bert_weight_0.0001_fold_2 --submit True
python Dense_classification.py --cuda 3 --start 8000 --end 12000 --model Dense_bert_weight_0.0001_fold_3 --submit True
python Dense_classification.py --cuda 3 --start 12000 --end 16000 --model Dense_bert_weight_0.0001_fold_4 --submit True
python Dense_classification.py --cuda 3 --start 16000 --end 20000 --model Dense_bert_weight_0.0001_fold_5 --submit True
