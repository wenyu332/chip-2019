python CNN_classification.py --cuda 0 --start 0 --end 4000 --model CNN_bert_weight_fold_1
python CNN_classification.py --cuda 0 --start 4000 --end 8000 --model CNN_bert_weight_fold_2
python CNN_classification.py --cuda 0 --start 8000 --end 12000 --model CNN_bert_weight_fold_3
python CNN_classification.py --cuda 0 --start 12000 --end 16000 --model CNN_bert_weight_fold_4
python CNN_classification.py --cuda 0 --start 16000 --end 20000 --model CNN_bert_weight_fold_5
python CNN_classification.py --cuda 0 --start 0 --end 4000 --model CNN_bert_weight_fold_1 --submit True
python CNN_classification.py --cuda 0 --start 4000 --end 8000 --model CNN_bert_weight_fold_2 --submit True
python CNN_classification.py --cuda 0 --start 8000 --end 12000 --model CNN_bert_weight_fold_3 --submit True
python CNN_classification.py --cuda 0 --start 12000 --end 16000 --model CNN_bert_weight_fold_4 --submit True
python CNN_classification.py --cuda 0 --start 16000 --end 20000 --model CNN_bert_weight_fold_5 --submit True
