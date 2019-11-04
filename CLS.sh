#python cls_classification.py --cuda 0 --start 0 --end 4000 --model CLS_bert_weight_0.0001_clean_pooler_fold_1
#python cls_classification.py --cuda 0 --start 4000 --end 8000 --model CLS_bert_weight_0.0001_clean_pooler_fold_2
#python cls_classification.py --cuda 0 --start 8000 --end 12000 --model CLS_bert_weight_0.0001_clean_pooler_fold_3
#python cls_classification.py --cuda 0 --start 12000 --end 16000 --model CLS_bert_weight_0.0001_clean_pooler_fold_4
#python cls_classification.py --cuda 0 --start 16000 --end 20000 --model CLS_bert_weight_0.0001_clean_pooler_fold_5
python cls_classification.py --cuda 0 --start 0 --end 4000 --model CLS_bert_weight_0.0001_clean_pooler_fold_1 --submit True
python cls_classification.py --cuda 0 --start 4000 --end 8000 --model CLS_bert_weight_0.0001_clean_pooler_fold_2 --submit True
python cls_classification.py --cuda 0 --start 8000 --end 12000 --model CLS_bert_weight_0.0001_clean_pooler_fold_3 --submit True
python cls_classification.py --cuda 0 --start 12000 --end 16000 --model CLS_bert_weight_0.0001_clean_pooler_fold_4 --submit True
python cls_classification.py --cuda 0 --start 16000 --end 20000 --model CLS_bert_weight_0.0001_clean_pooler_fold_5 --submit True
