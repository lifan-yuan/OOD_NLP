# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/steps-nlu.py --model_name t5 --scale small --dataset_name amazon >./log/steps/SentimentAnalysis-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/steps-nlu.py --model_name t5 --scale small --dataset_name civil_comments >./log/steps/ToxicDetection-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/steps-nlu.py --model_name t5 --scale small --dataset_name mnli >./log/steps/NaturalLanguageInference-small.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/steps-nlu.py --model_name t5 --scale small --dataset_name hellaswag >./log/steps/CommonSense-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/steps-ner.py --model_name deberta --scale small --dataset_name fewnerd >./log/steps/NameEntityRecognition-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/steps-qa.py --model_name t5 --scale small --dataset_name squad >./log/steps/QuestionAnswering-small.log 2>&1 &




# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/steps-nlu.py --model_name t5 --scale base --dataset_name amazon >./log/steps/SentimentAnalysis-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/steps-nlu.py --model_name t5 --scale base --dataset_name civil_comments >./log/steps/ToxicDetection-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/steps-nlu.py --model_name t5 --scale base --dataset_name mnli >./log/steps/NaturalLanguageInference-base.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/steps-nlu.py --model_name t5 --scale base --dataset_name hellaswag >./log/steps/CommonSense-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/steps-ner.py --model_name deberta --scale base --dataset_name fewnerd >./log/steps/NameEntityRecognition-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/steps-qa.py --model_name t5 --scale base --dataset_name squad >./log/steps/QuestionAnswering-base.log 2>&1 &




# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/steps-nlu.py --model_name t5 --scale large --dataset_name amazon >./log/steps/SentimentAnalysis-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/steps-nlu.py --model_name t5 --scale large --dataset_name civil_comments >./log/steps/ToxicDetection-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/steps-nlu.py --model_name t5 --scale large --dataset_name mnli >./log/steps/NaturalLanguageInference-large.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup python src/analysis/steps-nlu.py --model_name t5 --scale large --dataset_name hellaswag >./log/steps/CommonSense-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/steps-ner.py --model_name deberta --scale large --dataset_name fewnerd >./log/steps/NameEntityRecognition-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6 nohup python src/analysis/steps-qa.py --model_name t5 --scale large --dataset_name squad >./log/steps/QuestionAnswering-large.log 2>&1 &