# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/shots-nlu.py --model_name t5 --scale small --dataset_name amazon >./log/shots/SentimentAnalysis-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/shots-nlu.py --model_name t5 --scale small --dataset_name civil_comments >./log/shots/ToxicDetection-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/shots-nlu.py --model_name t5 --scale small --dataset_name mnli >./log/shots/NaturalLanguageInference-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/shots-nlu.py --model_name t5 --scale small --dataset_name hellaswag >./log/shots/CommonSense-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/shots-ner.py --model_name deberta --scale small --dataset_name fewnerd >./log/shots/NameEntityRecognition-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/shots-qa.py --model_name t5 --scale small --dataset_name squad >./log/shots/QuestionAnswering-small.log 2>&1 &




# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-nlu.py --model_name t5 --scale base --dataset_name amazon >./log/shots/SentimentAnalysis-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-nlu.py --model_name t5 --scale base --dataset_name civil_comments >./log/shots/ToxicDetection-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/shots-nlu.py --model_name t5 --scale base --dataset_name mnli >./log/shots/NaturalLanguageInference-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/shots-nlu.py --model_name t5 --scale base --dataset_name hellaswag >./log/shots/CommonSense-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/shots-ner.py --model_name deberta --scale base --dataset_name fewnerd >./log/shots/NameEntityRecognition-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/shots-qa.py --model_name t5 --scale base --dataset_name squad >./log/shots/QuestionAnswering-base.log 2>&1 &




# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-nlu.py --model_name t5 --scale large --dataset_name amazon >./log/shots/SentimentAnalysis-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/shots-nlu.py --model_name t5 --scale large --dataset_name civil_comments >./log/shots/ToxicDetection-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/shots-nlu.py --model_name t5 --scale large --dataset_name mnli >./log/shots/NaturalLanguageInference-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5 nohup python src/analysis/shots-nlu.py --model_name t5 --scale large --dataset_name hellaswag >./log/shots/CommonSense-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-ner.py --model_name deberta --scale large --dataset_name fewnerd >./log/shots/NameEntityRecognition-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5,6,7 nohup python src/analysis/shots-qa.py --model_name t5 --scale large --dataset_name squad >./log/shots/QuestionAnswering-large.log 2>&1 &






# t5-3b 0-shot
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name amazon >./log/shots_3b/SentimentAnalysis-t5-3b-0-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name civil_comments >./log/shots_3b/ToxicDetection-t5-3b-0-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name mnli >./log/shots_3b/NaturalLanguageInference-t5-3b-0-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1,2 nohup python src/analysis/shots-generation.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name fewnerd >./log/shots_3b/NameEntityRecognition-t5-3b-0-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-qa.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name squad >./log/shots_3b/QuestionAnswering-t5-3b-0-shots.log 2>&1 &

# t5-3b 5-shot
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 5 --dataset_name amazon >./log/shots_3b/SentimentAnalysis-t5-3b-5-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 5 --dataset_name civil_comments >./log/shots_3b/ToxicDetection-t5-3b-5-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 5 --dataset_name mnli >./log/shots_3b/NaturalLanguageInference-t5-3b-5-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python src/analysis/shots-generation.py --model_name t5 --scale 3b --repeats 1 --shots_list 5 --dataset_name fewnerd >./log/shots_3b/NameEntityRecognition-t5-3b-5-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python src/analysis/shots-qa.py --model_name t5 --scale 3b --repeats 1 --shots_list 5 --dataset_name squad >./log/shots_3b/QuestionAnswering-t5-3b-5-shots.log 2>&1 &

# t5-3b full data
# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list -1 --dataset_name amazon >./log/shots_3b/SentimentAnalysis-t5-3b-full-data.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list -1 --dataset_name civil_comments >./log/shots_3b/ToxicDetection-t5-3b-full-data.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list -1 --dataset_name mnli >./log/shots_3b/NaturalLanguageInference-t5-3b-full-data.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python src/analysis/shots-generation.py --model_name t5 --scale 3b --repeats 1 --shots_list -1 --dataset_name fewnerd >./log/shots_3b/NameEntityRecognition-t5-3b-full-data.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python src/analysis/shots-qa.py --model_name t5 --scale 3b --repeats 1 --shots_list -1 --dataset_name squad >./log/shots_3b/QuestionAnswering-t5-3b-full-data.log 2>&1 &


# t0-3b 0-shot
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-nlu.py --model_name t0 --scale 3b --repeats 1 --shots_list 0 --dataset_name amazon >./log/shots_3b/SentimentAnalysis-t0-3b-0-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-nlu.py --model_name t0 --scale 3b --repeats 1 --shots_list 0 --dataset_name civil_comments >./log/shots_3b/ToxicDetection-t0-3b-0-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-nlu.py --model_name t0 --scale 3b --repeats 1 --shots_list 0 --dataset_name mnli >./log/shots_3b/NaturalLanguageInference-t0-3b-0-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/shots-generation.py --model_name t0 --scale 3b --repeats 1 --shots_list 0 --dataset_name fewnerd >./log/shots_3b/NameEntityRecognition-t0-3b-0-shots.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python src/analysis/shots-qa.py --model_name t0 --scale 3b --repeats 1 --shots_list 0 --dataset_name squad >./log/shots_3b/QuestionAnswering-t0-3b-0-shots.log 2>&1 &

# t5-3b in-context
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name amazon --incontext >./log/shots_3b/SentimentAnalysis-t5-3b-in-context.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name civil_comments --incontext >./log/shots_3b/ToxicDetection-t5-3b-in-context.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/shots-nlu.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name mnli --incontext >./log/shots_3b/NaturalLanguageInference-t5-3b-in-context.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/shots-generation.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name fewnerd --incontext >./log/shots_3b/NameEntityRecognition-t5-3b-in-context.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3,4,5 nohup python src/analysis/shots-qa.py --model_name t5 --scale 3b --repeats 1 --shots_list 0 --dataset_name squad --incontext >./log/shots_3b/QuestionAnswering-t5-3b-in-context.log 2>&1 &


