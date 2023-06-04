# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name amazon --method adapter --parameter 1 >./log/delta/adapter/1/SentimentAnalysis-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name amazon --method adapter --parameter 4 >./log/delta/adapter/4/SentimentAnalysis-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name amazon --method adapter --parameter 16 >./log/delta/adapter/16/SentimentAnalysis-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name amazon --method adapter --parameter 64 >./log/delta/adapter/64/SentimentAnalysis-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name amazon --method adapter --parameter 256 >./log/delta/adapter/256/SentimentAnalysis-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name amazon --method adapter --parameter 1024 >./log/delta/adapter/1024/SentimentAnalysis-small.log 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name civil_comments --method adapter --parameter 1 >./log/delta/adapter/1/ToxicDetection-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name civil_comments --method adapter --parameter 4 >./log/delta/adapter/4/ToxicDetection-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name civil_comments --method adapter --parameter 16 >./log/delta/adapter/16/ToxicDetection-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name civil_comments --method adapter --parameter 64 >./log/delta/adapter/64/ToxicDetection-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name civil_comments --method adapter --parameter 256 >./log/delta/adapter/256/ToxicDetection-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name civil_comments --method adapter --parameter 1024 >./log/delta/adapter/1024/ToxicDetection-small.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name mnli --method adapter --parameter 1 >./log/delta/adapter/1/NaturalLanguageInference-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name mnli --method adapter --parameter 4 >./log/delta/adapter/4/NaturalLanguageInference-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name mnli --method adapter --parameter 16 >./log/delta/adapter/16/NaturalLanguageInference-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name mnli --method adapter --parameter 64 >./log/delta/adapter/64/NaturalLanguageInference-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name mnli --method adapter --parameter 256 >./log/delta/adapter/256/NaturalLanguageInference-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name mnli --method adapter --parameter 1024 >./log/delta/adapter/1024/NaturalLanguageInference-small.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name hellaswag --method adapter --parameter 1 >./log/delta/adapter/1/CommonSense-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name hellaswag --method adapter --parameter 4 >./log/delta/adapter/4/CommonSense-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name hellaswag --method adapter --parameter 16 >./log/delta/adapter/16/CommonSense-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name hellaswag --method adapter --parameter 64 >./log/delta/adapter/64/CommonSense-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name hellaswag --method adapter --parameter 256 >./log/delta/adapter/256/CommonSense-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale small --dataset_name hellaswag --method adapter --parameter 1024 >./log/delta/adapter/1024/CommonSense-small.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/delta-ner.py --model_name deberta --scale small --dataset_name fewnerd --method adapter --parameter 1 >./log/delta/adapter/1/NameEntityRecognition-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/delta-ner.py --model_name deberta --scale small --dataset_name fewnerd --method adapter --parameter 4 >./log/delta/adapter/4/NameEntityRecognition-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-ner.py --model_name deberta --scale small --dataset_name fewnerd --method adapter --parameter 16 >./log/delta/adapter/16/NameEntityRecognition-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/delta-ner.py --model_name deberta --scale small --dataset_name fewnerd --method adapter --parameter 64 >./log/delta/adapter/64/NameEntityRecognition-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/delta-ner.py --model_name deberta --scale small --dataset_name fewnerd --method adapter --parameter 256 >./log/delta/adapter/256/NameEntityRecognition-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-ner.py --model_name deberta --scale small --dataset_name fewnerd --method adapter --parameter 1024 >./log/delta/adapter/1024/NameEntityRecognition-small.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/delta-qa.py --model_name t5 --scale small --dataset_name squad --method adapter --parameter 1 >./log/delta/adapter/1/QuestionAnswering-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/delta-qa.py --model_name t5 --scale small --dataset_name squad --method adapter --parameter 4 >./log/delta/adapter/4/QuestionAnswering-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/delta-qa.py --model_name t5 --scale small --dataset_name squad --method adapter --parameter 16 >./log/delta/adapter/16/QuestionAnswering-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-qa.py --model_name t5 --scale small --dataset_name squad --method adapter --parameter 64 >./log/delta/adapter/64/QuestionAnswering-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-qa.py --model_name t5 --scale small --dataset_name squad --method adapter --parameter 256 >./log/delta/adapter/256/QuestionAnswering-small.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-qa.py --model_name t5 --scale small --dataset_name squad --method adapter --parameter 1024 >./log/delta/adapter/1024/QuestionAnswering-small.log 2>&1 &










# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name amazon --method adapter --parameter 1 >./log/delta/adapter/1/SentimentAnalysis-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name amazon --method adapter --parameter 4 >./log/delta/adapter/4/SentimentAnalysis-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name amazon --method adapter --parameter 16 >./log/delta/adapter/16/SentimentAnalysis-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name amazon --method adapter --parameter 64 >./log/delta/adapter/64/SentimentAnalysis-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name amazon --method adapter --parameter 256 >./log/delta/adapter/256/SentimentAnalysis-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name amazon --method adapter --parameter 1024 >./log/delta/adapter/1024/SentimentAnalysis-base.log 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name civil_comments --method adapter --parameter 1 >./log/delta/adapter/1/ToxicDetection-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name civil_comments --method adapter --parameter 4 >./log/delta/adapter/4/ToxicDetection-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name civil_comments --method adapter --parameter 16 >./log/delta/adapter/16/ToxicDetection-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name civil_comments --method adapter --parameter 64 >./log/delta/adapter/64/ToxicDetection-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name civil_comments --method adapter --parameter 256 >./log/delta/adapter/256/ToxicDetection-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name civil_comments --method adapter --parameter 1024 >./log/delta/adapter/1024/ToxicDetection-base.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name mnli --method adapter --parameter 1 >./log/delta/adapter/1/NaturalLanguageInference-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name mnli --method adapter --parameter 4 >./log/delta/adapter/4/NaturalLanguageInference-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name mnli --method adapter --parameter 16 >./log/delta/adapter/16/NaturalLanguageInference-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name mnli --method adapter --parameter 64 >./log/delta/adapter/64/NaturalLanguageInference-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name mnli --method adapter --parameter 256 >./log/delta/adapter/256/NaturalLanguageInference-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name mnli --method adapter --parameter 1024 >./log/delta/adapter/1024/NaturalLanguageInference-base.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name hellaswag --method adapter --parameter 1 >./log/delta/adapter/1/CommonSense-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name hellaswag --method adapter --parameter 4 >./log/delta/adapter/4/CommonSense-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name hellaswag --method adapter --parameter 16 >./log/delta/adapter/16/CommonSense-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name hellaswag --method adapter --parameter 64 >./log/delta/adapter/64/CommonSense-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name hellaswag --method adapter --parameter 256 >./log/delta/adapter/256/CommonSense-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale base --dataset_name hellaswag --method adapter --parameter 1024 >./log/delta/adapter/1024/CommonSense-base.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/delta-ner.py --model_name deberta --scale base --dataset_name fewnerd --method adapter --parameter 1 >./log/delta/adapter/1/NameEntityRecognition-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/delta-ner.py --model_name deberta --scale base --dataset_name fewnerd --method adapter --parameter 4 >./log/delta/adapter/4/NameEntityRecognition-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-ner.py --model_name deberta --scale base --dataset_name fewnerd --method adapter --parameter 16 >./log/delta/adapter/16/NameEntityRecognition-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/delta-ner.py --model_name deberta --scale base --dataset_name fewnerd --method adapter --parameter 64 >./log/delta/adapter/64/NameEntityRecognition-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/delta-ner.py --model_name deberta --scale base --dataset_name fewnerd --method adapter --parameter 256 >./log/delta/adapter/256/NameEntityRecognition-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-ner.py --model_name deberta --scale base --dataset_name fewnerd --method adapter --parameter 1024 >./log/delta/adapter/1024/NameEntityRecognition-base.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/delta-qa.py --model_name t5 --scale base --dataset_name squad --method adapter --parameter 1 >./log/delta/adapter/1/QuestionAnswering-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/delta-qa.py --model_name t5 --scale base --dataset_name squad --method adapter --parameter 4 >./log/delta/adapter/4/QuestionAnswering-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-qa.py --model_name t5 --scale base --dataset_name squad --method adapter --parameter 16 >./log/delta/adapter/16/QuestionAnswering-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/delta-qa.py --model_name t5 --scale base --dataset_name squad --method adapter --parameter 64 >./log/delta/adapter/64/QuestionAnswering-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/delta-qa.py --model_name t5 --scale base --dataset_name squad --method adapter --parameter 256 >./log/delta/adapter/256/QuestionAnswering-base.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/delta-qa.py --model_name t5 --scale base --dataset_name squad --method adapter --parameter 1024 >./log/delta/adapter/1024/QuestionAnswering-base.log 2>&1 &











# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name amazon --method adapter --parameter 1 >./log/delta/adapter/1/SentimentAnalysis-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name amazon --method adapter --parameter 4 >./log/delta/adapter/4/SentimentAnalysis-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name amazon --method adapter --parameter 16 >./log/delta/adapter/16/SentimentAnalysis-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name amazon --method adapter --parameter 64 >./log/delta/adapter/64/SentimentAnalysis-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name amazon --method adapter --parameter 256 >./log/delta/adapter/256/SentimentAnalysis-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name amazon --method adapter --parameter 1024 >./log/delta/adapter/1024/SentimentAnalysis-large.log 2>&1 &



# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method adapter --parameter 1 >./log/delta/adapter/1/ToxicDetection-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method adapter --parameter 4 >./log/delta/adapter/4/ToxicDetection-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method adapter --parameter 16 >./log/delta/adapter/16/ToxicDetection-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method adapter --parameter 64 >./log/delta/adapter/64/ToxicDetection-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method adapter --parameter 256 >./log/delta/adapter/256/ToxicDetection-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method adapter --parameter 1024 >./log/delta/adapter/1024/ToxicDetection-large.log 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name mnli --method adapter --parameter 1 >./log/delta/adapter/1/NaturalLanguageInference-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name mnli --method adapter --parameter 4 >./log/delta/adapter/4/NaturalLanguageInference-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name mnli --method adapter --parameter 16 >./log/delta/adapter/16/NaturalLanguageInference-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name mnli --method adapter --parameter 64 >./log/delta/adapter/64/NaturalLanguageInference-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name mnli --method adapter --parameter 256 >./log/delta/adapter/256/NaturalLanguageInference-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name mnli --method adapter --parameter 1024 >./log/delta/adapter/1024/NaturalLanguageInference-large.log 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name hellaswag --method adapter --parameter 1 >./log/delta/adapter/1/CommonSense-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name hellaswag --method adapter --parameter 4 >./log/delta/adapter/4/CommonSense-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name hellaswag --method adapter --parameter 16 >./log/delta/adapter/16/CommonSense-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name hellaswag --method adapter --parameter 64 >./log/delta/adapter/64/CommonSense-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name hellaswag --method adapter --parameter 256 >./log/delta/adapter/256/CommonSense-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-nlu.py --model_name t5 --scale large --dataset_name hellaswag --method adapter --parameter 1024 >./log/delta/adapter/1024/CommonSense-large.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python src/analysis/delta-ner.py --model_name deberta --scale large --dataset_name fewnerd --method adapter --parameter 1 >./log/delta/adapter/1/NameEntityRecognition-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/delta-ner.py --model_name deberta --scale large --dataset_name fewnerd --method adapter --parameter 4 >./log/delta/adapter/4/NameEntityRecognition-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/analysis/delta-ner.py --model_name deberta --scale large --dataset_name fewnerd --method adapter --parameter 16 >./log/delta/adapter/16/NameEntityRecognition-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/analysis/delta-ner.py --model_name deberta --scale large --dataset_name fewnerd --method adapter --parameter 64 >./log/delta/adapter/64/NameEntityRecognition-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/analysis/delta-ner.py --model_name deberta --scale large --dataset_name fewnerd --method adapter --parameter 256 >./log/delta/adapter/256/NameEntityRecognition-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-ner.py --model_name deberta --scale large --dataset_name fewnerd --method adapter --parameter 1024 >./log/delta/adapter/1024/NameEntityRecognition-large.log 2>&1 &


# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-qa.py --model_name t5 --scale large --dataset_name squad --method adapter --parameter 1 >./log/delta/adapter/1/QuestionAnswering-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/analysis/delta-qa.py --model_name t5 --scale large --dataset_name squad --method adapter --parameter 4 >./log/delta/adapter/4/QuestionAnswering-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/analysis/delta-qa.py --model_name t5 --scale large --dataset_name squad --method adapter --parameter 16 >./log/delta/adapter/16/QuestionAnswering-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/analysis/delta-qa.py --model_name t5 --scale large --dataset_name squad --method adapter --parameter 64 >./log/delta/adapter/64/QuestionAnswering-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/delta-qa.py --model_name t5 --scale large --dataset_name squad --method adapter --parameter 256 >./log/delta/adapter/256/QuestionAnswering-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/analysis/delta-qa.py --model_name t5 --scale large --dataset_name squad --method adapter --parameter 1024 >./log/delta/adapter/1024/QuestionAnswering-large.log 2>&1 &


