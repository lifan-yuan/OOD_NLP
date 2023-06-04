# CUDA_VISIBLE_DEVICES=5 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name amazon --method vanilla >./log/method_large/vanilla/SentimentAnalysis.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name amazon --method freelb >./log/method_large/freelb/SentimentAnalysis.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name amazon --method focal_loss >./log/method_large/focal_loss/SentimentAnalysis.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name amazon --method label_smoothing >./log/method_large/label_smoothing/SentimentAnalysis.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name amazon --method shallow_ensemble >./log/method_large/shallow_ensemble/SentimentAnalysis.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name amazon --method eda >./log/method_large/eda/SentimentAnalysis.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method vanilla >./log/method_large/vanilla/ToxicDetection.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method freelb >./log/method_large/freelb/ToxicDetection.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method focal_loss >./log/method_large/focal_loss/ToxicDetection.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method label_smoothing >./log/method_large/label_smoothing/ToxicDetection.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method shallow_ensemble >./log/method_large/shallow_ensemble/ToxicDetection.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name civil_comments --method eda >./log/method_large/eda/ToxicDetection.log 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name mnli --method vanilla >./log/method_large/vanilla/NaturalLanguageInference.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name mnli --method freelb >./log/method_large/freelb/NaturalLanguageInference.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name mnli --method focal_loss >./log/method_large/focal_loss/NaturalLanguageInference.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name mnli --method label_smoothing >./log/method_large/label_smoothing/NaturalLanguageInference.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name mnli --method shallow_ensemble >./log/method_large/shallow_ensemble/NaturalLanguageInference.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/evaluations/method-nlu.py --model_name t5 --scale large --dataset_name mnli --method eda >./log/method_large/eda/NaturalLanguageInference.log 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python src/evaluations/method-ner.py --model_name deberta --scale large --dataset_name fewnerd --method vanilla >./log/method_large/vanilla/NameEntityRecognition.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/evaluations/method-ner.py --model_name deberta --scale large --dataset_name fewnerd --method freelb >./log/method_large/freelb/NameEntityRecognition.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python src/evaluations/method-ner.py --model_name deberta --scale large --dataset_name fewnerd --method focal_loss >./log/method_large/focal_loss/NameEntityRecognition.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python src/evaluations/method-ner.py --model_name deberta --scale large --dataset_name fewnerd --method label_smoothing >./log/method_large/label_smoothing/NameEntityRecognition.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/evaluations/method-ner.py --model_name deberta --scale large --dataset_name fewnerd --method shallow_ensemble >./log/method_large/shallow_ensemble/NameEntityRecognition.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4,7 nohup python src/evaluations/method-qa.py --model_name t5 --scale large --dataset_name squad --method vanilla >./log/method_large/vanilla/QuestionAnswering.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3,4,5 nohup python src/evaluations/method-qa.py --model_name t5 --scale large --dataset_name squad --method freelb >./log/method_large/freelb/QuestionAnswering.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2 nohup python src/evaluations/method-qa.py --model_name t5 --scale large --dataset_name squad --method focal_loss >./log/method_large/focal_loss/QuestionAnswering.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3,4,5 nohup python src/evaluations/method-qa.py --model_name t5 --scale large --dataset_name squad --method label_smoothing >./log/method_large/label_smoothing/QuestionAnswering.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python src/evaluations/method-qa.py --model_name t5 --scale large --dataset_name squad --method shallow_ensemble >./log/method_large/shallow_ensemble/QuestionAnswering.log 2>&1 &