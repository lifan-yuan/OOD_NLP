import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from openprompt import PromptForClassification
from openprompt.data_utils.utils import InputExample
from typing import Optional, Sequence
from transformers import T5ForConditionalGeneration, DebertaV2ForTokenClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble  import RandomForestClassifier
from nltk import sent_tokenize

from tqdm import tqdm


def mixup(word_embeddings, inputs, alpha=0.5):

    x = word_embeddings(inputs["input_ids"].cuda()).detach().cpu()
    y = inputs["label"].detach().cpu()

    batch_size = x.shape[0]

    weight = torch.from_numpy(np.random.beta(alpha, alpha, batch_size)).float()
    x_weight = weight.reshape(batch_size, 1, 1)
    y_weight = weight.reshape(batch_size,)

    index = torch.from_numpy(np.random.permutation(batch_size))

    x1, x2 = x, x[index]
    inputs_embeds = x1 * x_weight + x2 * (1 - x_weight)

    y1, y2 = y, y[index]
    label = y1 * y_weight + y2 * (1 - y_weight)
    
    inputs["input_ids"] = None
    inputs["inputs_embeds"] = inputs_embeds
    inputs["label"] = label.long()

    return inputs




class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss



class FreeLB(object):
    '''
    https://arxiv.org/pdf/1909.11764.pdf
    freelb = FreeLB()
    K = 3
    for batch_input, batch_label in processor:
        loss = freelb.attack(model,inputs,.....)
    '''
    def __init__(self, adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2, adv_max_norm=0., adv_norm_type='l2', base_model='t5'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model
        self.loss_func = nn.CrossEntropyLoss()

    def attack(self, model, inputs, gradient_accumulation_steps=1):
        input_ids = inputs['input_ids'].cuda()
        
        if isinstance(model, torch.nn.DataParallel):
            word_embeddings = model.module.get_input_embeddings()
        elif isinstance(model, PromptForClassification):
            word_embeddings = model.plm.get_input_embeddings()
        else:
            word_embeddings = model.get_input_embeddings()

        embeds_init = word_embeddings(input_ids)

        if self.adv_init_mag > 0:
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)

        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init
            inputs['input_ids'] = None
            ##########################################################################
            if isinstance(model, PromptForClassification):
                logits = model(inputs)
                labels = inputs['label']
                loss = self.loss_func(logits, labels)
            else:
                inputs_embeds = inputs["inputs_embeds"].cuda()
                attention_masks = inputs["attention_mask"].cuda()
                labels = inputs["label"].cuda()
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_masks, labels=labels)
                loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
            ##########################################################################
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))

            embeds_init = word_embeddings(input_ids)
            
        return loss



class ShallowEnsembledModel(nn.Module):

    def __init__(self, model, loss_fn):
        super(ShallowEnsembledModel, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, inputs, labels=None, bias=None):

        if isinstance(self.model, PromptForClassification):
            # PromptModel
            batch = self.model.prompt_model.template.process_batch(inputs)
            input_batch = {key: batch[key] for key in batch if key in self.model.prompt_model.forward_keys}

            outputs = self.model.prompt_model.plm(**input_batch, output_hidden_states=True)
            # hidden state
            last_hidden_state = outputs.decoder_hidden_states[-1].mean(dim=1) # average pooling
            outputs = self.model.prompt_model.template.post_processing_outputs(outputs)
            # PromptModelForClassification
            outputs = self.model.verbalizer.gather_outputs(outputs)
            if isinstance(outputs, tuple):
                outputs_at_mask = [self.model.extract_at_mask(output, batch) for output in outputs]
            else:
                outputs_at_mask = self.model.extract_at_mask(outputs, batch)
            # logits
            logits = self.model.verbalizer.process_outputs(outputs_at_mask, batch=batch)
            # print(logits)
        elif isinstance(self.model, DebertaV2ForTokenClassification):
            outputs = self.model(input_ids=inputs["input_ids"].cuda(), attention_mask=inputs["attention_mask"].cuda(), output_hidden_states=True)
            logits, last_hidden_state = outputs.logits, outputs.hidden_states[-1]
            print(inputs["input_ids"].shape)
            print(logits.shape) # [batch_size, seq_len, num_labels]
            # exit()
        elif isinstance(self.model, T5ForConditionalGeneration):
            outputs = self.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
            logits, last_hidden_state = outputs.logits, outputs.decoder_hidden_states[-1]
        else: 
            raise ValueError("Model type {} not specified.".format(type(self.model)))

        loss = self.loss_fn(last_hidden_state, logits, bias, labels)
        return logits, loss


class ShallowModelForNLU(nn.Module):
    def __init__(self):
        super(ShallowModelForNLU, self).__init__()
        self.vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
        self.classifier = RandomForestClassifier(n_estimators = 100)

    def preprocess(self, dataset):
        # print("dataset[0].text_b:", dataset[0].text_b)
        if len(dataset[0].text_b) == 0:
            data_features = [data.text_a for data in dataset]
        else:
            data_features = [" ".join([data.text_a, data.text_b]) for data in dataset]
        labels = [data.label for data in dataset]

        return data_features, labels

    def capture_bias(self, dataset):
        # preprocess
        data_features, labels = self.preprocess(dataset)
        # vectorize
        data_features = self.vectorizer.fit_transform(data_features)
        # train
        self.classifier.fit(data_features, labels)
        # predict
        probs = self.classifier.predict_proba(data_features)
        preds = self.classifier.predict(data_features)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(labels, preds)
        print("shallow model accuracy:", accuracy)
        return probs


################### classification ##############
class LearnedMixin(nn.Module):

  def __init__(self, penalty, dim):
    super().__init__()
    self.penalty = penalty
    self.bias_lin = torch.nn.Linear(dim, 1)

  def forward(self, hidden, logits, bias, labels):
    logits = logits.float()  # In case we were in fp16 mode
    logits = F.log_softmax(logits, 1)

    factor = self.bias_lin.forward(hidden)
    factor = factor.float()
    factor = F.softplus(factor)

    bias = bias * factor

    bias_lp = F.log_softmax(bias, 1)
    entropy = -(torch.exp(bias_lp) * bias_lp).sum(1).mean(0)
    loss = F.cross_entropy(logits + bias, labels) + self.penalty*entropy
    return loss




import random
from random import shuffle
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1):
	
	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word != '']
	num_words = len(words)
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4)+1

	#sr
	if (alpha_sr > 0):
		n_sr = max(1, int(alpha_sr*num_words))
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr)
			augmented_sentences.append(' '.join(a_words))

	#ri
	if (alpha_ri > 0):
		n_ri = max(1, int(alpha_ri*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri)
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd)
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences

def eda_augmentation(dataset):
    augmented_dataset = []
    flag = True if len(dataset[0].text_b) == 0 else False
    for data in tqdm(dataset, desc="Augmenting dataset"):
        try:
            augmented_sentences = eda(sentence=data.text_a if flag else data.text_b)
        except:
            print("Error in eda_augmentation")
            continue
        for i, augmented_sentence in enumerate(augmented_sentences):
            guid = data.guid + "-" + str(i)
            if len(dataset[0].text_b) == 0:
                text_a = augmented_sentence
                text_b = ""
            else: 
                text_a = dataset[0].text_a
                text_b = augmented_sentence
            label = data.label
            augmented_dataset.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return augmented_dataset





