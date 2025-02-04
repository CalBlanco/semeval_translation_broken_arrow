import flair
from flair.data import Sentence
from flair.nn import Classifier
import torch
from tqdm import tqdm

import pdb
import os

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from knowledge import KnowledgeBase
from config import semeval_langs, HF_MBART_CONFIG


class FineTunedMBart():
    def __init__(self, path:str='./mbart-finetuned-01', flair_model='ner-large'):

        self.model_path = path
        self.lang_codes =  {
        'ar': 'ar_AR',
        'zh': 'zh_CN',
        'es': 'es_XX',
        'de': 'de_DE',
        'fr': 'fr_XX',
        'it': 'it_IT',
        'ja': 'ja_XX',
        'ko': 'ko_KR',
        'tr': 'tr_TR',
        'th': 'th_TH'
        }
        self.device = 'cuda'
        #self.tagger = Classifier.load(flair_model).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.tagger = lambda x: x
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_path, device_map='cuda')
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_path, device_map='cuda')

        self.kb = KnowledgeBase()

        

    def get_entity_sent(self, source, lang):
        sentences = [Sentence(s) for s in source]
        self.tagger.predict(sentences, mini_batch_size=32)
        entity_translation = []
        for sentence in sentences:
            batched_ents = []
            for entity in sentence.get_spans('ner'):
                found = self.kb.get(entity.text, lang)
                if found:
                    batched_ents.append(str(found))

        return ','.join(entity_translation)


    def translate(self, text, target_lang):
        #get ents
        """ ents = self.get_entity_sent(text, target_lang)
        
        # Tokenize input text
        text = text + " | " + ents
         """
        tokenized_input = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True)

        lang_ = self.lang_codes[target_lang]
        # Generate translation
        generated_tokens = self.model.generate(
            **tokenized_input,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[lang_]  # Use correct target language
        )

        # Decode output
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    def batch_translate(self, texts, target_lang, batch_size=HF_MBART_CONFIG['batch_size']):
        lang_ = self.lang_codes[target_lang]

        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Tokenize batch
            tokenized_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                pad_to_multiple_of=8  # Efficient padding
            ).to(self.device)

            # Generate translations
            generated_tokens = self.model.generate(
                **tokenized_inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[lang_]
            )

            # Decode translations
            batch_translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]
            translations.extend(batch_translations)

        return translations