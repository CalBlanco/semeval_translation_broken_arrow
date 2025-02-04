from hf_model import FineTunedMBart
from output_module import infer_test, load_test_dataframe, load_ent_test_info, format_frame_for_sub
from config import HF_MBART_CONFIG
import pandas as pd
import os

model = FineTunedMBart()

langs = ['ar', 'de', 'es', 'fr', 'ja', 'it', 'ko', 'th', 'tr', 'zh']

def merge_entity_info(ent_table): #Example function to set up entity information that will be passed in as data
    source = ent_table['source']
    target = ent_table['target']

    source = str(source).split("*|*")
    target = str(target).split("*|*")

    merged = [f'{x}={y}' for x,y in zip(source, target)]

    return ','.join(merged)



for lang in langs:
    test_data = load_test_dataframe(lang) # contiains 'source'
    ent_data = load_ent_test_info(lang) #source, target
    
    ent_data['parsed'] = ent_data['target'].apply(lambda x: ','.join(str(x).split("*|*"))) #run our function to merge entites

    test_data['mod'] = test_data['source'] + ent_data['parsed'] #create column in dataframe for our modified input


    wrapper = lambda x: model.batch_translate(x, lang, HF_MBART_CONFIG['batch_size']) #wrapper for model translation
    out_frame = infer_test(wrapper, frame=test_data, src_column='mod') #output frame from predictions

    formated = format_frame_for_sub(lang, out_frame)

    formated.to_json(f'./outputs/ex1_{lang}.json', orient='records', lines=True)

