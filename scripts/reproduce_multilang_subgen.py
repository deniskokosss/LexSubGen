import os
from pathlib import Path
import re
import logging
from deep_translator import GoogleTranslator
logging.basicConfig(level = logging.INFO)

os.environ['HOME'] = "/mnt/d/science/lexsubgen"

home = Path('..')

left_patterns = ["<T> or <M>", "<T> and <M>", "<T> or even <M>", "<T> and also <M>",
                 "<T> (or even <M>)", "<T> (and also <M>)"]

right_patterns = ["<M> or <T>", "<M> and <T>", "<M> or even <T>", "<M> and also <T>",
                  "<M> (or even <T>)", "<M> (and also <T>)"]

datasets = [
    "germeval2015",
    "lexsubfr_semdis2014",
    "semeval_test",
    # "coinco"
]
lemmatizer_names = [
    "de_spacy_old_sum", "fr_spacy_old_sum",
    "spacy_old_sum",
    # "spacy_old_sum"
]
langs = [
    'de', 'fr',
         'en',
    # 'en'
]
pattern_combs = ['cwm', '123']

dataset = "semeval_test"
logging.info(f"Running {(dataset, )} roberta+embs")
os.system(f"""
    cd {os.environ['HOME']} &&
    python lexsubgen/evaluations/lexsub.py
    solve
    --mode
    hyperparam_search
    --substgen-config-path
    configs/hyperparam_search/roberta_ftembs.jsonnet
    --dataset-config-path
    configs/dataset_readers/lexsub/{dataset}.jsonnet
    --run-dir='debug/lexsub-all-models/{dataset}_embs'
    --force
    --experiment-name='{dataset}_final'
    --batch-size=5
""".replace('\n', ' '))

for dataset, lemmatizer, lang in zip(datasets, lemmatizer_names, langs):
    for pc in pattern_combs:
        logging.info(f"Running {(dataset, lemmatizer,)} +embs")
        res = []
        with open(home / 'configs' / 'hyperparam_search' / f'xlmr{pc}_embs.jsonnet', 'r') as f:
            for s in f.readlines():
                s = re.sub('lang_template', lang, s)
                s = re.sub('lemmatizer_name', lemmatizer, s)
                res.append(s)
        with open(home / 'configs' / 'hyperparam_search' / f'xlmr{pc}_embs_temp.jsonnet', 'w') as fw:
            fw.writelines(res)
        logging.info("\n".join(res))
        os.system(f"""
            cd {os.environ['HOME']} &&
            python lexsubgen/evaluations/lexsub.py
            solve
            --mode
            hyperparam_search
            --substgen-config-path
            configs/hyperparam_search/xlmr{pc}_embs_temp.jsonnet
            --dataset-config-path
            configs/dataset_readers/lexsub/{dataset}.jsonnet
            --run-dir='debug/lexsub-all-models/{dataset}_embs'
            --force
            --experiment-name='{dataset}_final'
            --batch-size=5
        """.replace('\n', ' '))

# for dataset, lemmatizer, lang in zip(datasets, lemmatizer_names, langs):
#     trans = GoogleTranslator(source='en', target=lang)
#     for pc in pattern_combs:
#         for ldp, rdp in zip(left_patterns, right_patterns):
#
#             logging.info(f"Running {(dataset, lemmatizer, ldp, rdp, lang)}")
#             logging.info(trans.translate(ldp))
#             res = []
#             with open(home / 'configs' / 'hyperparam_search' / f'xlmr{pc}_dp.jsonnet', 'r') as f:
#                 for s in f.readlines():
#                     s = re.sub('left pattern', trans.translate(ldp), s)
#                     s = re.sub('right pattern', trans.translate(rdp), s)
#                     s = re.sub('lemmatizer_name', lemmatizer, s)
#                     res.append(s)
#             with open(home / 'configs' / 'hyperparam_search' / f'xlmr{pc}_dp_temp.jsonnet', 'w') as fw:
#                 fw.writelines(res)
#             logging.info("\n".join(res))
#             os.system(f"""
#                 cd {os.environ['HOME']} &&
#                 python lexsubgen/evaluations/lexsub.py
#                 solve
#                 --mode
#                 hyperparam_search
#                 --substgen-config-path
#                 configs/hyperparam_search/xlmr{pc}_dp_temp.jsonnet
#                 --dataset-config-path
#                 configs/dataset_readers/lexsub/{dataset}.jsonnet
#                 --run-dir='debug/lexsub-all-models/{dataset}_{ldp}'
#                 --force
#                 --experiment-name='{dataset}_final'
#                 --batch-size=10
#             """.replace('\n', ' '))

