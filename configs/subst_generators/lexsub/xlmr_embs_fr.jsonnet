local prob_estimator = import 'configs/prob_estimators/lexsub/xlmr_embs_fr.jsonnet';
local post_processing = import 'configs/subst_generators/post_processors/fr_spacy_old_sum.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AndPreprocessor"
        }
    ],
    prob_estimator: prob_estimator,
    post_processing: [
        {class_name: "post_processors.roberta_postproc.RobertaPostProcessor", strategy: "drop_subwords"}
    ] + post_processing,
    top_k: 15,
}