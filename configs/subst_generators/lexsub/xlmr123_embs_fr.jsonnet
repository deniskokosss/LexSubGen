local prob_estimator = import 'configs/prob_estimators/lexsub/xlmr_123.jsonnet';
local post_processing = import 'configs/subst_generators/post_processors/fr_spacy_old_sum.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AndPreprocessor"
        }
    ],
    prob_estimator: {
        class_name: "prob_estimators.combiner.BCombFasttextCombiner",
        prob_estimators: [
            prob_estimator
        ],
        verbose: false,
        k: 4.0,
        s: 1.05,
        beta: 0.0,
        temperature: 0.025,
        lang: "fr"
    },
    post_processing: post_processing,
    top_k: 15,
}