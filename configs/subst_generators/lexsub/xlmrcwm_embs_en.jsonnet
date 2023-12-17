local prob_estimator = import '../../prob_estimators/lexsub/xlmr_cwm.jsonnet';
local post_processing = import '../post_processors/spacy_old_sum.jsonnet';

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
        s: 1.0,
        beta: 1.0,
        temperature: 0.07,
        lang: "en"
    },
    post_processing: post_processing,
    top_k: 15,
}