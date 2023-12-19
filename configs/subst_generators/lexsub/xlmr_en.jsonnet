local prob_estimator = import '../../prob_estimators/lexsub/xlmr1.jsonnet';
local post_processing = import '../post_processors/spacy_old_sum.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AndPreprocessor"
        }
    ],
    prob_estimator: prob_estimator,
    post_processing: post_processing,
    top_k: 10,
}