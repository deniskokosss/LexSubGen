local post_processing = import '../subst_generators/post_processors/lemmatizer_name.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AndPreprocessor"
        }
    ],
    prob_estimator: {
        class_name: "prob_estimators.combiner.AverageCombiner",
        merge_vocab_type: "union",
        prob_estimators: [
            {
                class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                topk: 10,
                model_name: "/mnt/d/science/summer-wsi/xlmr_large_cwm_multi/model.pt",
                num_masks: 3,
                dynamic_pattern: {
                    class_name: "Hyperparam",
                    values: ["left pattern"],
                    name: "dpl"
                },
                decoding_type: {
                    class_name: "Hyperparam",
                    values: ["cwm"],
                    name: "multitoken"
                }
            },
            {
                class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                topk: 10,
                model_name: "/mnt/d/science/summer-wsi/xlmr_large_cwm_multi/model.pt",
                num_masks: 3,
                dynamic_pattern: {
                    class_name: "Hyperparam",
                    values: ["right pattern"],
                    name: "dpr"
                },
                decoding_type: {
                    class_name: "Hyperparam",
                    values: ["cwm"],
                    name: "multitoken"
                }
            },
        ]
    },
    post_processing: post_processing,
    top_k: 10,
}