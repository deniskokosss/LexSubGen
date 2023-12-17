
local post_processing = import '../subst_generators/post_processors/spacy_old_sum.jsonnet';


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
            {
                class_name: "prob_estimators.combiner.MaxCombiner",
                merge_vocab_type: "union",
                prob_estimators: [
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: {
                            class_name: "Hyperparam",
                            values: [500],
                            name: "mask1_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 1
                    },
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: {
                            class_name: "Hyperparam",
                            values: [500],
                            name: "mask2_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 2
                    },
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: {
                            class_name: "Hyperparam",
                            values: [500],
                            name: "mask3_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 3
                    },
                ],
            }
        ],
        verbose: false,
        k: 4.0,
        s: {
            class_name: "Hyperparam",
            name: "s",
            values: [1.0],
        },
        temperature: {
            class_name: "Hyperparam",
            values: [0.05],
            name: "embs_temp"
        },
        beta: {
            class_name: "Hyperparam",
            name: "beta",
            values: [0.0],
        },
        lang: "en"
    },
    post_processing: post_processing,
    top_k: 10,
}