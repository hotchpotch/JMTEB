// Configuration for Google EmbeddingGemma-300m model
// Prompt prefixes follow the official EmbeddingGemma documentation
// Reference: https://huggingface.co/google/embeddinggemma-300m

// ========== amazon_counterfactual_classification.jsonnet ==========
{
  amazon_counterfactual_classification: {
    class_path: 'ClassificationEvaluator',
    init_args: {
      prefix: 'task: classification | query: ',  // EmbeddingGemma classification format
      train_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'amazon_counterfactual_classification',
        },
      },
      val_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'amazon_counterfactual_classification',
        },
      },
      test_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'amazon_counterfactual_classification',
        },
      },
    },
  },
}

// ========== amazon_review_classification.jsonnet ==========
{
  amazon_review_classification: {
    class_path: 'ClassificationEvaluator',
    init_args: {
      prefix: 'task: classification | query: ',  // EmbeddingGemma classification format
      train_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'amazon_review_classification',
        },
      },
      val_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'amazon_review_classification',
        },
      },
      test_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'amazon_review_classification',
        },
      },
    },
  },
}

// ========== esci.jsonnet ==========
{
  esci: {
    class_path: 'RerankingEvaluator',
    init_args: {
      query_prefix: 'task: search result | query: ',  // EmbeddingGemma retrieval query format
      doc_prefix: 'title: none | text: ',  // EmbeddingGemma document format
      val_query_dataset: {
        class_path: 'HfRerankingQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'esci-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRerankingQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'esci-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRerankingDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'esci-corpus',
        },
      },
    },
  },
}

// ========== jagovfaqs_22k.jsonnet ==========
{
  jagovfaqs_22k: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_prefix: 'task: search result | query: ',  // EmbeddingGemma retrieval query format
      doc_prefix: 'title: none | text: ',  // EmbeddingGemma document format
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'jagovfaqs_22k-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jagovfaqs_22k-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'jagovfaqs_22k-corpus',
        },
      },
    },
  },
}

// ========== jaqket.jsonnet ==========
{
  jaqket: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_prefix: 'task: search result | query: ',  // EmbeddingGemma retrieval query format
      doc_prefix: 'title: none | text: ',  // EmbeddingGemma document format
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'jaqket-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jaqket-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'jaqket-corpus',
        },
      },
    },
  },
}

// ========== jsick.jsonnet ==========
{
  jsick: {
    class_path: 'STSEvaluator',
    init_args: {
      sentence1_prefix: 'task: sentence similarity | query: ',  // EmbeddingGemma similarity format
      sentence2_prefix: 'task: sentence similarity | query: ',  // EmbeddingGemma similarity format
      val_dataset: {
        class_path: 'HfSTSDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'jsick',
        },
      },
      test_dataset: {
        class_path: 'HfSTSDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jsick',
        },
      },
    },
  },
}

// ========== jsts.jsonnet ==========
{
  jsts: {
    class_path: 'STSEvaluator',
    init_args: {
      sentence1_prefix: 'task: sentence similarity | query: ',  // EmbeddingGemma similarity format
      sentence2_prefix: 'task: sentence similarity | query: ',  // EmbeddingGemma similarity format
      val_dataset: {
        class_path: 'HfSTSDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'jsts',
        },
      },
      test_dataset: {
        class_path: 'HfSTSDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jsts',
        },
      },
    },
  },
}

// ========== livedoor_news.jsonnet ==========
{
  livedoor_news: {
    class_path: 'ClusteringEvaluator',
    init_args: {
      prefix: 'task: clustering | query: ',  // EmbeddingGemma clustering format
      val_dataset: {
        class_path: 'HfClusteringDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'livedoor_news',
        },
      },
      test_dataset: {
        class_path: 'HfClusteringDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'livedoor_news',
        },
      },
    },
  },
}

// ========== massive_intent_classification.jsonnet ==========
{
  massive_intent_classification: {
    class_path: 'ClassificationEvaluator',
    init_args: {
      prefix: 'task: classification | query: ',  // EmbeddingGemma classification format
      train_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'massive_intent_classification',
        },
      },
      val_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'massive_intent_classification',
        },
      },
      test_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'massive_intent_classification',
        },
      },
    },
  },
}

// ========== massive_scenario_classification.jsonnet ==========
{
  massive_scenario_classification: {
    class_path: 'ClassificationEvaluator',
    init_args: {
      prefix: 'task: classification | query: ',  // EmbeddingGemma classification format
      train_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'massive_scenario_classification',
        },
      },
      val_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'massive_scenario_classification',
        },
      },
      test_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'massive_scenario_classification',
        },
      },
    },
  },
}

// ========== mewsc16.jsonnet ==========
{
  mewsc16: {
    class_path: 'ClusteringEvaluator',
    init_args: {
      prefix: 'task: clustering | query: ',  // EmbeddingGemma clustering format
      val_dataset: {
        class_path: 'HfClusteringDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'mewsc16_ja',
        },
      },
      test_dataset: {
        class_path: 'HfClusteringDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'mewsc16_ja',
        },
      },
    },
  },
}

// ========== mrtydi.jsonnet ==========
{
  mrtydi: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_prefix: 'task: search result | query: ',  // EmbeddingGemma retrieval query format
      doc_prefix: 'title: none | text: ',  // EmbeddingGemma document format
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'mrtydi-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'mrtydi-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'mrtydi-corpus',
        },
      },
    },
  },
}

// ========== nlp_journal_abs_intro.jsonnet ==========
{
  nlp_journal_abs_intro: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_prefix: 'task: search result | query: ',  // EmbeddingGemma retrieval query format
      doc_prefix: 'title: none | text: ',  // EmbeddingGemma document format
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'nlp_journal_abs_intro-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'nlp_journal_abs_intro-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'nlp_journal_abs_intro-corpus',
        },
      },
    },
  },
}

// ========== nlp_journal_title_abs.jsonnet ==========
{
  nlp_journal_title_abs: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_prefix: 'task: search result | query: ',  // EmbeddingGemma retrieval query format
      doc_prefix: 'title: none | text: ',  // EmbeddingGemma document format
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'nlp_journal_title_abs-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'nlp_journal_title_abs-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'nlp_journal_title_abs-corpus',
        },
      },
    },
  },
}

// ========== nlp_journal_title_intro.jsonnet ==========
{
  nlp_journal_title_intro: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_prefix: 'task: search result | query: ',  // EmbeddingGemma retrieval query format
      doc_prefix: 'title: none | text: ',  // EmbeddingGemma document format
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'nlp_journal_title_intro-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'nlp_journal_title_intro-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'nlp_journal_title_intro-corpus',
        },
      },
    },
  },
}

// ========== paws_x_ja.jsonnet ==========
{
  paws_x_ja: {
    class_path: 'PairClassificationEvaluator',
    init_args: {
      sentence1_prefix: 'task: sentence similarity | query: ',  // EmbeddingGemma similarity format
      sentence2_prefix: 'task: sentence similarity | query: ',  // EmbeddingGemma similarity format
      val_dataset: {
        class_path: 'HfPairClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'paws_x_ja',
        },
      },
      test_dataset: {
        class_path: 'HfPairClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'paws_x_ja',
        },
      },
    },
  },
}