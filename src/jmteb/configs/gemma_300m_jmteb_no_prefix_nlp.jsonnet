// Configuration for Google EmbeddingGemma-300m model (without prefixes)
// Based on gemma_300m_jmteb.jsonnet but with all prefixes set to empty string
// Reference: https://huggingface.co/google/embeddinggemma-300m


// ========== nlp_journal_abs_intro.jsonnet ==========
{
  nlp_journal_abs_intro: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_prefix: '',  // No prefix
      doc_prefix: '',  // No prefix
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
      query_prefix: '',  // No prefix
      doc_prefix: '',  // No prefix
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
      query_prefix: '',  // No prefix
      doc_prefix: '',  // No prefix
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
