{
  jacwir: {
    class_path: 'RetrievalEvaluator',
    init_args: {
#      query_prefix: 'query: ',
#      doc_prefix: 'passage: ',
#      query_prefix: 'クエリ: ',
#      doc_prefix: '文章: ',
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'hotchpotch/JaCWIR-JMTEB',
          split: 'test_200_samples',
          name: 'JaCWIR-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'hotchpotch/JaCWIR-JMTEB',
          split: 'test',
          name: 'JaCWIR-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'hotchpotch/JaCWIR-JMTEB',
          split: 'corpus',
          name: 'JaCWIR-corpus',
        },
      },
      "doc_chunk_size":10000
    },
  },
}
