from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import os
import logging
import nltk
import numpy as np
import tensorflow as tf
from tensorflow import keras

# If the dataset is gated/private, make sure you have run huggingface-cli login
train_df, test_df = load_dataset("yale-nlp/QTSumm", token = "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", split = ["train", "test"])

source_text = [
    f"query:  {query} header: {' '.join(map(str, entry.get('header', [])))} rows: {' '.join(map(str, entry.get('rows', [])))} title: {' '.join(map(str, entry.get('title', [])))}"
    for query, entry in zip(train_df['query'], train_df['table'])
]

train_df = train_df.add_column('source_text', source_text)

source_text = [
    f"query:  {query} header: {' '.join(map(str, entry.get('header', [])))} rows: {' '.join(map(str, entry.get('rows', [])))} title: {' '.join(map(str, entry.get('title', [])))}"
    for query, entry in zip(test_df['query'], test_df['table'])
]

test_df = test_df.add_column('source_text', source_text)

len(train_df['summary'])

import keras_nlp

from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
model_path = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import AutoTokenizer

model_path = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenization_train(example):
    inputs = tokenizer(example['source_text'], truncation=True, padding=True, return_tensors='tf')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['summary'], truncation=True, padding=True, return_tensors='tf')

    return {
        'input_ids': inputs['input_ids'].numpy(),
        'labels': labels['input_ids'].numpy(),
        'attention_mask': inputs['attention_mask'].numpy()
    }

def tokenization_test(example):
    inputs = tokenizer(example['source_text'], truncation=True, padding=True, return_tensors='tf')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['summary'], truncation=True, padding=True, return_tensors='tf')

    return {
        'input_ids': inputs['input_ids'].numpy(),
        'labels': labels['input_ids'].numpy(),
        'attention_mask': inputs['attention_mask'].numpy()
    }

tokenized_dataset_train = train_df.map(tokenization_train, batched=True)
tokenized_dataset_test = test_df.map(tokenization_test, batched=True)

tokenized_dataset_train

processed_data_train = tokenized_dataset_train.remove_columns(['table','summary', 'row_ids', 'example_id', 'query', 'source_text'])
processed_data_test = tokenized_dataset_test.remove_columns(['table','summary', 'row_ids', 'example_id', 'query', 'source_text'])

processed_data_train

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model= model, return_tensors="tf")

train_dataset = model.prepare_tf_dataset(processed_data_train, batch_size=32, tokenizer= tokenizer, collate_fn=data_collator, shuffle=True, drop_remainder=True)

test_dataset = model.prepare_tf_dataset(processed_data_test, batch_size=32, tokenizer= tokenizer, collate_fn=data_collator, shuffle=False, drop_remainder=True)

optimizer = keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer)

from rouge_score import rouge_scorer

rouge_l = rouge_scorer.RougeScorer(['rougeL'])

def metric_fn(eval_predictions):
    predictions, labels = eval_predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_l(decoded_labels, decoded_predictions)
    result = {"RougeL": result["f1_score"]}

    return result

from transformers.keras_callbacks import KerasMetricCallback

metric_callback = KerasMetricCallback(
    metric_fn, eval_dataset=test_dataset)

callbacks = [metric_callback]

model.fit(train_dataset, validation_data=test_dataset, epochs=5, verbose=True)

model.save_weights('summarized_model')



