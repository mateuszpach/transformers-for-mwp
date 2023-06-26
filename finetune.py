import argparse
import os

import numpy as np
import nltk
import torch
from nltk.tokenize import sent_tokenize
from utils import extract_number_from_answer
import evaluate
from datasets import concatenate_datasets, load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq
from random import randrange
import wandb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default='complex', type=str, help="strategy name")
    parser.add_argument("--model", default='base', type=str, help="T5 version large/base")
    parser.add_argument("--no_cuda", default=False, type=bool, help="Forces cpu training")
    parser.add_argument("--limit_train", default=None, type=int, help="Reduce train ds size to given number")
    parser.add_argument("--limit_test", default=None, type=int, help="Reduce test ds size to given number")
    parser.add_argument("--epochs", default=5, type=int, help="Epochs")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint path")
    parser.add_argument("--allow_longer", default=1.0, type=float, help="Multiplier for max_target length")
    parser.add_argument("--skip_eval", default=False, type=bool, help="Skips final evaluation")

    return parser.parse_args()


def get_dataset(args):
    nltk.download("punkt")

    # Dataset
    dataset_id = args.strategy
    file_path = f"dataset/gsm8k_train/model_output_{dataset_id}_correct.jsonl"
    dataset = load_dataset("json", data_files=file_path)
    test_dataset = load_dataset('json',data_files='dataset/gsm8k/gsm8k.jsonl')

    # Print a sample from the dataset
    sample = dataset["train"][randrange(len(dataset))]
    print(f"question: {sample['question']}")
    print(f"answer: {sample['answer']}")

    if args.model.lower() == 'base':
        model_id = "google/flan-t5-base"
    elif args.model.lower() == 'large':
        model_id = "google/flan-t5-large"
    else:
        raise ValueError('model arg unknown')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenized_inputs = dataset["train"].map(
        lambda x: tokenizer(x["question"], truncation=True), batched=True, remove_columns=["question", "answer"]
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = dataset["train"].map(
        lambda x: tokenizer(x["answer"], truncation=True), batched=True, remove_columns=["question", "answer"]
    )
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")
    tokenized = {'target': tokenized_targets, 'source': tokenized_inputs}
    max_length = {'target':int(max_target_length * args.allow_longer),'source':max_source_length}
    train = {'tokens':tokenized,'length':max_length}
    ids = {'model': model_id, "dataset": dataset_id}

    test_tokenized_inputs = test_dataset["train"].map(
        lambda x: tokenizer(x["question"], truncation=True), batched=True, remove_columns=["question", "answer"]
    )
    test_max_source_length = max([len(x) for x in test_tokenized_inputs["input_ids"]])
    print(f"Max test_source length: {test_max_source_length}")

    test_tokenized_targets = test_dataset["train"].map(
        lambda x: tokenizer(x["answer"], truncation=True), batched=True, remove_columns=["question", "answer"]
    )
    test_max_target_length = max([len(x) for x in test_tokenized_targets["input_ids"]])
    print(f"Max test_target length: {test_max_target_length}")
    test_tokenized = {'target': test_tokenized_targets, 'source': test_tokenized_inputs}
    test_max_length = {'target':int(test_max_target_length * args.allow_longer),'source': test_max_source_length}
    test = {'tokens':test_tokenized,'length':test_max_length}
    datasets = {'train':dataset,'test':test_dataset}
    data = {'train':train,'test':test}
    return tokenizer, data, datasets, ids


def preprocess_function(sample, tokenizer, max_source_length, max_target_length, padding="max_length"):
    # NOTE: you can add prefix here if needed e.g. "translate English to German: "
    inputs = ["" + item for item in sample["question"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=sample["answer"], max_length=max_target_length, padding=padding, truncation=True)
    label_pad_token_id = tokenizer.pad_token_id  # -100

    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else label_pad_token_id) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds[preds == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    correct = 0
    all = 0
    for pred, label in zip(decoded_preds,decoded_labels):
        all += 1
        if extract_number_from_answer(pred) == extract_number_from_answer(label):
            correct += 1

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result['reasoning'] = correct/all
    return result


if __name__ == '__main__':
    args = get_args()
    tokenizer, data, datasets, ids = get_dataset(args)
    # Decoder test
    # for i in range(32128):
    #     print(f'{i} == "{tokenizer.decode([i])}"')
    tokenized_dataset = datasets['train'].map(preprocess_function, batched=True, remove_columns=["question", "answer"],
                                    fn_kwargs={'tokenizer': tokenizer,
                                               'max_target_length': data['train']['length']['target'],
                                               'max_source_length': data['train']['length']['source'], })
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # Model
    model = AutoModelForSeq2SeqLM.from_pretrained(ids['model'])
    global metric
    metric = evaluate.load("rouge")

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = tokenizer.pad_token_id  # -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        padding='longest'
    )

    # Hugging Face repository id
    repository_id = f"{ids['model'].split('/')[1]}-{ids['dataset']}"

    gen_kwargs = {'max_tokens': data['train']['length']['target']}
    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        auto_find_batch_size= True,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=args.epochs,
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # wandb
        report_to="wandb",
        run_name = args.strategy,
        # push to hub parameters
        push_to_hub=False,
        # sets max length in evaluation
        generation_max_length=data['train']['length']['target'],
        no_cuda=args.no_cuda
    )

    wandb.init(project="mwp", entity="mwp", tags=["flan-t5"])
    # Create Trainer instance
    if args.limit_train is not None:
        train_ds = tokenized_dataset["train"].select(range(args.limit_train))
    else:
        train_ds = tokenized_dataset["train"]
    if args.limit_test is not None:
        test_ds = tokenized_dataset["train"].select(range(args.limit_test))
    else:
        test_ds = tokenized_dataset["train"]
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    if args.checkpoint is not None:
        trainer.train(args.checkpoint)
    else:
        trainer.train()
    if not args.skip_eval:
        trainer.evaluate()
        tokenized_test_dataset = datasets['test'].map(preprocess_function, batched=True, remove_columns=["question", "answer"],
                                                  fn_kwargs={'tokenizer': tokenizer,
                                                             'max_target_length': data['test']['length']['target'],
                                                             'max_source_length': data['test']['length']['source'], })
        trainer.evaluate(eval_dataset=tokenized_test_dataset['train'],metric_key_prefix='test/eval')

