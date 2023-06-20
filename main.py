import os
import openai
import csv
import time
import json
import argparse
import re

import utils
import wandb

from utils import extract_number_from_answer

# OpenAI API key
key = os.getenv("OPENAI_API_KEY")
if key is None:
    try:
        with open("OPENAI_API_KEY") as f:
            for line in f:
                key = line.strip()
                break
    except FileNotFoundError:
        print("Neither OPENAI_API_KEY file nor environmental variable was found ")
        exit(1)
openai.api_key = key

run = wandb.init(
    # Set the wandb project where this run will be logged
    project="mwp",
)
config = run.config


def data_reader(args):
    """
    Read and parse data from a JSON file to get the numerical answer.
    """
    decoder = json.JSONDecoder()
    questions = []
    answers = []

    if args.dataset in ["gsm8k", "gsm8k_train"]:
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))

    else:
        raise ValueError(f"Dataset {args.dataset} not supported!")

    print("Dataset: {}".format(args.dataset))
    print("Dataset size: {}".format(len(answers)))

    return questions, answers


def get_response(prompt_input, eng, max_tokens, temperature):
    """
    Send request using OpenAI's Chat API.
    """
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=eng,
                messages=prompt_input,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            time.sleep(3)
            return response
        except:
            print('.', end='', flush=True)
            time.sleep(5)

    return response


def interactive_get_answer_from_model(prompts, states, problem, question, assistant_role, args):
    """
    Get an answer from model.
    """
    eng = args.eng
    max_tokens = args.max_tokens
    temperature = args.temp
    if "gpt-3.5-turbo" in eng:
        initial_prompt = prompts['START']
        current_state = 'START'
        initial_prompt = initial_prompt.replace(args.interactive_strategy_Q_tag, question)
        initial_prompt = initial_prompt.replace(args.interactive_strategy_P_tag, problem)
        conversation = [
            {"role": "system", "content": assistant_role},
        ]
        new_prompt = initial_prompt
        while max_tokens > 0 and current_state != 'END':
            conversation.append({"role": "user", "content": new_prompt})

            # print("Q",new_prompt)
            response = get_response(conversation, eng, max_tokens, temperature)
            # print("A",response['choices'][0]['message']["content"].strip())

            max_tokens -= response['usage']['total_tokens']
            response_msg = response['choices'][0]['message']
            response_content = response['choices'][0]['message']['content'].strip()
            conversation.append(response_msg)
            for regex, state in states[current_state].items():
                # Normally the dot matches any character except newlines.
                if re.fullmatch(regex, response_content.lower(), re.DOTALL) is not None:
                    current_state = state
                    break
            if current_state == 'END':
                # there is no prompts['END']
                break
            new_prompt = prompts[current_state]
            new_prompt = new_prompt.replace(args.interactive_strategy_Q_tag, question)
            new_prompt = new_prompt.replace(args.interactive_strategy_P_tag, problem)

        if current_state != 'END' and args.interactive_strategy_failsafe is not None:
            new_prompt = prompts[args.interactive_strategy_failsafe]
            new_prompt = new_prompt.replace(args.interactive_strategy_Q_tag, question)
            new_prompt = new_prompt.replace(args.interactive_strategy_P_tag, problem)
            conversation.append({"role": "user", "content": new_prompt})

            # print("Q",new_prompt)
            response = get_response(conversation, eng, args.max_tokens // 2, temperature)
            # print("A",response['choices'][0]['message']["content"].strip())

            max_tokens -= response['usage']['total_tokens']
            response_msg = response['choices'][0]['message']
            conversation.append(response_msg)
        # print(f"Used {args.max_tokens - max_tokens} tokens")
        return response['choices'][0]['message']["content"].strip(), conversation

    else:
        raise ValueError(f"Engine {eng} not supported!")


def get_answer_from_model(prompt, question, assistant_role, eng, max_tokens, temperature):
    """
    Get an answer from model.
    """
    if "gpt-3.5-turbo" in eng:
        prompt_input = prompt + "\n\nQ: {}\nA:".format(question)
        conversation = [
            {"role": "system", "content": assistant_role},
            {"role": "user", "content": prompt_input},
        ]
        response = get_response(conversation, eng, max_tokens, temperature)

        return response['choices'][0]['message']["content"].strip()

    else:
        raise ValueError(f"Engine {eng} not supported!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", type=str, help="Dataset to use")
    parser.add_argument("--eng", default="gpt-3.5-turbo", type=str, help="Engine")
    # gpt-3.5-turbo-0613 has 60 responses per minute limit (default is 3)
    parser.add_argument("--temp", default=0.0, type=float, help="Temperature for generation")
    parser.add_argument("--max_tokens", default=1024, type=int, help="Max # of tokens for generation. In interactive "
                                                                     "mode, at most 1.5*max_tokens will be used "
                                                                     "through out whole conversation")
    parser.add_argument("--test_size", default=3, type=int, help="Size of the dataset to test")
    parser.add_argument("--prompt", default="gsm8k", type=str, help="Prompt to use")
    parser.add_argument("--save_correct", default="false", type=bool, help="Save correct model answers")
    parser.add_argument("--assistant_role", default="default", type=str,
                        help="Assistants role in the reasoning process. Available: default/concise")
    parser.add_argument("--split_PQ", default=False, type=bool,
                        help="Split task into Problem and Question parts (used only in interactive strategy)")
    parser.add_argument("--interactive_strategy", default=False, type=bool,
                        help="If graph state interactive strategy is provided")
    parser.add_argument("--interactive_strategy_Q_tag", default="##Q##", type=str,
                        help="Tag used in int. strategy that is to be replaced with question")
    parser.add_argument("--interactive_strategy_P_tag", default="##P##", type=str,
                        help="Tag used in int. strategy that is to be replaced with question")
    parser.add_argument("--interactive_strategy_failsafe", default=None, type=str,
                        help="State from which to prompt as a last chance to obtain the answer if max_tokens was reached without producing the answer")

    args = parser.parse_args()
    assistants = {
        'default': 'You are an assistant who helps with math problems.',
        'concise': 'You are a concise mathematician who carefully solves maths problems',
    }
    assistant = assistants[args.assistant_role]

    config.dataset = args.dataset
    config.eng = args.eng
    config.temp = args.temp
    config.max_tokens = args.max_tokens
    config.test_size = args.test_size

    dataset_paths = {"gsm8k": "gsm8k/gsm8k.jsonl", "gsm8k_train": "gsm8k_train/gsm8k_train.jsonl"}

    args.dataset_path = "dataset/{}".format(dataset_paths[args.dataset])

    if args.save_correct:
        correct_answers = []

    questions, answers = data_reader(args)
    qa_pairs = [(questions[idx], answers[idx]) for idx in range(len(questions))]
    print(f"Loading dataset complete. Altogether {len(questions)} questions.")

    # Save the extracted numerical values
    with open(f"dataset/{args.dataset}/numbers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for number in answers:
            writer.writerow([number])

    prompts = None
    states = None
    prompt = None

    if args.interactive_strategy:
        # Load interactive strategy prompts and state graph
        with open(f"prompt/gsm8k/{args.prompt}.txt", "r", encoding='utf-8') as f:
            prompts, states = utils.load_strategy(f)
        model_output_logs = open(f'dataset/{args.dataset}/model_output_{args.prompt}_logs.txt', 'w')
    else:
        # Load the initial prompt
        with open(f"prompt/gsm8k/{args.prompt}.txt", "r", encoding='utf-8') as f:
            prompt = f.read().strip()

    config.prompt = prompt

    count = 0
    correct = 0.0
    acc = 0
    model_output = open(f'dataset/{args.dataset}/model_output_{args.prompt}.txt', 'w', encoding="utf-8")

    table = wandb.Table(columns=["question", "answer"])

    for num, (question, answer) in enumerate(qa_pairs[:min(args.test_size, len(qa_pairs))]):
        count += 1
        problem = ''
        if args.split_PQ:
            problem, question = utils.split_question(question)
        if args.interactive_strategy:
            model_answer, conversation = interactive_get_answer_from_model(prompts=prompts, states=states,
                                                                           problem=problem, question=question,
                                                                           assistant_role=assistant, args=args)
            model_output_logs.write(f'P:"{problem}"\nQ:"{question}"\nConv:{utils.log_conversation(conversation)}\n')
        else:
            # Try to get the answer
            model_answer = get_answer_from_model(prompt=prompt, question=question, assistant_role=assistant,
                                                 eng=args.eng, max_tokens=args.max_tokens, temperature=args.temp)

        model_answer = ' '.join(model_answer.split())
        table.add_data(question, model_answer)

        model_answer_value = extract_number_from_answer(model_answer)

        model_output.write(f"The correct answer is: #{answer}#\n")
        model_output.write(f'Model answer: #{model_answer_value}# extracted from "{model_answer}"\n')
        model_output.write('-' * 50 + '\n')
        if args.interactive_strategy:
            model_output_logs.write(f"The correct answer is: #{answer}#\n")
            model_output_logs.write(f'Model answer: #{model_answer_value}# extracted from "{model_answer}"\n')
            model_output_logs.write('-' * 50 + '\n')

        if model_answer_value == extract_number_from_answer(answer):
            correct += 1
            if args.save_correct:
                correct_answers.append({"question": question, "answer": model_answer})


        acc = correct / count
        print("Correct ratio: ", acc)

    model_output.close()
    run.log({"accuracy": acc})
    run.log({"table_qa": table})

    if args.save_correct:
        with open(f"dataset/{args.dataset}/model_output_{args.prompt}_correct.jsonl", "w") as file:
            for item in correct_answers:
                json_string = json.dumps(item)
                file.write(json_string + '\n')


if __name__ == '__main__':
    main()
