import os
import openai
import csv
import time
import json
import argparse

from utils import extract_number_from_answer


# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def data_reader(args):
    """
    Read and parse data from a JSON file to get the numerical answer.
    """
    decoder = json.JSONDecoder()
    questions = []
    answers = []

    if args.dataset == "gsm8k":
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
    response = openai.ChatCompletion.create(
        model=eng,
        messages=prompt_input,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response


def get_answer_from_model(prompt, question, eng, max_tokens, temperature):
    """
    Get an answer from model.
    """
    if eng == "gpt-3.5-turbo":
        prompt_input = prompt + "\n\nQ: {}\nA:".format(question)
        response = get_response([
            {"role": "system", "content": "You are an assistant who helps with math problems."},
            {"role": "user", "content": prompt_input},
        ], eng, max_tokens, temperature)

        return response['choices'][0]['message']["content"].strip()

    else:
        raise ValueError(f"Engine {eng} not supported!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", type=str, help="Dataset to use")
    parser.add_argument("--eng", default="gpt-3.5-turbo", type=str, help="Engine")
    parser.add_argument("--temp", default=0.0, type=float, help="Temperature for generation")
    parser.add_argument("--max_tokens", default=1024, type=int, help="Max # of tokens for generation")

    args = parser.parse_args()

    dataset_paths = {"gsm8k": "gsm8k/gsm8k.jsonl"}
    args.dataset_path = "dataset/{}".format(dataset_paths[args.dataset])

    questions, answers = data_reader(args)
    qa_pairs = [(questions[idx], answers[idx]) for idx in range(len(questions))]
    print(f"Loading dataset complete. Altogether {len(questions)} questions.")

    # Save the extracted numerical values
    with open(f"dataset/{args.dataset}/numbers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for number in answers:
            writer.writerow([number])

    # Load the initial prompt
    with open(f"prompt/{args.dataset}/{args.dataset}.txt", "r", encoding='utf-8') as f:
        prompt = f.read().strip()

    count = 0
    correct = 0.0
    model_output = open('dataset/gsm8k/model_output.txt', 'w')
    # For testing purposes qa_pairs len set to 2
    for (question, answer) in qa_pairs[:100]:
        count += 1
        while True:
            try:
                # Try to get the answer
                model_answer = get_answer_from_model(prompt, question,
                                                     eng=args.eng, max_tokens=args.max_tokens,
                                                     temperature=args.temp)

                model_answer = ' '.join(model_answer.split())
                model_output.write(model_answer + '\n')

                model_answer = extract_number_from_answer(model_answer)

                if model_answer == answer:
                    correct += 1
                print("Correct ration: ", correct / count)
                time.sleep(21)
                break

            except Exception as e:
                print(repr(e))
                time.sleep(5)

    model_output.close()


if __name__ == '__main__':
    main()
