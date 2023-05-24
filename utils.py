import re


def extract_number_from_answer(answer):
    """
    Extract the numerical value from model's answer.
    """
    if answer.endswith('.'):
        answer = answer[:-1]

    answer = answer.split('The answer is ')[-1].strip('$')

    print(answer)
    pattern = r"-?\d*\.?\d+"
    number = re.findall(pattern, answer)[0]

    return number