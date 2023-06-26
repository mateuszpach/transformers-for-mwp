import re


def extract_number_from_answer(answer):
    """
    Extract the numerical value from model's answer.
    Finds the last number in the answer
    """
    j = 0
    answer = answer.replace(',', '')  # changes 50,000 into 50000
    n = len(answer)
    ans = -1
    answer = answer + '.'
    for i in range(n):
        if j <= i:
            j = i + 1
        if answer[i] == '-':  # float('-') would throw ValueError despite being correct number prefix
            j = max(j, i + 2)
        while j <= n:
            try:
                tmp_ans = float(answer[i:j])
                ans = tmp_ans
                j += 1
            except ValueError:
                break
    return float(ans)
    """
    if answer.endswith('.'):
        answer = answer[:-1]

    answer = answer.split('The answer is ')[-1].strip('$')

    pattern = r"-?\d*\.?\d+"
    number = re.findall(pattern, answer)

    if len(number) == 0:
        return -1

    return number[0]
    """


def split_question(problem):
    """
    Separates the last sentence from the problem and treats it as the question
    """
    sentences = problem.strip().split('.')
    split = 1
    while len('.'.join(sentences[-split:])) < 5 and split < len(sentences):
        split += 1
    problem = '.'.join(sentences[:-split])
    question = '.'.join(sentences[-split:])
    return problem, question


def load_strategy(f):
    """
    load strategy's state graph from file
    Format:
        STATE_NAME
        ####
        PROMPT
        ######
        regex1 ### NEXT_STATE1
        regex2 ### NEXT_STATE2
        ...
        ######
    Strategy needs to have START state and terminate in END state (transitioning to end state terminates conversation).
    Regexes are checked in order of appearance and are case-insensitive,
        make sure every possible response is covered by regexes.

    Empty lines and "# ...." (python comment lines) are ignored

    return
        prompts - state_name -> prompt dictionary
        states - state_name -> (regex -> state_name dictionary) dictionary
    """
    states = {}
    prompts = {}
    phase = 0  # 0 - name, 1 - prompt, 2 - transitions
    prompt = None
    current_state = None
    for line in f:
        line = line.strip()
        if line == '####':
            prompt = None
            phase = 1
            continue
        elif line == '######':
            prompts[current_state] = prompt
            phase = 2
            continue
        elif line == '########':
            phase = 0
            continue
        elif re.fullmatch('#.*', line, re.DOTALL) is not None or len(line) == 0:
            # skip comment lines and empty lines
            continue
        if phase == 0:
            current_state = line.upper()
            states[current_state] = {}
        elif phase == 1:
            if prompt is None:
                prompt = line
            else:
                prompt += '\n' + line
        elif phase == 2:
            line = line.split('###')
            regex = line[0].strip().lower()
            next_state = line[1].strip().upper()
            states[current_state][regex] = next_state
    print('Interactive strategy state graph:', states)
    return prompts, states


def log_conversation(conversation):
    out = '\n'
    for msg in conversation:
        out = out + f'{msg["role"].strip().upper()}  : {msg["content"].strip()}\n'
    return out
