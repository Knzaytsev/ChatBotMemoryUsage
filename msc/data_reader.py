import json
import random
import re

sessions = ['session_1', 'session_2', 'session_3']

sessions_data = list()

for session in sessions:
    with open(f'./msc/msc_personasummary/{session}/train.txt', 'r') as f:
        data = [json.loads(dialog) for dialog in list(f)]
        data = {dialog['initial_data_id']: dialog for dialog in data}

    sessions_data.append(data)

from pprint import pprint
pprint(sessions_data[0]['train:ordered_7140'])
# concat_sessions = list()
# for id, dialog in sessions_data[0].items():
#     concat_session = {'session_1': dialog}

#     for i, session in enumerate(sessions_data[1:]):
#         if id in session:
#             concat_session[f'session_{i+2}'] = session[id]

#     concat_sessions.append(concat_session)

# sample_full_sessions = random.sample(list(filter(lambda x: len(x) == 3, concat_sessions)), 500)

# with open('sample_data_summ_sess.json', 'w') as f:
#     json.dump(sample_full_sessions, f)

# print(data[0]['initial_data_id'])

# sample_dialogs = random.sample(data, 500)

# with open('sample_data_summ.json', 'w') as f:
#     json.dump(sample_dialogs, f)

# def replace_seq(x, patterns):
#     for pattern in patterns:
#         x = re.sub(*pattern, x)
#     return x

# with open('flan_outputs.json', 'r') as f:
#     data = json.loads(f.read())

# sub_patterns = [(r'(\d\.)', r'\n\1'), (r'([a-z]+\.)', r'\1\n'), (r'\n+', r'\n')]
# new_line_replacer = lambda x: [fact.strip() for fact in replace_seq(x, sub_patterns).split('\n') if fact]

# dialog_data = [{'dialogue': [{'author': replica['id'], 'text': replica['text']} 
#                             for replica in dialog['dialog']], 
#                 's1': [{'value': fact.strip()} for fact in new_line_replacer(dialog['facts']['bot_0'])],
#                 's2': [{'value': fact.strip()} for fact in new_line_replacer(dialog['facts']['bot_1'])],
#                 'summary': dialog['summarization']['Summarization'],
#                 'dialog_id': dialog['initial_data_id']} 
#                 for dialog in data]

# # # print(dialog_data[0].keys())

# with open('flan_ann_inputs.json', 'w') as f:
#     json.dump(dialog_data, f)
