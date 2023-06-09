import json
import random

sessions = ['session_1', 'session_2', 'session_3']

sessions_data = list()

for session in sessions:
    with open(f'./data/msc/msc_personasummary/{session}/train.txt', 'r') as f:
        data = [json.loads(dialog) for dialog in list(f)]
        data = {dialog['initial_data_id']: dialog for dialog in data}

    sessions_data.append(data)

concat_sessions = list()
for id, dialog in sessions_data[0].items():
    concat_session = {'session_1': dialog}

    for i, session in enumerate(sessions_data[1:]):
        if id in session:
            concat_session[f'session_{i+2}'] = session[id]

    concat_sessions.append(concat_session)

sample_full_sessions = random.sample(list(filter(lambda x: len(x) == 3, concat_sessions)), 500)

with open('/data/sample_data_summ_sess.json', 'w') as f:
    json.dump(sample_full_sessions, f)
