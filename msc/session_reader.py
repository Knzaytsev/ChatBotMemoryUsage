import json
import random
import re

with open('flan_ann_inputs.json', 'r') as f:
    init_dialogs = json.loads(f.read())

with open('./data/msc/msc_personasummary/session_2/train.txt', 'r') as f:
    dialog_continuation = [json.loads(dialog) for dialog in list(f)]

ids = [dialog['dialog_id'] for dialog in init_dialogs]
filtered_dialogs = [dialog for dialog in dialog_continuation if dialog['initial_data_id'] in ids]

with open('./data/cont_dialogs.json', 'w') as f:
    json.dump(filtered_dialogs, f)
