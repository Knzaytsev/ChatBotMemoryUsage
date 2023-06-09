import json
from os.path import join
from tqdm import tqdm

SESSIONS = ['session_2',]
FILES = ['train.txt']
PATH = './data/msc/msc_personasummary'

new_dialogs = []
for session in tqdm(SESSIONS):
    for file in tqdm(FILES):
        with open(join(PATH, session, file), 'r') as f:
            dialogs = list(f)

        for dialog in tqdm(dialogs):
            dialog = json.loads(dialog)

            metadata = dialog['metadata']
            personas = dialog['personas']
            init_personas = dialog['init_personas']

            phrases = []
            for phrase in dialog['dialog']:
                phrases.append(phrase['id'] + ': ' + phrase['text'])

            new_dialogs = {}

            new_dialog = {'metadata': metadata, 'dialog': '\n'.join(phrases),
                          'personas': personas, 'init_personas': init_personas}
            
            new_dialogs.append(new_dialog)

with open('data.jsonl', 'w') as f:
    json.dump(new_dialogs, f)
