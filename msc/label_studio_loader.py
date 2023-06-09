import json
import re

PREFIX = '_summary_'

def replace_seq(x, patterns):
    for pattern in patterns:
        x = re.sub(*pattern, x)
    return x

with open(f'sess{PREFIX}outputs.json', 'r') as f:
    data = json.loads(f.read())

fix_str_pattern = (r'([Bb]ot)\\+(_[01])', r'\1\2')
sub_patterns = [(r'(\d\.\*)', r'\n\1'), (r'([a-z]+\.)', r'\1\n'), (r'\n+', r'\n'), fix_str_pattern]
new_line_replacer = lambda x: [fact.strip() for fact in replace_seq(x, sub_patterns).split('\n') if fact]


dialog_data = [{'dialogue': [{'author': replica['id'], 'text': replica['text']} 
                            for replica in dialog['dialog']], 
                # 's1': [{'value': fact.strip()} for fact in new_line_replacer(dialog['facts']['bot_0'])],
                # 's2': [{'value': fact.strip()} for fact in new_line_replacer(dialog['facts']['bot_1'])],
                'summary': replace_seq(dialog['summarization']['Summarization'], [fix_str_pattern]),
                'dialog_id': dialog['initial_data_id'],
                'session': session}
                for dialog in data for session, dialog in dialog.items()]


# for i, dialog in enumerate(dialog_data):
#     incorrect_facts = []
#     for fact in dialog['s1']:
#         if re.findall(r'[\*0-9\.]+\s+[Bb]ot_1', fact['value']):
#             incorrect_facts.append(fact)
#     dialog['s2'] += incorrect_facts
#     dialog['s1'] = [fact for fact in dialog['s1'] if fact not in incorrect_facts]
    
#     incorrect_facts = []
#     for fact in dialog['s2']:
#         if re.findall(r'[\*0-9\.]+\s+[Bb]ot_0', fact['value']):
#             incorrect_facts.append(fact)
#     dialog['s1'] += incorrect_facts
#     dialog['s2'] = [fact for fact in dialog['s2'] if fact not in incorrect_facts]

print(len(dialog_data))

with open(f'sess_ann{PREFIX}inputs.json', 'w') as f:
    json.dump(dialog_data, f)
