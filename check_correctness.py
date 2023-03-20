import tqdm
import random
import openai
from indra.assemblers.english import EnglishAssembler

curs = jsonload('indra_assembly_curations.json')
stmts = pklload('../../data/bioexp_asmb_preassembled.pkl')
stmts_by_hash = {s.get_hash(): s for s in stmts}
curated_hashes = {c['pa_hash'] for c in curs}
openai.api_key = #
prompt = 'You need to help me verify if a sentence I give you implies a statement I give you. Provide a correct answer with a simple yes or no. Sentence: "%s" Statement: "%s" Answer:'

curs_sample = random.sample(curs, 100)
responses = []

for cur in tqdm.tqdm(curs_sample):
    stmt = stmts_by_hash[cur['pa_hash']]
    ev = [e for e in stmt.evidence if e.get_source_hash() == cur['source_hash']][0]
    eng_stmt = EnglishAssembler([stmt]).make_model()
    response = openai.Completion.create(model='text-davinci-003',
                                        prompt=prompt % (ev.text, eng_stmt),
                                        temperature=0, max_tokens=1,
                                        top_p=1.0, frequency_penalty=0.0,
                                        presence_penalty=0.0)
    responses.append(response)

choices = [r['choices'][0]['text'].strip() for r in responses]
tags = [c['tag'] for c in curs_sample]
confusion = defaultdict(int)
for choice, tag in zip(choices, tags):
    if choice == 'Yes' and tag == 'correct':
        confusion[('Yes', 'correct')] += 1
    elif choice == 'Yes' and tag != 'correct':
        confusion[('Yes', 'incorrect')] += 1
    elif choice == 'No' and tag == 'correct':
        confusion[('No', 'correct')] += 1
    elif choice == 'No' and tag != 'correct':
        confusion[('No', 'incorrect')] += 1
