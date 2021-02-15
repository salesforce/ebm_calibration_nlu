import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb

import spacy
nlp = spacy.load("en_core_web_sm")

data = ['John is a very smart person , he lives in Ireland.',
        'Mark Stewart was born and raised in Chicago',
        'Luke Smith has 3 sisters.',
        'Mary is not a nurse.',
        '  Julianne is an engineer.',
        'My brother  Andrew  used to be a lawyer.']

pdata = list(nlp.pipe(data))
for k in range(len(data)):
    print('|' + data[k] + '|', '|' + Perturb.add_typos(data[k]) + '|')

