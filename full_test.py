import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv('.env')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY', '')

from graph_rag_engine import HybridGraphRAGEngine
engine = HybridGraphRAGEngine()

TESTS = [
    ('T01', 'FR - LCB-FT acronym',         'Quelles sont les exigences en matiere de LCB-FT?',                         'answer'),
    ('T02', 'FR - governance',              'Quelles sont les regles de gouvernance des banques en Tunisie?',            'answer'),
    ('T03', 'FR - circular 2017-08',        'Quelles sont les obligations de la circulaire 2017-08?',                    'answer'),
    ('T04', 'FR - RTGS Elyssa',             'Quels sont les participants au systeme RTGS Elyssa?',                       'answer'),
    ('T05', 'FR - PME financing',           'Comment financer une PME selon la BCT?',                                    'answer'),
    ('T06', 'FR - bureau de change',        'Quelles sont les conditions pour ouvrir un bureau de change?',              'answer'),
    ('T07', 'EN - forex operations',        'What are the requirements for foreign exchange operations?',                 'answer'),
    ('T08', 'AR - forex',                   'ما هي متطلبات عمليات الصرف الأجنبي؟', 'answer'),
    ('T09', 'AR - bank license',            'ما هي شروط الترخيص البنكي في تونس؟', 'not_in_corpus'),
    ('T10', 'FR - loi 2016-48',             'Quels sont les liens entre la loi 2016-48 et les circulaires BCT?',         'answer'),
    ('T11', 'FR - fonds propres',           'Quelles sont les exigences en matiere d adequation des fonds propres?',     'answer'),
    ('T12', 'FR - fake circular 1985',      'Quelles sont les dispositions de la circulaire 1985-03?',                   'not_in_corpus'),
    ('T13', 'FR - greeting',                'bonjour',                                                                   'answer'),
    ('T14', 'FR - off-topic food',          'Comment preparer un couscous tunisien?',                                    'refuse'),
    ('T15', 'EN - off-topic sport',         'Who won the last FIFA World Cup?',                                          'refuse'),
    ('T16', 'AR - off-topic oil price',     'ما هو سعر البترول اليوم؟', 'refuse'),
    ('T17', 'FR - capital minimum',         'Quel est le montant minimum du capital pour creer une banque?',             'not_in_corpus'),
    ('T18', 'EN - AML requirements',        'What are the anti-money laundering requirements for Tunisian banks?',       'answer'),
    ('T19', 'FR - crypto',                  'Quelles sont les regles BCT sur les crypto-monnaies?',                      'not_in_corpus'),
    ('T20', 'FR - intermediaires devises',  'Quelles sont les obligations des intermediaires agrees en devises?',        'answer'),
]

def classify(answer, expected):
    ans = answer.lower()
    refused   = 'desolé' in ans or 'désolé' in ans or 'je ne reponds' in ans
    is_error  = ans.startswith('error') or ('erreur' in ans[:40] and 'limite' in ans[:60])
    not_found = any(p in ans for p in [
        "n'est pas disponible", "pas disponible", "ne trouve pas",
        "لا توجد معلومات", "لا أستطيع", "je ne sais pas", "aucune information",
        "non disponible", "not available", "aucun document"
    ])
    if expected == 'refuse':
        return 'PASS' if refused else 'FAIL'
    elif expected == 'not_in_corpus':
        return 'PASS' if (not_found or refused) and not is_error else 'FAIL'
    else:
        return 'FAIL' if (refused or is_error) else 'PASS'

results = []
for tid, label, q, expected in TESTS:
    t0 = time.time()
    try:
        res = engine.query(q)
        answer = res.get('answer', '')
        sources = res.get('sources', [])
        elapsed = time.time() - t0
        status = classify(answer, expected)
        results.append((tid, label, expected, status, elapsed, answer, sources))
    except Exception as e:
        results.append((tid, label, expected, 'ERROR', time.time()-t0, str(e), []))

print('\n' + '='*72)
print('FULL SCALE TEST RESULTS  (20 queries, 3 languages, 4 categories)')
print('='*72)

passed = sum(1 for r in results if r[3] == 'PASS')
failed = sum(1 for r in results if r[3] != 'PASS')

for tid, label, expected, status, elapsed, answer, sources in results:
    mark = 'PASS' if status == 'PASS' else 'FAIL'
    print('\n[' + mark + '] ' + tid + ' | ' + label + ' (' + str(round(elapsed,1)) + 's)')
    src_short = [s.split('(')[0].strip() for s in sources[:3]]
    print('  Expect : ' + expected)
    print('  Sources: ' + str(src_short))
    if status != 'PASS':
        print('  Answer : ' + answer[:220])
    else:
        print('  Answer : ' + answer[:100] + ('...' if len(answer) > 100 else ''))

print('\n' + '='*72)
print('FINAL SCORE: ' + str(passed) + '/20 PASS  |  ' + str(failed) + '/20 FAIL')
print('='*72)
