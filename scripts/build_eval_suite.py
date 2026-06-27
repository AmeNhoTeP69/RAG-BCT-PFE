"""
build_eval_suite.py
───────────────────
Generates eval_questions.json — a balanced 200-question evaluation suite for the
BCT RAG system, following the methodology requested by the supervisor (RandomForce
style: balanced categories, documented expected behavior, honest scoring).

Every question carries:
  id, question, language (fr|ar|en), category, difficulty (easy|medium|hard),
  expected_behavior (answer | refuse | not_found)

Design rules respected:
  - Answerable questions cover only themes actually present in the BCT corpus
    (the 8 LDA topics + real laws: loi 2016-48 bancaire, loi 2015-26 LCB-FT,
    RTGS/SGMT Elyssa, etc.). No invented regulations.
  - refuse  -> clearly off-topic (cooking, sport, weather...). System must decline.
  - not_found -> plausible-sounding but genuinely absent specifics (fake circular
    numbers, crypto rules, exact capital amounts). System must say it doesn't know.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "eval_questions.json"

# (text, language, difficulty)
POOLS = {
    # ── ANSWERABLE — Anti-money laundering / KYC (loi 2015-26, CTAF) ───────────
    ("aml_kyc", "answer"): [
        ("Quelles sont les exigences en matière de LCB-FT pour les banques tunisiennes ?", "fr", "easy"),
        ("Quelles sont les obligations de vigilance à l'égard de la clientèle (KYC) ?", "fr", "medium"),
        ("Comment les banques doivent-elles déclarer les opérations suspectes à la CTAF ?", "fr", "medium"),
        ("Quelles sont les obligations de lutte contre le financement du terrorisme ?", "fr", "medium"),
        ("Quelles mesures de vigilance renforcée s'appliquent aux personnes politiquement exposées ?", "fr", "hard"),
        ("Quelles sont les obligations d'identification du bénéficiaire effectif ?", "fr", "hard"),
        ("Quel est le dispositif de gel des avoirs prévu par la réglementation BCT ?", "fr", "hard"),
        ("Quelles sont les obligations de formation du personnel en matière de LCB-FT ?", "fr", "medium"),
        ("ما هي التزامات مكافحة غسل الأموال بالنسبة للبنوك التونسية؟", "ar", "easy"),
        ("ما هي إجراءات اليقظة تجاه العملاء المعروفة باسم اعرف عميلك؟", "ar", "medium"),
        ("كيف يجب على البنوك الإبلاغ عن العمليات المشبوهة؟", "ar", "medium"),
        ("ما هي تدابير اليقظة المعززة تجاه الأشخاص المعرضين سياسيا؟", "ar", "hard"),
        ("What are the anti-money laundering requirements for Tunisian banks?", "en", "easy"),
        ("What customer due diligence obligations apply to banks?", "en", "medium"),
        ("How should banks report suspicious transactions?", "en", "medium"),
        ("What are the obligations regarding identification of the beneficial owner?", "en", "hard"),
        ("Quelles sont les obligations relatives aux virements électroniques transfrontaliers ?", "fr", "hard"),
        ("What record-keeping obligations apply to customer identification documents?", "en", "medium"),
        ("ما هي التزامات اليقظة المستمرة تجاه العمليات البنكية؟", "ar", "hard"),
    ],
    # ── ANSWERABLE — Governance & capital adequacy (loi 2016-48, Bâle) ─────────
    ("governance", "answer"): [
        ("Quelles sont les règles de gouvernance des banques en Tunisie ?", "fr", "easy"),
        ("Quelles sont les exigences en matière d'adéquation des fonds propres ?", "fr", "medium"),
        ("Quel est le rôle du conseil d'administration dans la gouvernance bancaire ?", "fr", "medium"),
        ("Quelles sont les missions du comité d'audit d'une banque ?", "fr", "medium"),
        ("Quelles sont les exigences relatives au système de contrôle interne ?", "fr", "hard"),
        ("Quelles sont les règles d'incompatibilité pour les dirigeants de banques ?", "fr", "hard"),
        ("Comment est défini le ratio de solvabilité des établissements de crédit ?", "fr", "medium"),
        ("Quelles sont les obligations relatives à la gestion des risques opérationnels ?", "fr", "hard"),
        ("Quelles sont les exigences de fonds propres pour la couverture des risques ?", "fr", "hard"),
        ("ما هي قواعد حوكمة البنوك في تونس؟", "ar", "easy"),
        ("ما هي متطلبات كفاية الأموال الذاتية للبنوك؟", "ar", "medium"),
        ("ما هو دور مجلس الإدarة في الحوكمة البنكية؟", "ar", "medium"),
        ("ما هي متطلبات نظام الرقابة الداخلية في البنوك؟", "ar", "hard"),
        ("What are the corporate governance rules for banks in Tunisia?", "en", "easy"),
        ("What are the capital adequacy requirements for credit institutions?", "en", "medium"),
        ("What is the role of the audit committee in a bank?", "en", "hard"),
        ("Quelles sont les obligations de publication des états financiers des banques ?", "fr", "medium"),
        ("What are the fit-and-proper requirements for bank executives?", "en", "hard"),
        ("ما هي مهام لجنة المخاطر داخل البنك؟", "ar", "medium"),
    ],
    # ── ANSWERABLE — Forex & bureau de change (intermédiaires agréés) ──────────
    ("forex_change", "answer"): [
        ("Quelles sont les conditions pour ouvrir un bureau de change ?", "fr", "easy"),
        ("Quelles sont les obligations des intermédiaires agréés en devises ?", "fr", "medium"),
        ("Quelles sont les règles applicables aux opérations de change manuel ?", "fr", "medium"),
        ("Comment s'effectue le rapatriement des recettes d'exportation ?", "fr", "hard"),
        ("Quelles sont les conditions d'ouverture d'un compte en devises ?", "fr", "medium"),
        ("Quelles sont les obligations de déclaration des transferts en devises ?", "fr", "hard"),
        ("Quel est le régime de l'allocation touristique pour les résidents ?", "fr", "medium"),
        ("Quelles sont les obligations comptables d'un bureau de change manuel ?", "fr", "hard"),
        ("ما هي شروط فتح مكتب صرافة؟", "ar", "easy"),
        ("ما هي التزامات الوسطاء المعتمدين في العملة الأجنبية؟", "ar", "medium"),
        ("ما هي القواعد المطبقة على عمليات الصرف اليدوي؟", "ar", "medium"),
        ("ما هي شروط فتح حساب بالعملة الأجنبية؟", "ar", "hard"),
        ("What are the requirements for foreign exchange operations?", "en", "easy"),
        ("What are the conditions to open a manual exchange office?", "en", "medium"),
        ("What are the obligations of authorized foreign-exchange intermediaries?", "en", "medium"),
        ("How is the repatriation of export proceeds regulated?", "en", "hard"),
        ("Quelles sont les règles de cotation des devises par les intermédiaires agréés ?", "fr", "hard"),
        ("What are the rules for non-resident foreign-currency accounts?", "en", "hard"),
    ],
    # ── ANSWERABLE — Payment systems (RTGS/SGMT Elyssa, chèques) ───────────────
    ("payments", "answer"): [
        ("Quels sont les participants au système RTGS Elyssa ?", "fr", "easy"),
        ("Comment fonctionne le système de paiement de gros montants en Tunisie ?", "fr", "medium"),
        ("Quelles sont les règles applicables au système de télécompensation ?", "fr", "medium"),
        ("Quelles sont les obligations relatives au traitement des chèques sans provision ?", "fr", "hard"),
        ("Quel est le cadre de surveillance des systèmes de paiement par la BCT ?", "fr", "hard"),
        ("Quelles sont les règles relatives à la monnaie électronique ?", "fr", "medium"),
        ("Comment sont gérés les incidents de paiement par carte bancaire ?", "fr", "hard"),
        ("ما هم المشاركون في نظام الدفع الإجمالي الفوري إليسا؟", "ar", "easy"),
        ("كيف يعمل نظام المقاصة الإلكترونية في تونس؟", "ar", "medium"),
        ("ما هي القواعد المتعلقة بالنقود الإلكترونية؟", "ar", "hard"),
        ("What are the participants in the RTGS Elyssa system?", "en", "easy"),
        ("How does the large-value payment system work in Tunisia?", "en", "medium"),
        ("What rules govern electronic money in Tunisia?", "en", "hard"),
        ("What is the BCT oversight framework for payment systems?", "en", "hard"),
        ("Quelles sont les obligations des établissements de paiement en Tunisie ?", "fr", "medium"),
        ("ما هي قواعد المقاصة بين البنوك في تونس؟", "ar", "medium"),
        ("What security requirements apply to electronic payment instruments?", "en", "hard"),
    ],
    # ── ANSWERABLE — SME credit & financing ───────────────────────────────────
    ("credit_financing", "answer"): [
        ("Comment financer une PME selon la réglementation de la BCT ?", "fr", "easy"),
        ("Quelles sont les conditions de refinancement des crédits accordés aux PME ?", "fr", "medium"),
        ("Quelles sont les règles relatives à la classification des créances ?", "fr", "hard"),
        ("Quelles sont les obligations de provisionnement des créances classées ?", "fr", "hard"),
        ("Quel est le cadre des lignes de crédit extérieures pour le financement ?", "fr", "medium"),
        ("Quelles sont les règles applicables au crédit à la consommation ?", "fr", "medium"),
        ("Quelles sont les conditions d'octroi des microcrédits ?", "fr", "medium"),
        ("ما هي شروط تمويل المؤسسات الصغرى والمتوسطة لدى البنك المركزي؟", "ar", "easy"),
        ("ما هي القواعد المتعلقة بتصنيف الديون البنكية؟", "ar", "hard"),
        ("ما هي شروط إعادة تمويل القروض الممنوحة للمؤسسات؟", "ar", "medium"),
        ("How can an SME be financed under BCT regulations?", "en", "easy"),
        ("What are the rules for the classification of bank receivables?", "en", "hard"),
        ("What are the conditions for refinancing loans granted to SMEs?", "en", "medium"),
        ("What rules apply to consumer credit in Tunisia?", "en", "medium"),
        ("Quelles sont les règles relatives au ratio de division des risques ?", "fr", "hard"),
        ("ما هي قواعد تكوين المدخرات لمواجهة الديون المصنفة؟", "ar", "hard"),
    ],
    # ── ANSWERABLE — Monetary policy (taux directeur, réserve obligatoire) ─────
    ("monetary_policy", "answer"): [
        ("Quels sont les instruments de politique monétaire de la BCT ?", "fr", "medium"),
        ("Comment fonctionne l'appel d'offres de refinancement de la BCT ?", "fr", "hard"),
        ("Quelles sont les règles relatives à la réserve obligatoire des banques ?", "fr", "medium"),
        ("Quelles garanties sont éligibles aux opérations de politique monétaire ?", "fr", "hard"),
        ("Quel est le rôle du taux directeur dans la politique monétaire ?", "fr", "easy"),
        ("Comment la BCT intervient-elle sur le marché monétaire ?", "fr", "medium"),
        ("ما هي أدوات السياسة النقدية للبنك المركزي التونسي؟", "ar", "medium"),
        ("ما هي القواعد المتعلقة بالاحتياطي الإجباري للبنوك؟", "ar", "medium"),
        ("ما هو دور نسبة الفائدة المديرية في السياسة النقدية؟", "ar", "easy"),
        ("What are the monetary policy instruments of the BCT?", "en", "medium"),
        ("How does the BCT refinancing tender operate?", "en", "hard"),
        ("What are the reserve requirement rules for banks?", "en", "medium"),
        ("Quelles sont les conditions de la facilité de dépôt auprès de la BCT ?", "fr", "hard"),
        ("What collateral is eligible for BCT open-market operations?", "en", "hard"),
        ("ما هي شروط تسهيلات الإيداع لدى البنك المركزي؟", "ar", "hard"),
    ],
    # ── ANSWERABLE — Specific REAL circulaires / laws ─────────────────────────
    ("specific_circulaires", "answer"): [
        ("Quelles sont les obligations de la circulaire 2017-08 ?", "fr", "medium"),
        ("Quel est l'objet de la circulaire BCT 2016-06 ?", "fr", "medium"),
        ("Quels sont les liens entre la loi 2016-48 et les circulaires de la BCT ?", "fr", "hard"),
        ("Que prévoit la loi bancaire 2016-48 relative aux banques et établissements financiers ?", "fr", "medium"),
        ("Quelles dispositions introduit la circulaire 2018-06 ?", "fr", "hard"),
        ("Quelle est la portée de la loi 2015-26 relative à la LCB-FT ?", "fr", "medium"),
        ("ما هو موضوع منشور البنك المركزي عدد 2017-08؟", "ar", "medium"),
        ("ماذا ينص عليه القانون البنكي عدد 2016-48؟", "ar", "hard"),
        ("What does circular 2017-08 require from banks?", "en", "medium"),
        ("What is the scope of banking law 2016-48?", "en", "hard"),
        ("Quel est l'objet de la circulaire 2018-10 de la BCT ?", "fr", "hard"),
        ("Que prévoit la circulaire 2020-05 de la BCT ?", "fr", "medium"),
        ("What is the purpose of BCT circular 2016-06?", "en", "medium"),
    ],
    # ── ANSWERABLE — Intermediaries & devises (cross-border) ───────────────────
    ("intermediaries", "answer"): [
        ("Quelles sont les obligations de reporting des intermédiaires agréés ?", "fr", "medium"),
        ("Quelles opérations courantes les intermédiaires agréés peuvent-ils exécuter ?", "fr", "hard"),
        ("Quelles sont les règles applicables aux comptes étrangers en dinars convertibles ?", "fr", "hard"),
        ("Quelles sont les obligations en matière de position de change des banques ?", "fr", "hard"),
        ("ما هي التزامات التصريح للوسطاء المعتمدين؟", "ar", "medium"),
        ("ما هي القواعد المتعلقة بمركز الصرف لدى البنوك؟", "ar", "hard"),
        ("What reporting obligations apply to authorized intermediaries?", "en", "medium"),
        ("What rules apply to convertible-dinar foreign accounts?", "en", "hard"),
        ("Quelles sont les conditions d'octroi de la qualité d'intermédiaire agréé ?", "fr", "medium"),
        ("Quelles sont les obligations de couverture des opérations en devises ?", "fr", "hard"),
        ("Quelles sont les obligations de rapatriement pour les exportateurs ?", "fr", "medium"),
        ("ما هي شروط منح صفة الوسيط المعتمد؟", "ar", "medium"),
        ("What hedging obligations apply to foreign-currency transactions?", "en", "hard"),
    ],
    # ── REFUSE — clearly off-topic (system must decline) ──────────────────────
    ("off_topic", "refuse"): [
        ("Comment préparer un couscous tunisien traditionnel ?", "fr", "easy"),
        ("Qui a gagné la dernière Coupe du monde de la FIFA ?", "fr", "easy"),
        ("Quelle est la météo prévue demain à Tunis ?", "fr", "easy"),
        ("Donne-moi une recette de pâtisserie au miel.", "fr", "easy"),
        ("Quels sont les meilleurs films à voir cette année ?", "fr", "easy"),
        ("Comment réparer un moteur de voiture diesel ?", "fr", "medium"),
        ("Quelle est la capitale de l'Australie ?", "fr", "easy"),
        ("Peux-tu m'écrire un poème sur l'amour ?", "fr", "easy"),
        ("Quels exercices de musculation pour les bras ?", "fr", "easy"),
        ("Comment apprendre à jouer de la guitare ?", "fr", "easy"),
        ("ما هو سعر البترول اليوم؟", "ar", "easy"),
        ("كيف أحضر طبق الكسكسي التونسي؟", "ar", "easy"),
        ("من فاز بكأس العالم لكرة القدم؟", "ar", "easy"),
        ("ما هي أفضل الوجهات السياحية في العالم؟", "ar", "easy"),
        ("ما هي حالة الطقس غدا في تونس؟", "ar", "easy"),
        ("Who won the last FIFA World Cup?", "en", "easy"),
        ("What is the weather forecast for tomorrow?", "en", "easy"),
        ("Can you recommend a good Italian restaurant?", "en", "easy"),
        ("How do I train for a marathon?", "en", "easy"),
        ("What is the best smartphone to buy in 2025?", "en", "easy"),
        ("Write me a short story about a dragon.", "en", "easy"),
        ("Comment planter des tomates dans un jardin ?", "fr", "easy"),
        ("Quelle est la distance entre la Terre et la Lune ?", "fr", "medium"),
        ("Quels sont les bienfaits du yoga pour la santé ?", "fr", "easy"),
        ("Comment fonctionne un réacteur nucléaire ?", "fr", "hard"),
        ("ما هي فوائد ممارسة الرياضة يوميا؟", "ar", "easy"),
        ("كيف أتعلم اللغة الإنجليزية بسرعة؟", "ar", "easy"),
        ("What movies are playing in cinemas this week?", "en", "easy"),
        ("How do airplanes stay in the air?", "en", "medium"),
        ("Quelle équipe de football est la meilleure au monde ?", "fr", "easy"),
        ("Donne-moi des conseils pour perdre du poids.", "fr", "easy"),
        ("ما هو أفضل نظام غذائي لإنقاص الوزن؟", "ar", "easy"),
        ("What is the tallest mountain in the world?", "en", "easy"),
        ("Comment décorer un salon moderne ?", "fr", "easy"),
        ("Quelles sont les règles du jeu d'échecs ?", "fr", "medium"),
    ],
    # ── NOT_FOUND — plausible but genuinely absent from the corpus ────────────
    ("absent", "not_found"): [
        ("Quelles sont les dispositions de la circulaire 1985-03 ?", "fr", "medium"),
        ("Quelles sont les règles de la BCT sur les crypto-monnaies et le bitcoin ?", "fr", "medium"),
        ("Quel est le montant exact minimum du capital pour créer une banque en dinars ?", "fr", "hard"),
        ("Que prévoit la circulaire 1990-07 sur les agences bancaires ?", "fr", "medium"),
        ("Quel est le taux directeur exact fixé par la BCT en janvier 2099 ?", "fr", "hard"),
        ("Quelles sont les dispositions de la circulaire 2099-01 ?", "fr", "medium"),
        ("Quel est le montant précis des amendes prévues à l'article 47 bis ?", "fr", "hard"),
        ("Quelles sont les règles BCT concernant les comptes de trading en actions Tesla ?", "fr", "medium"),
        ("Quelle est la politique de la Banque de France sur les taux négatifs ?", "fr", "medium"),
        ("Combien d'agences possède exactement la Banque Centrale de Tunisie ?", "fr", "hard"),
        ("ما هي أحكام المنشور عدد 1985-03؟", "ar", "medium"),
        ("ما هي قواعد البنك المركزي بشأن العملات المشفرة والبيتكوين؟", "ar", "medium"),
        ("ما هو المبلغ الأدنى الدقيق لرأس مال البنك بالدينار؟", "ar", "hard"),
        ("ما هي أحكام المنشور عدد 2099-01؟", "ar", "medium"),
        ("ما هي شروط الترخيص البنكي الدقيقة في تونس بالمبالغ؟", "ar", "hard"),
        ("What are the provisions of circular 1985-03?", "en", "medium"),
        ("What are the BCT rules on cryptocurrencies and bitcoin?", "en", "medium"),
        ("What is the exact minimum capital amount to create a bank in dinars?", "en", "hard"),
        ("What does circular 2099-01 stipulate?", "en", "medium"),
        ("What is the BCT policy on trading Tesla stocks?", "en", "medium"),
        ("Quelles sont les dispositions de la circulaire 2030-99 sur l'intelligence artificielle ?", "fr", "hard"),
        ("Quel est le salaire moyen d'un employé de la BCT ?", "fr", "medium"),
        ("Quelles sont les prévisions de croissance du PIB tunisien pour 2040 ?", "fr", "hard"),
        ("ما هو متوسط راتب موظف البنك المركزي التونسي؟", "ar", "medium"),
        ("ما هي توقعات نمو الاقتصاد التونسي لسنة 2040؟", "ar", "hard"),
        ("What is the average salary of a BCT employee?", "en", "medium"),
        ("What are the GDP growth forecasts for Tunisia in 2040?", "en", "hard"),
        ("Quelles sont les règles de la circulaire 1975-01 sur le crédit agricole ?", "fr", "hard"),
        ("Quel est le numéro de téléphone du gouverneur de la BCT ?", "fr", "easy"),
        ("ما هو رقم هاتف محافظ البنك المركزي؟", "ar", "easy"),
        ("How many employees work at the Central Bank of Tunisia exactly?", "en", "hard"),
        ("Quelle circulaire BCT régit les paris sportifs en ligne ?", "fr", "medium"),
        ("ما هو المنشور الذي ينظم المراهنات الرياضية عبر الإنترنت؟", "ar", "medium"),
        ("What circular governs online sports betting in Tunisia?", "en", "medium"),
        ("Quelles sont les dispositions exactes sur les NFT dans la réglementation BCT ?", "fr", "hard"),
    ],
}


def build():
    """Round-robin interleave across categories so every contiguous batch of 20
    is a balanced mini-suite (a mix of categories, languages and expected
    behaviors) rather than one homogeneous category — better batch methodology."""
    pools = [list(qs) for qs in POOLS.values()]
    metas = list(POOLS.keys())  # [(category, expected), ...]
    ordered = []
    for i in range(max(len(p) for p in pools)):
        for pi, pool in enumerate(pools):
            if i < len(pool):
                text, lang, diff = pool[i]
                category, expected = metas[pi]
                ordered.append((text, lang, diff, category, expected))

    items = []
    for n, (text, lang, diff, category, expected) in enumerate(ordered, start=1):
        items.append({
            "id": f"EVAL_{n:03d}",
            "question": text,
            "language": lang,
            "category": category,
            "difficulty": diff,
            "expected_behavior": expected,
        })
    return items


if __name__ == "__main__":
    items = build()
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    # Distribution report
    from collections import Counter
    by_exp = Counter(i["expected_behavior"] for i in items)
    by_lang = Counter(i["language"] for i in items)
    by_diff = Counter(i["difficulty"] for i in items)
    by_cat = Counter(i["category"] for i in items)
    print(f"Wrote {len(items)} questions -> {OUT.name}")
    print("  expected_behavior:", dict(by_exp))
    print("  language         :", dict(by_lang))
    print("  difficulty       :", dict(by_diff))
    print("  categories       :", dict(by_cat))
    assert len(items) == 200, f"Expected 200, got {len(items)}"
    print("OK: exactly 200 questions.")
