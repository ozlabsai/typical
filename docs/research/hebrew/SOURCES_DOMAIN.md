# Hebrew domain sources for Typical decision cases (researched 2026-10-20)

Scope: real Hebrew source material (not benchmark items) for authoring decision cases in the five areas: security,
agent traces, AP/invoice, customer service, and policy compliance. Also covers training data and teacher/translator models.
Fit = H/M/L for authoring decision cases. "Comm?" = whether commercial use is allowed.

**Legal baseline (Israel).** Copyright Act 2007 §6: "no copyright in laws, regulations, Knesset publications, or judicial
decisions of a court or of a governmental agency with adjudicative powers." Unofficial MoJ translation:
https://www.wipo.int/wipolex/en/legislation/details/5016. The rest of gov.il content (guides, reports, alerts) is State
copyright. The MoD, State Comptroller, and PMO sites each show "State copyright, fair use permitted" terms; the main gov.il
terms page returned 403 to the fetcher, so I couldn't check it. Rule: verbatim reuse and redistribution are fine for §6
material. For other gov.il text, paraphrase it into authored cases and don't ship it verbatim.

## 1. Customer service / complaints / FAQ

No open Hebrew call-center or chat-support dialogue corpus exists (HF + web search, Oct 2026). The only ones are
commercial vendors (Andovar, Defined.ai, SpeechDataAI 500h conversational, license "other"/paid).

| Source | URL | License (Comm?) | Size | Format / access | Fit |
|---|---|---|---|---|---|
| MFAQ Hebrew (schema.org FAQPage from Common Crawl; company/service FAQs) | https://huggingface.co/datasets/clips/mfaq | CC0 on packaging; site text not owned by creators (Comm: yes with care) | he: 11,212 QA pairs / 1,921 pages (configs `he`, `he_flat`) | HF parquet, `load_dataset("clips/mfaq","he_flat")`, has `domain` field | H (real Israeli business FAQ policies: returns, billing, delivery) |
| Kol-Zchut (rights portal: benefits, labor, consumer, banking rights) | https://www.kolzchut.org.il/he/ ; terms: https://kolzchut.org.il/llms.txt | CC BY-NC-SA 2.5 IL (**Comm: NO**) | thousands of articles (count not stated); he/ar/ru | No dump or API. Crawl via sitemap.xml with robots Crawl-delay 5; `Accept: text/markdown` per article | H for eligibility/policy decisions, but use as reference only and don't copy into the commercial product |
| Bank of Israel, "Bank-customer relations" supervisory review (complaint stats, selected justified/unjustified cases) | https://boi.org.il/publications/regularpublications/public-enquiries/ | State copyright; fair use (Comm: paraphrase only) | annual PDF, ~100+ pp | PDF download | H (real complaint → regulator ruling, a ready-made yes/no decision) |
| Capital Market, Insurance & Savings Authority, public-inquiries annual report (insurance/pension complaints) | https://www.gov.il/BlobFolder/reports/report_0009/he/report_0009.pdf | State copyright (paraphrase) | annual PDF | PDF | H for insurance claim / complaint decisions |
| data.gov.il, FOI complaints DB (מאגר תלונות חופש המידע) | https://data.gov.il/dataset/hofesh_hameida | "Other (Open)" (Comm: yes, attribute) | small tabular | CSV/XLSX via CKAN API | M (complaint triage fields) |
| omilab Hebrew sentiment (comments on the President's Facebook page) | https://huggingface.co/datasets/omilab/hebrew_sentiment | MIT (yes) | 12,804 comments | HF | L (tone only, not service) |
| AmazonScience MASSIVE he-IL (virtual-assistant intents/slots) | https://huggingface.co/datasets/AmazonScience/massive | CC BY 4.0 (yes) | ~16.5k utterances per locale | HF | L–M (short user utterances, intent labels) |
| Bitext customer support (English; translate to HE) | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset | CDLA-Sharing-1.0 (yes, share-alike on data) | 26,872 QA, 27 intents | HF | M as a translation seed for CS cases |

## 2. Business documents: laws, regulation, courts, procurement, finance, invoices

| Source | URL | License (Comm?) | Size | Format / access | Fit |
|---|---|---|---|---|---|
| Open Book of Laws (Hebrew Wikisource) packaged | https://huggingface.co/datasets/gkour/israeli_law | CC BY-SA 4.0 on wiki edits; statutes are §6-free (yes, SA) | 5,942 pages, ~99.9M chars, ~25M tokens, 335 MB | single JSONL of raw wikitext (title, revid, timestamp) | H (VAT law, Consumer Protection Law, Privacy Protection Regs, Income Tax bookkeeping orders = invoice rules) |
| Hebrew Wikisource directly | https://he.wikisource.org/wiki/ספר_החוקים_הפתוח | CC BY-SA (yes, SA) | same + fresher | MediaWiki API / dumps.wikimedia.org | H |
| Supreme Court of Israel verdicts (2022 snapshot) | https://huggingface.co/datasets/LevMuchnik/SupremeCourtOfIsrael | tag "openrail" (odd for data); judgments are §6-free (Comm: yes for the text) | 751,194 docs, 5.31 GB (2.68 GB text) | HF parquet; ~46 metadata cols (judges, parties, verdict type, dates) | H (contract, tender-petition, labor, insurance disputes with outcomes = labels) |
| Technion/Bar-Ilan Israeli Supreme Court DB 1997–2022 | https://idsai.net.technion.ac.il/files/2026/03/Final-Report_-Israeli-Supreme-Court-Database-1997-2022.pdf | not stated; request access | 249,115 cases, up to 135 coded vars | on request | M (coded outcomes are useful for calibration labels) |
| Knesset Proceedings Corpus (plenary + committee, incl. Finance / Economy committees) | https://huggingface.co/datasets/HaifaCLGroup/KnessetCorpus | CC BY-SA 4.0 (yes, SA) | 35M+ sentences, 1992–2024, 145 GB | HF; JSONL shards (bz2 tar), CoNLL-U, raw doc/pdf | M (regulatory reasoning, policy debates) |
| Knesset open data (OData API mirror: bills, votes, committee sessions) | https://huggingface.co/datasets/ZeAlenu/knesset-data | card text says CC BY-SA 4.0, tag says ODC-By (yes) | 1M–10M rows, 44 tables | partitioned JSONL | L–M (structured metadata) |
| Knesset press releases | https://huggingface.co/datasets/Wissotsky/KnessetNews | not stated (Knesset publications are §6-free) | 10k–100k | parquet | L |
| Govt Procurement Administration, tenders report (דוח מכרזים) | https://data.gov.il/dataset/tenders | "Other (Open)" (yes, attribute) | 5.2 MB CSV, last updated 2021-02-16 | CKAN API: `data.gov.il/api/3/action/package_show?id=tenders` | M (ministry, dates, winner, amount; seeds for procurement-approval cases) |
| Govt exemption contracts (התקשרויות בפטור) | https://data.gov.il/dataset/exemptions | "Other (Open)" (yes) | CSV | CKAN | H for compliance (was the tender exemption justified?) |
| Municipal tenders, e.g. Be'er Sheva | https://data.gov.il/dataset/tender-br7 | "Other (Open)" (yes) | JSON/CSV/XLSX/XML | CKAN | M |
| Insurance / pension product data (ביטוח-נט, פנסיה-נט, גמל-נט) | https://data.gov.il/dataset/pensia-net | "Other (Open)"; ביטוח-נט license unset (yes / verify) | CSV | CKAN | L–M (numeric product terms) |
| HebFinDocs (Israeli listed-company financial statements, doc-VQA with page images) | https://huggingface.co/datasets/asafd60/HebFinDocs | **none stated** (Comm: unclear) | 1k–10k rows | parquet with images, OCR words, bboxes, QA | M for AP/finance multimodal; license blocks use |
| Israel Tax Authority "Invoices Israel" allocation-number model (חשבוניות ישראל) | secondary: https://www.avalara.com/blog/en/europe/2024/04/israel-invoice-allocation-number-mandate.html | spec = State copyright; facts are free | n/a (spec) | no official spec reached; ITA site | H as a **rule source** for synthetic חשבונית מס cases (allocation # needed above a threshold that is falling, ILS 25k in 2024; sources disagree on the 2026–28 schedule) |

Invoices/receipts: no open Hebrew invoice or receipt dataset found. The OCR sets on HF (cyttic/*, isaacmg/*) are line images,
not documents. Recommend synthetic Hebrew tax invoices built from the legal rules (Income Tax bookkeeping orders via the Open
Book of Laws, VAT Law, ITA allocation rules), with HebFinDocs-style rendering only if a license is obtained.

## 3. IT / security text in Hebrew

Openly licensed Hebrew security corpora are effectively nonexistent.

| Source | URL | License (Comm?) | Size | Format / access | Fit |
|---|---|---|---|---|---|
| INCD / CERT-IL unit page (alerts, publications; ~2,480 alerts sent in 2025 per INCD annual report via ICE) | https://www.gov.il/he/departments/units/unit_cert_il | State copyright (paraphrase only) | hundreds of Hebrew alerts/yr | gov.il HTML/PDF; no feed or API found, fetcher got 403 | H for style/realism (Israeli alert format), but must be scraped and paraphrased |
| INCD Israeli Cyber Defense Methodology (ICDM 2.0; NIST CSF 1.1 + 800-53r5) | https://www.gov.il/en/pages/cyber_security_methodology_2 | State copyright | ~100+ pp (EN PDF; HE version not located) | PDF | H as a **policy** source for compliance cases (controls by org tier) |
| data.gov.il INCD cyber-solutions marketplace | https://data.gov.il/dataset/cybermarketplace | "Other (Open)" (yes) | small CSV | CKAN | L |
| NVD CVE records (English → translate) | https://nvd.nist.gov/developers/vulnerabilities | US Gov public domain (yes) | 250k+ CVEs | JSON API | H as a translation seed for incident/patch-priority cases |

## 4. Large general Hebrew corpora (mine for domain pages: banks, insurers, telcos, gov, e-commerce)

| Source | URL | License (Comm?) | Size | Format / access | Fit |
|---|---|---|---|---|---|
| HPLT 3.0 heb_Hebr | https://huggingface.co/datasets/HPLT/HPLT3.0 | CC0 packaging (yes; web text caveats) | 26.1M docs, 37.0B tokens, 47.9 GB | not hosted on HF: `wget -O - https://data.hplt-project.org/three/sorted/heb_Hebr.map \| wget -x -nH --cut-dirs=2 -i -` (JSONL.zst, has URLs) | M (largest; filter by domain/URL) |
| FineWeb-2 heb_Hebr | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | ODC-By 1.0 (yes) | ~24.9 GB parquet (6 files) | HF `data/heb_Hebr/train/*.parquet`; has `url` | M (cleanest; URL filter to *.co.il terms/FAQ/policy pages) |
| CulturaX he (mC4 + OSCAR) | https://huggingface.co/datasets/uonlp/CulturaX | mC4 ODC-By + OSCAR terms; gated auto-accept (yes, verify) | ~13.3 GB parquet (6 parts) | HF gated | M |
| HeDC4 (dedup Common Crawl, HeRo paper) | https://huggingface.co/datasets/HeNLP/HeDC4 | **none stated** | 47.6 GB single CSV | HF | L (license gap; prefer HPLT/FW2) |
| OSCAR 22.01 / 23.01 he | https://huggingface.co/datasets/oscar-corpus/OSCAR-2201 | CC0 packaging; manual gating | multi-GB | HF gated (manual) | L–M |
| Hebrew Wikipedia (20231101.he) | https://huggingface.co/datasets/wikimedia/wikipedia | CC BY-SA 3.0 + GFDL (yes, SA) | ~0.98 GB parquet | HF config `20231101.he`; monthly: https://huggingface.co/datasets/omarkamali/wikipedia-monthly | L–M (entity background) |
| Hebrew news (Ynet/Haaretz-type articles) | https://huggingface.co/datasets/imvladikon/hebrew_news | "other" (Comm: no/unclear) | 100k–1M | HF | L |
| Fine-News multilingual (CC-News, has he) | https://huggingface.co/datasets/ksolovev/fine-news | see card (CC-News derived) | 635M docs all langs | HF | L–M (current events for security/fraud cases) |
| HeQ (SQuAD-style over Wikipedia + Geektime tech news) | https://huggingface.co/datasets/Etelis/HeQ_v1 | CC BY 4.0 (yes) | 30,147 Qs | HF | L (Geektime paragraphs are tech/business) |

### Hebrew instruction / chat data
DictaLM 3.0 SFT data is **not released**. The tech report (https://arxiv.org/abs/2602.02104) lists only checkpoints, and the
dicta-il HF org has no SFT dataset.

| Source | URL | License (Comm?) | Size | Fit |
|---|---|---|---|---|
| OASST2 (he subset) | https://huggingface.co/datasets/OpenAssistant/oasst2 | Apache-2.0 (yes) | small he slice of 135k msgs | L |
| Bactrian-X he (Google-translated Alpaca+Dolly) | https://huggingface.co/datasets/MBZUAI/Bactrian-X | CC BY-NC 4.0 (**no**) | ~67k pairs | L |
| alpaca-hebrew-cleaned (TaCo) | https://huggingface.co/datasets/saillab/alpaca-hebrew-cleaned | not stated | 10k–100k | L |
| hebrew_alpaca_gpt4 | https://huggingface.co/datasets/ashercn97/hebrew_alpaca_gpt4 | not stated (GPT-4 outputs) | 10k–100k | L |
| hebrew-hrm-corpus (our own account) | https://huggingface.co/datasets/guychuk/hebrew-hrm-corpus | Apache-2.0 | see card | L–M |
| AlephBench (Hebrew LLM eval, 11 tasks; eval only, don't train) | https://huggingface.co/datasets/HebArabNlpProject/AlephBench | CC BY 4.0 | 10k–100k | contamination check |

## 5. Hebrew-capable open models (translators / teachers) and EN→HE MT evidence

| Model | URL | License (Comm?) | Notes |
|---|---|---|---|
| Qwen3.5-397B-A17B / 122B-A10B | https://huggingface.co/Qwen/Qwen3.5-397B-A17B , https://huggingface.co/Qwen/Qwen3.5-122B-A10B | Apache-2.0 (yes) | best open EN→HE on the leaderboard below (397B) |
| Gemma 4 31B-it / 26B-A4B-it | https://huggingface.co/google/gemma-4-31B-it | Apache-2.0 (yes) | strong nikud and Hebrew; multimodal |
| DictaLM-3.0-24B-Thinking (Mistral-Small-3.1 base) | https://huggingface.co/dicta-il/DictaLM-3.0-24B-Thinking | Apache-2.0 (yes) | Hebrew-native, 65k ctx; best Summary among open models |
| DictaLM-3.0-Nemotron-12B-Instruct | https://huggingface.co/dicta-il/DictaLM-3.0-Nemotron-12B-Instruct | "other" (NVIDIA model license; verify) | weaker MT (13.5) |
| DictaLM-3.0-1.7B (Qwen3-1.7B base) | https://huggingface.co/dicta-il/DictaLM-3.0-1.7B-Instruct | Apache-2.0 | too weak for MT (2.2) |
| neodictabert-bilingual-embed (HE/EN embeddings for dedup/retrieval) | https://huggingface.co/dicta-il/neodictabert-bilingual-embed | CC BY 4.0 (yes) | source-page retrieval and near-dup filtering |

**EN→HE MT quality evidence.** Hebrew LLM Leaderboard `TranslationPairwise`, raw results:
https://huggingface.co/datasets/hebrew-llm-leaderboard/chat-results (updated 2026-08-31). Protocol (DictaLM 3.0 report, §eval):
the task translates English 20–40-word paragraphs into Hebrew. Each output is compared pairwise with a Gemini-2.5-Pro
translation, GPT-4o judges, and a score of 50 means parity with Gemini-2.5-Pro.

| Model | EN→HE | Model | EN→HE |
|---|---|---|---|
| gemini-3-pro-preview | 51.2 | Qwen3.5-122B-A10B | 31.0 |
| gemini-3-flash-preview | 50.7 | DictaLM-3.0-24B-Thinking | 30.1 |
| gemini-2.5-pro (reference) | 50.5 | gemma-3-27b-it | 26.7 |
| gpt-5.1 | 44.2 | Llama-3.3-70B-Instruct | 19.3 |
| Qwen3.5-397B-A17B | 42.9 | claude-haiku-4.5 | 13.7 |
| claude-sonnet-4.5 | 33.8 | gpt-oss-120b | 10.8 |
| gemma-4-31B-it | 33.3 | Qwen3-30B-A3B-Thinking-2507 | 1.0 |

Caveats: the judge is an LLM, the paragraphs are short and general-domain, and there are no business or legal terms. Our
domain jargon (חשבונית מס, מספר הקצאה, זיכוי, CVE/IOC transliteration) must be spot-checked by a native reviewer.

Human-reference MT test sets for our own EN→HE QA:
- WMT24++ en→he_IL (human refs + post-edits, Apache-2.0): https://huggingface.co/datasets/google/wmt24pp (paper https://arxiv.org/abs/2502.12404)
- MTQE.en-he (959 WMT24++ segments, MT output + human DA scores; ensemble of ChatGPT + CometKiwi + TransQuest beats CometKiwi by 6.4 Pearson): https://aclanthology.org/2026.loreslm-1.48/ (URL pattern; the preview link returned 404)
- FLORES+ heb_Hebr (CC BY-SA 4.0, gated auto): https://huggingface.co/datasets/openlanguagedata/flores_plus

## Gaps / not found
- No open Hebrew call-center or chat transcripts, invoices/receipts, contracts corpus, insurance policy-wording corpus, or
  Hebrew CERT advisory dump.
- Not reached: the gov.il terms page and the INCD alert listing (403), the ITA official allocation API spec, the per-language
  FineWeb-2 doc count, and a confirmed open license for the Technion Supreme Court DB.
