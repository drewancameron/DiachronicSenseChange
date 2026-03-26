"""
Grew pattern rules for Ancient Greek grammar checking.

Each rule is a (name, severity, description, pattern_string) tuple.
Patterns use the Grew DSL — see https://grew.fr/doc/pattern/

These patterns are applied against stanza-parsed CoNLL-U of our
translations and also validated against the Perseus/PROIEL treebanks.
"""

# =================================================================
# AGREEMENT VIOLATIONS
# =================================================================
# NOTE: det-noun and amod-noun agreement rules are demoted to "noisy"
# severity because stanza's AG dependency parser frequently misattaches
# articles to the wrong noun (~7-17% error rate in gold treebanks).
# These are still useful for manual review but produce too many false
# positives for automated pipelines.  The neuter plural rule is reliable
# because it only requires subject-verb attachment, which stanza gets right.

AGREEMENT_RULES = [
    (
        "det_noun_case",
        "noisy",  # ~7% false positive rate in gold treebanks
        "Article-noun case disagreement",
        'pattern { N []; D [upos=DET]; N -[det]-> D } without { N.Case = D.Case }',
    ),
    (
        "det_noun_gender",
        "noisy",  # ~17% false positive rate — stanza misattaches articles
        "Article-noun gender disagreement",
        'pattern { N []; D [upos=DET]; N -[det]-> D } without { N.Gender = D.Gender }',
    ),
    (
        "det_noun_number",
        "noisy",  # ~6% false positive rate
        "Article-noun number disagreement",
        'pattern { N []; D [upos=DET]; N -[det]-> D } without { N.Number = D.Number }',
    ),
    (
        "amod_case",
        "noisy",
        "Adjective-noun case disagreement",
        'pattern { N []; A [upos=ADJ]; N -[amod]-> A } without { N.Case = A.Case }',
    ),
    (
        "amod_gender",
        "noisy",
        "Adjective-noun gender disagreement",
        'pattern { N []; A [upos=ADJ]; N -[amod]-> A } without { N.Gender = A.Gender }',
    ),
    (
        "amod_number",
        "noisy",
        "Adjective-noun number disagreement",
        'pattern { N []; A [upos=ADJ]; N -[amod]-> A } without { N.Number = A.Number }',
    ),
    (
        "neut_pl_plural_verb",
        "warning",  # reliable: stanza handles nsubj well
        "Neuter plural subject + plural verb (Attic rule: should be singular)",
        'pattern { S [Gender=Neut, Number=Plur]; V [upos=VERB, Number=Plur]; V -[nsubj]-> S }',
    ),
]

# =================================================================
# PREPOSITION GOVERNANCE
# =================================================================
GOVERNMENT_RULES = [
    # Single-case prepositions
    (
        "en_dat",
        "warning",
        "ἐν governs dative only",
        'pattern { N [Case<>Dat]; P [lemma="ἐν"]; N -[case]-> P }',
    ),
    (
        "eis_acc",
        "warning",
        "εἰς governs accusative only",
        'pattern { N [Case<>Acc]; P [lemma="εἰς"]; N -[case]-> P }',
    ),
    (
        "ek_gen",
        "warning",
        "ἐκ/ἐξ governs genitive only",
        'pattern { N [Case<>Gen]; P [lemma="ἐκ"]; N -[case]-> P }',
    ),
    (
        "ex_gen",
        "warning",
        "ἐξ governs genitive only",
        'pattern { N [Case<>Gen]; P [lemma="ἐξ"]; N -[case]-> P }',
    ),
    (
        "apo_gen",
        "warning",
        "ἀπό governs genitive only",
        'pattern { N [Case<>Gen]; P [lemma="ἀπό"]; N -[case]-> P }',
    ),
    (
        "pro_gen",
        "warning",
        "πρό governs genitive only",
        'pattern { N [Case<>Gen]; P [lemma="πρό"]; N -[case]-> P }',
    ),
    (
        "syn_dat",
        "warning",
        "σύν governs dative only",
        'pattern { N [Case<>Dat]; P [lemma="σύν"]; N -[case]-> P }',
    ),
    # Multi-case preps: flag only if case is impossible for the prep
    (
        "meta_not_gen_acc",
        "warning",
        "μετά governs genitive or accusative only",
        'pattern { N [Case=Dat]; P [lemma="μετά"]; N -[case]-> P }',
    ),
    (
        "dia_not_gen_acc",
        "warning",
        "διά governs genitive or accusative only",
        'pattern { N [Case=Dat]; P [lemma="διά"]; N -[case]-> P }',
    ),
    (
        "hypo_not_gen_acc",
        "warning",
        "ὑπό governs genitive or accusative only",
        'pattern { N [Case=Dat]; P [lemma="ὑπό"]; N -[case]-> P }',
    ),
    (
        "kata_not_gen_acc",
        "warning",
        "κατά governs genitive or accusative only",
        'pattern { N [Case=Dat]; P [lemma="κατά"]; N -[case]-> P }',
    ),
]

# =================================================================
# CONSTRUCTION DETECTION (informational)
# =================================================================
CONSTRUCTION_RULES = [
    (
        "genitive_absolute",
        "info",
        "Genitive absolute — check matches English source construction",
        'pattern { P [VerbForm=Part, Case=Gen]; N [Case=Gen]; P -[nsubj]-> N }',
    ),
    (
        "articular_infinitive",
        "info",
        "Articular infinitive — check if English had gerund/purpose clause",
        'pattern { V [VerbForm=Inf]; D [upos=DET]; V -[det]-> D }',
    ),
    (
        "acc_inf",
        "info",
        "Accusative + infinitive — check if English had direct/indirect speech",
        'pattern { V [VerbForm=Inf]; S [Case=Acc]; V -[nsubj]-> S }',
    ),
    (
        "relative_clause",
        "info",
        "Relative clause detected",
        'pattern { V []; R [upos=PRON]; V -[nsubj|obj|obl|advmod]-> R; H []; H -[acl:relcl]-> V }',
    ),
    (
        "conditional_ei",
        "info",
        "Conditional clause (εἰ)",
        'pattern { V []; M [lemma="εἰ"]; V -[mark]-> M; H []; H -[advcl]-> V }',
    ),
    (
        "conditional_ean",
        "info",
        "Conditional clause (ἐάν)",
        'pattern { V []; M [lemma="ἐάν"]; V -[mark]-> M; H []; H -[advcl]-> V }',
    ),
    (
        "purpose_hina",
        "info",
        "Purpose clause (ἵνα + subjunctive)",
        'pattern { V [Mood=Sub]; M [lemma="ἵνα"]; V -[mark]-> M }',
    ),
    (
        "result_hoste",
        "info",
        "Result clause (ὥστε)",
        'pattern { V []; M [lemma="ὥστε"]; V -[mark]-> M }',
    ),
    (
        "temporal_hote",
        "info",
        "Temporal clause (ὅτε)",
        'pattern { V []; M [lemma="ὅτε"]; V -[mark]-> M }',
    ),
    (
        "temporal_epei",
        "info",
        "Temporal clause (ἐπεί/ἐπειδή)",
        'pattern { V []; M [lemma=re"ἐπε(ί|ιδή)"]; V -[mark]-> M }',
    ),
]

ALL_RULES = AGREEMENT_RULES + GOVERNMENT_RULES + CONSTRUCTION_RULES
