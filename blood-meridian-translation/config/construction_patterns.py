"""
Grew patterns for detecting named Greek constructions in UD parses.

Each pattern returns matches from a CoNLL-U corpus. The pattern name
maps to an entry in construction_taxonomy.yaml.

These patterns work on both UD_Ancient_Greek-Perseus and UD_Ancient_Greek-PROIEL.
"""

# Format: (name, grew_pattern_string)
# Patterns use Grew DSL: https://grew.fr/doc/pattern/

CONSTRUCTION_PATTERNS = [

    # === CASE CONSTRUCTIONS ===

    ("Genitive Absolute",
     # Broad: any genitive participle near a genitive noun — either direction of dependency
     'pattern { P [VerbForm=Part, Case=Gen]; S [Case=Gen, upos=NOUN|PROPN|PRON] }'),

    ("Dative of Time When",
     'pattern { N [Case=Dat, upos=NOUN]; V [upos=VERB]; V -[obl:tmod|obl]-> N }'
     # Note: obl:tmod is rare in AG treebanks; most use plain obl
     ),

    ("Accusative of Duration",
     'pattern { N [Case=Acc, upos=NOUN]; V [upos=VERB]; V -[obl:tmod|obl]-> N; Q [upos=NUM]; N -[nummod]-> Q }'),

    ("Dative of Instrument",
     'pattern { N [Case=Dat]; V [upos=VERB]; V -[obl]-> N }'
     # Broad — will need post-filtering to distinguish from dative of time, etc.
     ),

    ("Genitive of Separation",
     'pattern { A [upos=ADJ]; N [Case=Gen]; A -[obl|nmod]-> N }'),

    # === INFINITIVE CONSTRUCTIONS ===

    ("Articular Infinitive",
     'pattern { V [VerbForm=Inf]; D [upos=DET]; V -[det]-> D }'),

    ("Accusative + Infinitive",
     'pattern { V [VerbForm=Inf]; S [Case=Acc]; V -[nsubj]-> S }'),

    ("Purpose Infinitive",
     'pattern { V [VerbForm=Inf]; H [upos=VERB]; H -[advcl|xcomp]-> V }'),

    # === PURPOSE AND RESULT ===

    ("Purpose Clause (ἵνα)",
     'pattern { V [Mood=Sub]; M [lemma="ἵνα"]; V -[mark]-> M }'),

    ("Purpose Clause (ὅπως)",
     'pattern { V []; M [lemma="ὅπως"]; V -[mark]-> M }'),

    ("Result Clause (ὥστε)",
     'pattern { V []; M [lemma="ὥστε"]; V -[mark]-> M }'),

    # === CONDITIONALS ===

    ("Conditional (εἰ + indicative)",
     'pattern { V [Mood=Ind]; M [lemma="εἰ"]; V -[mark]-> M; H []; H -[advcl]-> V }'),

    ("Conditional (ἐάν + subjunctive)",
     'pattern { V [Mood=Sub]; M [lemma="ἐάν"]; V -[mark]-> M; H []; H -[advcl]-> V }'),

    ("Conditional (εἰ + optative)",
     'pattern { V [Mood=Opt]; M [lemma="εἰ"]; V -[mark]-> M; H []; H -[advcl]-> V }'),

    # === TEMPORAL ===

    ("Temporal (ὅτε + indicative)",
     'pattern { V [Mood=Ind]; M [lemma="ὅτε"]; V -[mark]-> M }'),

    ("Temporal (ἐπεί/ἐπειδή)",
     'pattern { V []; M [lemma="ἐπεί"]; V -[mark]-> M }'),

    ("Temporal (πρίν)",
     'pattern { V []; M [lemma="πρίν"]; V -[mark]-> M }'),

    ("Temporal (ἕως)",
     'pattern { V []; M [lemma="ἕως"]; V -[mark]-> M }'),

    # === RELATIVE CLAUSES ===

    ("Relative Clause (defining)",
     'pattern { V []; R [upos=PRON, Case=Nom]; V -[nsubj]-> R; H []; H -[acl:relcl|acl]-> V }'),

    ("Relative Clause (oblique)",
     'pattern { V []; R [upos=PRON]; V -[obj|obl|nsubj]-> R; H []; H -[acl:relcl|acl]-> V }'),

    # === PARTICIPLES ===

    ("Circumstantial Participle",
     'pattern { P [VerbForm=Part]; V [upos=VERB, VerbForm=Fin]; V -[advcl]-> P }'),

    ("Attributive Participle",
     'pattern { P [VerbForm=Part]; D [upos=DET]; N [upos=NOUN]; N -[acl]-> P; N -[det]-> D }'),

    # === VOICE ===

    ("Middle Voice",
     'pattern { V [upos=VERB, Voice=Mid] }'),

    ("Passive Voice",
     'pattern { V [upos=VERB, Voice=Pass] }'),

    # === INDIRECT SPEECH ===

    ("Indirect Speech (ὅτι + indicative)",
     'pattern { V [Mood=Ind]; M [lemma="ὅτι"]; V -[mark]-> M; H [upos=VERB]; H -[ccomp]-> V }'),

    ("Indirect Speech (acc + infinitive)",
     # Same as Accusative + Infinitive but under a speech verb
     'pattern { V [VerbForm=Inf]; S [Case=Acc]; V -[nsubj]-> S; H [upos=VERB]; H -[ccomp|xcomp]-> V }'),
]

# Only label when the construction genuinely helps a translation decision.
# Skip labels that are too common to be informative (middle voice, circumstantial
# participle) or too broad to be reliable (dative of instrument).
#
# A label earns its place if: knowing it would change what Greek the LLM produces.
HIGH_VALUE_PATTERNS = [
    # These change the sentence structure:
    "Genitive Absolute",           # EN temporal clause → GRC gen.abs. (key decision)
    "Articular Infinitive",        # EN gerund → GRC τό + inf (unusual, worth flagging)
    "Accusative + Infinitive",     # EN indirect speech → GRC acc+inf vs ὅτι
    "Relative Clause (defining)",  # confirms relative preserved (not converted to participle)
    "Relative Clause (oblique)",   # same
    "Attributive Participle",      # EN relative → GRC participle (the conversion we watch for)
    # These change the clause type:
    "Conditional (εἰ + indicative)",   # specific conditional type
    "Conditional (ἐάν + subjunctive)", # specific conditional type
    "Conditional (εἰ + optative)",     # specific conditional type
    "Purpose Clause (ἵνα)",        # purpose construction choice
    "Result Clause (ὥστε)",        # result construction choice
    # These are only labelled on the English side (predicting what Greek to use):
    # "Oath-Conditional", "Modal/Future Relative", etc. — handled by conditional_guide.py
]
