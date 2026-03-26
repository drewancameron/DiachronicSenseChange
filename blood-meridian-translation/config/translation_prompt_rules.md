# Translation Prompt Rules

These rules MUST be included in every translation prompt (for new passages or revision passes).

## Sentence Structure Mirroring

**Mirror the syntactic construction type of the English source sentence:**

- English relative clause ("the creature **who would** carry her off") → Greek relative clause (τὸ πλάσμα **ὃ** αὐτὴν ἀπάξει), NOT articular participle
- English participial phrase ("crouching by the fire") → Greek participle (πτώσσων παρὰ τὸ πῦρ), NOT relative clause
- English bare fragment ("The firewood, the washpots.") → Greek bare fragment (τὰ ξύλα, τοὺς πλυνούς.), NOT a full sentence
- English coordination ("he walks and he hears") → Greek coordination (περιπατεῖ καὶ ἀκούει), NOT subordination
- English conditional ("if they got my boots") → Greek conditional (εἰ ἔλαβον τὰς ἐμβάδας), NOT participial
- English direct speech → Greek direct speech with same structure

**Do NOT convert between construction types unless the alternative is more natural for Ancient Greek (Koine/Attic).** The LLM default is to over-subordinate and over-participialise. Resist this — but where Greek idiom genuinely favours a different construction (e.g. genitive absolute for an English temporal clause), that is fine. The test: would a competent Greek prose author use this construction here? If the English structure works naturally in Greek too, keep it.

## Neuter Plural Agreement

Neuter plural subjects take SINGULAR verbs (Attic rule, maintained in our register):
- τὰ πλοῖα βοᾷ (not βοῶσι)
- τὰ Λεοντίδια ἐκαλεῖτο (not ἐκαλοῦντο)
- τὰ ὀνόματα ἀπόλωλεν (not ἀπολώλασιν)

## Time Expressions

- Dative for point in time ("on that night" = νυκτὶ ἐκείνῃ)
- Accusative for duration ("fourteen years" = τεσσαρεσκαίδεκα ἔτη)
- Genitive for time within which ("by night" = νυκτός)

## Word Order

Place time expressions and adverbial modifiers where the English places them relative to the word they modify. If McCarthy puts the time expression between subject and participle, do the same in Greek:
- "The mother dead these fourteen years did incubate..."
- → ἡ μήτηρ τεσσαρεσκαίδεκα ταῦτα ἔτη τεθνηκυῖα ἐνεθάλπετο...

## Vocabulary

- Use the IDF glossary for all locked terms
- Prefer πλάσμα over κτίσμα for "creature" (more physical, Genesis resonance)
- Mark all loans/transliterations with * prefix
- Words attested only in Byzantine or later Greek (e.g. *βαμβάκιον) must also be * marked as loans
- Herodotean periphrasis is acceptable for concepts without a Koine word (e.g. cotton = τὸ ἐριοφόρον φυτόν, Hdt. 3.106), but a short marked loan is also fine for passing references
- Every word in the translation should be attestable in Morpheus/LSJ or our AG corpus. If not, it is probably invented. Replace it.
- Do not use modern Greek vocabulary. All prompts must specify "Ancient Greek (Koine/Attic)" — never just "Greek" which risks modern Greek output.

## Register

- Koine structure with Attic vocabulary richness
- Septuagintal colouring for biblical passages
- Do not over-subordinate — McCarthy's power comes from coordination
