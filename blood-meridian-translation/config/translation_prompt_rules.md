# Translation Prompt Rules

These rules MUST be included in every translation prompt (for new passages or revision passes).

## Sentence Structure Guidance

**Prefer the most natural Greek idiom, but be alert to these common LLM pitfalls:**

The LLM default when translating into Greek is to over-subordinate and over-participialise — collapsing McCarthy's distinctive parataxis into nested subordinate clauses. Guard against this specifically:

- English coordination ("he walks and he hears") → preserve as coordination (περιπατεῖ καὶ ἀκούει), NOT subordination
- English bare fragment ("The firewood, the washpots.") → preserve as fragment (τὰ ξύλα, τοὺς πλυνούς.), NOT a full sentence
- English "and...and...and" chains → preserve as καί chains. This is McCarthy's signature.
- English direct speech → Greek direct speech with same structure

For other constructions, use whatever is most natural in Greek:
- English relative clause → Greek relative clause OR participle, whichever a competent Greek prose author would prefer
- English temporal clause → Greek temporal clause OR genitive absolute, as idiom dictates
- English conditional → Greek conditional (the form should follow Smyth's categories naturally)

The test: does this sound like something Thucydides, Xenophon, or the Septuagint translators would write? If so, it's fine — even if the construction type differs from the English.

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
