# Glossary (Module 6: NLP Pipeline)

See earlier glossaries for additional terms.

## EVTAL (Extract, Validate, Transform, Analyze, Load)

An extension of the EVTL pipeline that introduces an explicit **Analyze** stage:

- **Extract**: fetch HTML from a web page and save it to a file
- **Validate**: parse the HTML and confirm expected structure is present
- **Transform**: an iterative loop: extract fields, clean and normalize text,
  engineer derived features; repeat until the data is analysis-ready
- **Analyze**: compute frequency distributions and produce visualizations
  that surface patterns in the text
- **Load**: write the data to a destination

## Text Cleaning (Process)

The process of removing noise from raw text to improve the quality
of downstream analysis.
Common cleaning steps include lowercasing, removing punctuation,
normalizing whitespace, and removing stopwords.
Each cleaning step involves a tradeoff as _signal_ may be lost alongside _noise_.

## Normalization (Process)

Converting text to a consistent form.
In this module, normalization includes lowercasing and whitespace reduction
so that "The" and "the" are treated as the same token.

## Stopword

A common word that carries little semantic signal for analysis purposes,
such as "the", "a", "is", "of".
Removing stopwords reduces noise before frequency analysis.
spaCy provides a built-in stopword list for English.
Note: some stopwords matter in certain contexts (e.g. "not").
Removal is a **tradeoff**, not a universal rule.

## Whitespace

Characters used to separate text elements without visible marks.
Common whitespace characters include **spaces, tabs, and line breaks (newlines)**.
In text processing, whitespace is often used as a simple delimiter
to split text into tokens (e.g., words).

## Tokenization (Process)

The process of splitting text into individual units called **tokens**.

In this module, we use **whitespace tokenization (simplified)**,
a rule-based approach where tokens are individual words
produced by splitting on whitespace (e.g., the spaces between words).
For example, the sentence "NLP is fun" contains three tokens
when split on whitespace: ["NLP", "is", "fun"].

This approach is fast and easy to implement but does not account for
punctuation, contractions, or language-specific rules.

More sophisticated approaches use **linguistic tokenization**,
as implemented in libraries like spaCy.
Linguistic tokenizers apply language-aware rules and
pre-trained statistical models
to correctly handle punctuation, contractions
(e.g., "don't" splits into "do" and "n't"),
and hyphenated or compound words.

Linguistic tokenization produces tokens that more accurately
reflect the structure and meaning of natural language.

## Token

A single unit of text produced by tokenization.
In this module, tokens are individual words produced by splitting on whitespace.
In LLM processing, tokens are often **subword units**
(parts of words or punctuation) depending on the tokenizer used.

## Token Count

The number of tokens in a text at a given stage of the pipeline.
In this module, token count is measured after cleaning and stopword removal.
A lower token count than the raw word count indicates that
stopwords and noise were successfully removed.

## Unique Token Count

The number of distinct tokens in a text.
Comparing unique token count to total token count reveals
how much vocabulary repetition exists.

## Type-Token Ratio (TTR)

A measure of vocabulary richness calculated as:

```
TTR = unique token count / total token count
```

A TTR close to 1.0 means almost every word is unique (high diversity).
Lower TTR values (closer to 0.0) indicate more repetition (lower vocabulary diversity).
Short texts tend to have higher TTRs than long texts.

## Frequency Distribution

A count of how often each token appears in a text.
Frequency distributions reveal which words dominate the content
and whether cleaning was effective.
In this module, `collections.Counter` is used as the implementation
mechanism to compute frequency distributions.

## collections.Counter (Python library class)

A Python standard library class that counts the occurrences of items
in a list or other iterable.
`.most_common(n)` returns the n most frequent items as a list of
`(item, count)` tuples.

Example:

```python
from collections import Counter
Counter(["cat", "dog", "cat"]).most_common(2)
# returns [("cat", 2), ("dog", 1)]
```

## spaCy (Python library)

An industrial-strength Python NLP library used in this project
for tokenization, stopword removal, part-of-speech (POS) tagging,
and named entity recognition (NER).
spaCy is fast, well-documented, and widely used in production NLP systems.

## spaCy Language Model (pre-trained data model)

A pre-trained model that spaCy uses to perform linguistic analysis.
This project uses `en_core_web_sm`, the small English model.
It provides tokenization, stopword lists, POS tags, and basic NER.
Larger models (`en_core_web_md`, `en_core_web_lg`) add word vectors
for similarity comparisons.

## Part-of-Speech (POS) Tagging

The process of labeling each token with its grammatical role:
noun, verb, adjective, adverb, etc.
spaCy assigns POS tags automatically when processing text.
Access via `token.pos_` on any token in a spaCy Doc object.

## Named Entity Recognition (NER)

The process of identifying and classifying named things in text:
people, organizations, locations, dates, products.
spaCy performs NER automatically when processing text.
Access via `doc.ents` on a spaCy Doc object.

## spaCy Doc (object)

The object returned by `nlp(text)` - a processed text object
containing tokens, POS tags, named entities, and other annotations.
Iterating over a Doc yields individual `Token` objects.

## Matplotlib (Python library)

A Python library for creating static visualizations.
Used in this module to produce bar charts of token frequencies.
`plt.savefig()` saves a figure to a file without displaying it,
which is required for pipeline execution without a display.

## WordCloud (Python library)

A Python library that generates word cloud images from text.
Word size reflects frequency - more frequent words appear larger.
Useful for a quick gestalt view of text content.
Less precise than a bar chart but easier to read at a glance.

## Bar Chart (Visualization)

A chart that uses horizontal or vertical bars to show
the frequency or magnitude of categorical values.
In this module, a horizontal bar chart shows the top N tokens
by frequency, making it easy to compare relative dominance.

## Word Cloud (Visualization)

A visual representation of text where word size reflects frequency.
Produced by the `wordcloud` library.
Useful for immediate topic recognition but less precise than a bar chart.

## Static Embedding

A fixed numeric vector representation of a word learned from
a large text corpus.
Words with similar meanings are geometrically close in vector space.
spaCy's `en_core_web_md` and `en_core_web_lg` models include static embeddings.
Called "static" because the vector does not change based on context -
"bank" in "river bank" and "bank account" gets the same vector.

## Attention Mechanism

The **core innovation** of **transformer-based large language models (LLMs)**.
For each word in a sentence, attention computes
**which other words matter most** for understanding it in context.
Specifically, it computes **weighted relationships between tokens**
to determine context relevance.
The same word gets a different representation depending on
the full sentence it appears in.
This is how LLMs resolve ambiguity and understand meaning contextually,
unlike static embeddings or n-gram models.
See: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Large Language Model (LLM)

A **neural network** trained on large text corpora using **transformer architecture**.
LLMs generate contextual representations of text using attention mechanisms.
They do not require explicit preprocessing steps such as
stopword removal, POS taggers, or static word vectors,
but they work best when given clean, well-structured input.
Preprocessing pipelines like this one prepare data for LLM consumption.

## Neural Network (NN)

A computational model loosely inspired by the structure of the human brain,
consisting of layers of interconnected nodes (neurons) that learn
to recognize patterns from data.

During training, the network adjusts the strength of connections
(called weights) to minimize the difference between its predictions
and the correct answers.

Neural networks are the foundation of modern NLP:

- early NLP models used shallow networks with few layers
- **deep neural networks** (many layers) enabled word vectors and embeddings
- **transformer-based LLMs are very large deep neural networks**
  trained on billions of text examples

You do not need to understand the mathematics to use neural network tools,
but understanding that they _learn from data_ automatically rather than following
hand-written rules or specific instructions is essential for
interpreting their outputs and knowing when their output is likely to be reliable.
