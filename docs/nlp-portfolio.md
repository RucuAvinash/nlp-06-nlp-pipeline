1. NLP Techniques Implemented
Describe the NLP techniques you used.

Tokenization (word, sentence):
The main NLP technique used was Tokenization, removing whitespace based words.The Raw text from the Input file was split into tokens using Python strip() , stripped leading punctuation characters. Short tokens (<= 2 characters) were discarded as they were noise to the data and not used for analyzing.

tokens = text.lower().split()
token.strip(".,:;!?()\\"'")

Frequency analysis (unigram / n-gram):
I also built a Frequency table using Polars to group_by("token").len().sort() to count unigrams across web pages and processing pipelines and corpus categories.
Then created Bigrams, and combined each word with the next word in the text to understand the local structure and arrangement of words.
for doc in corpus:
    tokens = tokenize(doc["text"])
    for i in range (len(tokens)-1):
    bigrams_list.append((tokens[i],tokens[i+1]))

Text cleaning and normalization:
I used sever techniques like lowercase conversion, regex punctuation removal, custom stop word filtering  and filtering based on minimum token length  to pre-process data in the pipeline. The Python pre-built function re.findall was used to match the pattern in a string.
            re.findall(r"\b\w+b",text)

API-based text analysis (and JSON):
Retrieved and processed structured text data from web API's in JSON format. The goal of the project was to acquire JSON data from external source , inspect and validate the structure, transform it into usable format and load it into reproducible output. After extracting fields for analysis, derived categorical fields using conditional logic such as (pl.when().then()) on the executed JSON Values.

   pl.when(pl.col("origin")is in (["natural","natural/standard"])) .then(pl.lit("low-maintanence")).otherwise(pl.lit("high-maintanence"))

Web scraping / content extraction from HTML:
Used some of the web-scraping techniques to work with HTML-data retrieved from web pages using a structured EVTL pipelline.
Fetched HTML from web-pages , then used the BeautifulSoup library to parse raw HTML into structured tree. Using the get_text(separator="",strip=True)extracted the visible content leaving behind the markup.
soup = BeautifulSoup(html_content, "html.parser")

Sentiment analysis (e.g., spaCy + SpacyTextBlob):

spaCy , a industrial strength NLP Library was added to the pipeline for text cleaning and linguistic analysis.Loaded the spaCy English model ("en_core_web_sm")to Tokenize text to individual words, and split them on white spaces.Identify and remove Stop words annotate tokens with part of speech tags and identify named entities such as people, organization.

nlp = spacy.load("en_core_web_sm")

2.  Systems and Data Sources
Describe what you analyzed:

Using Missouri DSS Healthcare at mydss.mo.gov/healthcare/seniors-disabled/ boilerplate and markup language was stripped using the BeautifulSoup library.Using the extracted data frequeency of tokens were analyzed to understand the vocabulary of the page.
On nlp projects 5 and 6 arXiv abstract pages were used and data was scraped and validated against eight required HTML elements(h1, title,div,authors, blockquote.abstract, DOI link,etc).
One of the NLP projects involved in extracting JSON list information from an external REST API of cat breed records. Each record included breed, country, origin,coat and pattern. Using Polar Dataframe, a derived Cat_maintanence_level field was computed from the origin fields.

3. Pipeline Structure (EVTL)
Describe your pipeline using EVTL:

Extract (from source): how data was collected:
The Extract process mainly was used to collect raw data from source.It mainly was a HTTP GET with requests, Browser User-Agent header, API JSON fetch, Local .txt file read.
THe data was extracted and saved in the /data/raw folder.

Validate: structure/content checks performed:
Once the Raw data is extracted from the different sources, the data structure and content was validated across different checks. The Validation process checks for structural elements in the HTML page, If the input was a JSON extract then the pipeline validated the list of dicts, or if the input is a .CSV file then the pipeline raises ValueError on failure.

Transform: NLP processing steps:
The Transform process involved BeautifulSoup field extractions, Parsing date using the Regex function, Derived fields like word count, system stability, rolling mean, rolling SD. Both Polar and Panda library was used to build dataframes .

Load (to sink): outputs (files, summaries, visualizations):
Once the Transformation process was complete, outputs were written to /data/processed folder in a .CSV format, visually represented as Bar charts, Word cloud  and Frequency tables. After running the .py file, a structured log was saved to track the working of the pipeline.

4. Signals and Analysis Methods
Word Frequency: Word frequency was computed across all projects with Polars group_by().len()
Built Frequency dictionaries using zip() to generate the frequent words for the word cloud.
Bigrams: Based on the Frequent used words, Bigrams were created using the consecutive token pairs captured with index arithematic in the corpus explorer.
Co-occurance:Evaluvated co-occurance to see which words tend to appear near each other.The Neighbor list was stored in a defaultdict and inspected for target words like 'cat','dog','car' to reveal semantic neighborhood structure.
Type-TOken Ratio: The Vocabulary richness was measured by dividing unique token count by total token count.

5. Insights
Across these 6 projects, I developed a deep understanding of how NLP , web extraction, and Polar based processing work together to form a reliable, insight driven text analytics pipeline.
Most valuable insight was recognizing the NLP techniques and summarizing and visualizing the large amount of data. I also got a good understanding of Co-Occurance and Bi-grams which helped to see the words that are related  within context. I experimented with different token words  and understood how token relationship shift depending on the anchor term.
Through API based extraction, I gained clarity on how URL's are structured and how JSON responses differ when represented as lists vs dictionaries. I also deepened my knowledge of Polars expression and how they support conditional formatting.
Validating, Spliting and Formatting extracted data:
I implemented several modifications to improve the data quality and structure, like the text-splitting techniques where words are separated by semi-colon, numeric extraction.


6. Representative Work
(https://github.com/RucuAvinash/nlp-03-text-exploration)

The nlp-03-exploration project (Module 3) mainly focuses in performing exploratory analysis of small controlled corpus. I consider this as one of my strongest projects because of the knowledge I gained after completing this module.
This project composes of Tokenization to understanding the frequency distribution and understand the token usage across categories, Once tokens are created , I was able to analyze co -occurance which examines tokens that appear near each other. In this project Bigrams were created to combine each word with the next word in the text, to understand local structure and how words were used together.


nlp-04-api-text-data(Module 4):
https://github.com/RucuAvinash/nlp-04-api-text-data

In this project, I got an understanding of transforming JSON data to a structured format . The goal was to acquire a JSON format data and inspect , validate and transorm it to a usable format and load it to a reproducible output.

7. Skills:

At the end of this course , I am able to build amodular python pipelinewith typed function signatures and structured logging.
I was able to manipulate tabular data with Polars and Pandas and write and apply regex patterns to normalise text.
I am able to create sensible tokens, change the data to lowercase, remoe stopords, and use Polar groupby, and have a understanding of bigram frequency distribution.
I have gained the skills to extract ra data from JSON lists,live websites and parse them using the BeautifulSoup.
