"""
The Macro-Etymological Analyzer.
Author: Jonathan Reeve, jonathan@jonreeve.com
License: GPLv3

This program looks up all the words in your text, using the Etymological
Wordnet.

I made the file included here, etymwn-smaller.tsv, by running these unix
commands on the Etymological Wordnet:

First, get only those entries with the relation "rel:etymology": grep
    "rel:etymology" etymwn.tsv > etymwn-small.tsv Now we can remove the relation
    column, since it's all "rel:etymology": cat etymwn-small.tsv | cut -f1,3 >
    etymwn-smaller.tsv
"""

from collections import Counter
from string import punctuation
import codecs
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
# from nltk.tokenize import RegexpTokenizer
from pkg_resources import resource_filename
from pycountry import languages
import click
import csv
import logging
# import matplotlib
import pandas as pd

# Parse the CSV file.
etymdict = {}
etymwn = resource_filename(__name__, 'etymwn-smaller.tsv')
with open(etymwn) as csvfile:
    csvreader = csv.reader(csvfile, delimiter='\t')
    for line in csvreader:
        if line[0] in etymdict:
            etymdict[line[0]].append(line[1])
        else:
            etymdict[line[0]] = [line[1]]

class LangList():
    """
    A class for language lists, that helps to count languages.
    """
    def __init__(self, langs):
        self.langs = langs

    def __repr__(self):
        return str(self.langs)

    @property
    def stats(self):
        """ Generates statistics about languages present in the list. """
        counter = Counter(self.langs)
        stats = {}
        for lang in counter.keys():
            stats[lang] = (counter[lang] / len(self.langs))*100
        return stats


class Word():
    """
    A word object, for looking up etymologies of single words.
    """
    def __init__(self, word, lang='eng', ignoreAffixes=True, ignoreCurrent=True):
        self.lang = lang
        self.word = word
        self.ignoreAffixes = ignoreAffixes
        self.ignoreCurrent = ignoreCurrent

    def __repr__(self):
        return '%s (%s)' % (self.word, self.lang)

    def __str__(self):
        return self.word

    @staticmethod
    def old_versions(language):
        """
        Returns a list of older versions of a language, such that given "eng"
        (Modern English) it will return "enm" (Middle English). This is used
        for filtering out current languages in the ignoreCurrent option of
        parents() below.
        """
        lang_map = {'eng': ['enm'],
                    'fra': ['frm', 'xno'],
                    'dut': ['dum'],
                    'gle': ['mga']}
        # TODO: add other languages here.
        return lang_map.get(language, [])

    @property
    def parents(self):
        """ A wrapper for getParents"""
        return self.getParents()

    def getParents(self, level=0):
        """
        The main etymological lookup method.

        ignoreAffixes will remove suffixes like -ly, so that the parent list
        for "universally" returns "universal (eng)" instead of "universal
        (eng), -ly (eng)."

        ignoreCurrent will ignore etymologies in the current language and
        slightly older versions of that language, so that it skips "universal
        (eng)," and goes straight to the good stuff, i.e. "universalis (lat)."
        Given a word in English, it will skip all other English and Middle
        English ancestors, but won't skip Old English.
        """
        language = self.lang

        # Finds the first-generation ancestor(s) of a word.
        try:
            raw_parent_list = etymdict[language + ": " + self.word]
        except KeyError:
            logging.debug('No etymology found for word %s in language %s' % (self.word, language))
            raw_parent_list = []
        parent_list = [self.split(parent) for parent in raw_parent_list]
        if self.ignoreAffixes:
            parent_list = [p for p in parent_list if p.word[0] != '-']
            parent_list = [p for p in parent_list if p.word[-1] != '-']
        if self.ignoreCurrent:
            newParents = []
            for parent in parent_list:
                if (parent.lang == language or parent.lang in self.old_versions(language)) and level < 3:
                    logging.debug('Searching deeper for word %s with lang %s' % (parent.word, parent.lang))
                    for otherParent in parent.getParents(level=level+1):
                        # Go deeper.
                        newParents.append(otherParent)
                else:
                    newParents.append(parent)
            parent_list = newParents
        return parent_list

    @property
    def parent_languages(self):
        """ Get the parent languages of a word."""
        parent_langs = []
        for parent in self.parents:
            parent_langs.append(parent.lang)
        return LangList(parent_langs)

    @property
    def grandparents(self):
        """ Get the grandparent words/languages of a word."""
        return [Word(parent.word, lang=parent.lang).parents
        for parent in self.parents]

    @property
    def grandparent_languages(self):
        """ Get the grandparent languages for a word. """
        grandparent_langs = []
        for grandparent_list in self.grandparents:
            for grandparent in grandparent_list:
                grandparent_langs.append(grandparent.lang)
        return LangList(grandparent_langs)

    @staticmethod
    def split(expression):
        """ Takes and expression in the form 'enm: not' and returns
        a Word object where word.lang is 'enm' and word.word is 'not'.
        """
        parts = expression.split(':')
        return Word(parts[1].strip(), parts[0])

class Text():
    GERMANIC = 'Germanic'
    LATINATE = 'Latinate'
    INDOIRANIAN = 'Indo-Iranian'
    CELTIC = 'Celtic'
    HELLENIC = 'Hellenic'
    SEMITIC = 'Semitic'
    TURKIC = 'Turkic'
    AUSTRONESIAN = 'Austronesian'
    BALTOSLAVIC = 'Balto-Slavic'
    URALIC = 'Uralic'
    JAPONIC = 'Japonic'

    LANGUAGE_FAMILIES = [
        GERMANIC,
        LATINATE,
        INDOIRANIAN,
        CELTIC,
        HELLENIC,
        SEMITIC,
        TURKIC,
        AUSTRONESIAN,
        BALTOSLAVIC,
        URALIC,
        JAPONIC,
    ]

    LANGUAGE_MAP = {
        'eng': GERMANIC,
        'enm': GERMANIC,
        'ang': GERMANIC,
        'deu': GERMANIC,
        'dut': GERMANIC,
        'nld': GERMANIC,
        'dum': GERMANIC,
        'non': GERMANIC,
        'gml': GERMANIC,
        'yid': GERMANIC,
        'swe': GERMANIC,
        'rme': GERMANIC,
        'sco': GERMANIC,
        'isl': GERMANIC,
        'dan': GERMANIC,
        'fra': LATINATE,
        'frm': LATINATE,
        'fro': LATINATE,
        'lat': LATINATE,
        'sap': LATINATE,
        'xno': LATINATE,
        'por': LATINATE,
        'ita': LATINATE,
        'hin': INDOIRANIAN,
        'fas': INDOIRANIAN,
        'gle': CELTIC,
        'gla': CELTIC,
        'grc': HELLENIC,
        'ara': SEMITIC,
        'heb': SEMITIC,
        'tur': TURKIC,
        'tgl': AUSTRONESIAN,
        'mri': AUSTRONESIAN,
        'smo': AUSTRONESIAN,
        'rus': BALTOSLAVIC,
        'fin': URALIC,
        'hun': URALIC,
        'jpn': JAPONIC,
    }

    IGNORED = 'ignored'

    def __init__(self, text, lang='eng', ignoreAffixes=True, ignoreCurrent=True):
        self._stopwords = None
        self.text = text
        self.lang = lang
        self.lemmatizer = WordNetLemmatizer()
        self._word_objects = []
        self.annotated_text = {}
        ignoreAffixes = ignoreAffixes
        ignoreCurrent = ignoreCurrent
        tokenized_text = word_tokenize(text)
        lower_tokens = [w.lower() for w in tokenized_text]
        valid_words = [word for word in lower_tokens if self.valid_word(word)]
        pos_words = {x[0]: x for x in pos_tag(set(valid_words))}
        for word in tokenized_text:
            lower_word = word.lower()
            if lower_word in pos_words:
                lemma = self.lemmatize(pos_words[lower_word]).lower()
                word_object = Word(
                    lemma,
                    lang,
                    ignoreAffixes=ignoreAffixes,
                    ignoreCurrent=ignoreCurrent
                )
                self._word_objects.append(word_object)
                self.annotated_text[word] = {
                    'word': word_object,
                    'lemma': lemma
                }
            else:
                self.annotated_text[word] = self.IGNORED

    def valid_word(self, word):
        return word not in punctuation and word.isalpha() and word not in self.stopwords

    def annotate(self):
        """ Returns an annotated text in HTML format. """
        return HTMLify(self).html

    @property
    def stopwords(self):
        if self._stopwords is None:
            self._stopwords = self._get_stopwords()
        return self._stopwords

    def _get_stopwords(self):
        stop_dict = {
            'dan': 'danish',
            'eng': 'english',
            'fra': 'french',
            'hun': 'hungarian',
            'nor': 'norwegian',
            'spa': 'spanish',
            'tur': 'turkish',
            'dut': 'dutch',
            'fin': 'finnish',
            'deu': 'german',
            'ita': 'italian',
            'por': 'portuguese',
            'rus': 'russian',
            'swe': 'swedish',
            'ger': 'german',
            'fre': 'french',
        }
        if self.lang in stop_dict:
            return set(stopwords.words(stop_dict[self.lang]))
        return set()

    def _wordnet_pos(self, treebank_tag):
        """
        Translate between treebank tag style and WordNet tag style.

        Here, we map the treebank tag to the wordnet tag by taking the
        first letter of the treebank tag and mapping it to the wordnet tag.

        Upenn Treebank part-of-speech tags are used by the nltk pos tagger.
        The possible tags are ennumerated by nltk.help.upenn_tagset().

        - Nouns, e.g., are tagged as 'NN', 'NNS', 'NNP', 'NNPS'.
        - Verbs, e.g., are tagged as 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'.
        - Adjectives, e.g., are tagged as 'JJ', 'JJR', 'JJS'.
        - Adverbs, e.g., are tagged as 'RB', 'RBR', 'RBS'.

        Wordnet uses a different part-of-speech tagset.

        - Nouns are 'n'
        - verbs are 'v'
        - adjectives are 'a'
        - adverbs are 'r'.

        If the treebank tag is not in the map, we default to 'n' (noun).
        """
        treebank_tag = treebank_tag[0]
        tag_map = {"J": wn.ADJ,
                   "V": wn.VERB,
                   "N": wn.NOUN,
                   "R": wn.ADV}
        return tag_map.get(treebank_tag, 'n')

    def lemmatize(self, wordpos):
        word, pos = wordpos
        return self.lemmatizer.lemmatize(word, self._wordnet_pos(pos))

    def _get_stats(self):
        if not hasattr(self, '_stats'):
            stats_list = [word.parent_languages.stats for word in self._word_objects]
            stats = {}
            for item in stats_list:
                if len(item) > 0:
                    for lang, perc in item.items():
                        if lang not in stats:
                            stats[lang] = perc
                        else:
                            stats[lang] += perc
            allPercs = sum(stats.values())
            for lang, perc in stats.items():
                stats[lang] = ( perc / allPercs ) * 100
            self._stats = stats
        return self._stats

    def _get_pretty_stats(self):
        stats = self._get_stats()
        pretty_stats = {}
        for lang, perc in stats.items():
            try:
                pretty_lang = languages.get(alpha_3=lang).name
            except:
                pretty_lang = "Other Language"
            pretty_stats[pretty_lang] = round(perc, 2)
        return pretty_stats

    def _get_family(self, language):
        return self.LANGUAGE_MAP.get(language, 'Other')

    def get_family_stats(self):
        stats = self._get_stats()
        families = {}
        for lang, perc in stats.items():
            fam = self._get_family(lang)
            #print( fam, lang, perc) #debugging
            if fam in families:
                families[fam].append((lang, perc))
            else:
                families[fam] = [(lang, perc)]
        return families

    def compile_family_stats(self, pad=True):
        families = self.get_family_stats()
        totals = {}
        for family, langs in families.items():
            totals[family] = 0
            for lang in langs:
                totals[family] += lang[1]
        # optionally add language families not represented by the text
        if pad:
            for fam in self.LANGUAGE_FAMILIES:
                if fam not in totals:
                    totals[fam] = 0.0
        return totals

    @property
    def stats(self):
        return self._get_stats()

    def family_stats(self, pad=True):
        return self.compile_family_stats(pad)

    @property
    def pretty_stats(self):
        if not hasattr(self, '_pretty_stats'):
            self._pretty_stats = self._get_pretty_stats()
        return self._pretty_stats

    def print_pretty_stats(self, filename):
        d = {filename: self.pretty_stats}
        df = pd.DataFrame(d)
        print(df)

    def print_CSV_stats(self, filename):
        d = {filename: self.pretty_stats}
        df = pd.DataFrame(d)
        print(df.to_csv())


class HTMLify():
    HEADER = """
            <html>
            <head>
            <meta charset="utf-8">
            <style>
            body {
                font-family: sans-serif;
                color: black;
                margin: 2em auto;
                max-width: 72em;
            }
            .Germanic { color: green; }
            .Latinate { color: red; }
            .Indo-Iranian { color: blue; }
            .Celtic { color: purple; }
            .Hellenic { color: orange; }
            .Semitic { color: yellow; }
            .Turkic { color: pink; }
            .Austronesian { color: brown; }
            .Balto-Slavic { color: teal; }
            .Uralic { color: gray; }
            .Japonic { color: pink; }
            </style>
            </head>
            <body>
            <p>Key</p>
            <ul>
            <li><span class="Germanic">Green</span> = Germanic</li>
            <li><span class="Latinate">Red</span> = Latinate</li>
            <li><span class="Indo-Iranian">Blue</span> = Indo-Iranian</li>
            <li><span class="Celtic">Purple</span> = Celtic</li>
            <li><span class="Hellenic">Orange</span> = Hellenic</li>
            <li><span class="Semitic">Yellow</span> = Semitic</li>
            <li><span class="Turkic">Pink</span> = Turkic</li>
            <li><span class="Austronesian">Brown</span> = Austronesian</li>
            <li><span class="Balto-Slavic">Teal</span> = Balto-Slavic</li>
            <li><span class="Uralic">Gray</span> = Uralic</li>
            <li><span class="Japonic">Pink</span> = Japonic</li>
            <li>Black = Unknown</li>
            </ul>
            <p>Text</p>
    """
    FOOTER = "</body></html>"

    def __init__(self, text):
        self.text = text
        html_body = self._html_body()
        self.html = self.HEADER + html_body + self.FOOTER

    def _html_body(self):
        _ = self.text._get_stats()
        html = ""
        for line in self.text.text.split("\n"):
            if not line.strip():
                html += "<br>"
            for word in word_tokenize(line):
                word = word.lower()
                if word in self.text.annotated_text:
                    parents = self.text.annotated_text[word]
                    if parents == self.text.IGNORED:
                        html += f"<span>{word}</span> "
                    else:
                        root = None
                        if parents['word'].parents:
                            parent = parents['word'].parents[0]
                            root = self.text.LANGUAGE_MAP.get(parent.lang)
                        if root:
                            html += f"<span class={root}>{word}</span> "
                        else:
                            html += f"<span>{word}</span> "
                else:
                    html += f"<span>{word}</span> "
            html += "<br>"
        return html



@click.command()
@click.argument('filenames', nargs=-1, required=True)
@click.option('--allstats', is_flag=True,
        help="Get all etymological statistics about the file(s).")
@click.option('--lang', default='eng',
        help="Specify the language of the texts. Use ISO639-3 "\
             "three-letter language code. Default is English.")
@click.option('--show-families', help="A comma-separated list of language "\
              "families to show, e.g. Latinate,Germanic")
@click.option('--affixes', is_flag=True, help="Don't ignore affixes. "\
              "Default is to ignore them.")
@click.option('--current', is_flag=True, help="Don't ignore current language "\
              "and its middle variants. Default is to ignore them.")
@click.option('-c', '--csv', is_flag=True, help="Print a machine-readable "
              "CSV instead of a pretty table.")
@click.option('--chart', is_flag=True, help="Make a pretty graph of the "\
              "results. For one text, a pie; for multiple, a bar.")
@click.option('--annotate', is_flag=True, help="Annotate the text with etymological information.")
@click.option('--verbose', is_flag=True, help="Show debugging messages.")
def cli(filenames, allstats, lang, show_families, affixes,
        current, csv, chart, annotate, verbose):
    """
    Analyzes a text(s) for the etymologies of its words, and tallies the words
    by origin language, and origin language family.
    """
    single = len(filenames) == 1
    ignoreAffixes = not affixes
    ignoreCurrent = not current
    cumulativeStats = {}
    cumulativeAllStats = {}

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    for filename in filenames:
        try:
            with open(filename) as fdata:
                text = fdata.read()
        except UnicodeDecodeError as e:
            logging.error(f"Can't open file {filename} in the normal way, using utf-8.")
            logging.error(f"I'll try opening using other encodings, but you may want to Try converting it to utf-8..")
            logging.error(f"Here's the error: {e}")
            try:
                with open(filename, encoding='utf-16') as fdata:
                    text = fdata.read()
            except UnicodeDecodeError as e:
                logging.error(f"Can't open file {filename} as UTF-16, either. Try converting it to utf-8. Error: {e}")
                try:
                    with open(filename, encoding='latin1') as fdata:
                        text = fdata.read()
                except UnicodeDecodeError as e:
                    logging.error(f"Can't open file {filename} as latin1, either. Try converting it to utf-8. Error: {e}")

        t = Text(text, lang, ignoreAffixes, ignoreCurrent)

        if annotate:
            print(t.annotate())
            return

        if single and allstats and csv:
            t.print_CSV_stats(filename)
        elif single and allstats:
            t.print_pretty_stats(filename)

        if chart and single:
            s = pd.Series(t.family_stats())
            if show_families:
                famlist = show_families.split(',')
                s = s.loc[famlist]
            ax = s.plot(kind='pie', figsize=(6,6))
            ax.set_ylabel('') # Don't write the series name "None" on the left.
            fig = ax.get_figure()
            fig.savefig('chart.png')
            print('Chart saved as chart.png.')

        cumulativeStats[filename] = t.family_stats(pad=single)
        cumulativeAllStats[filename] = t.pretty_stats

    df = pd.DataFrame(cumulativeStats)
    df = df.fillna(0)

    dfAll = pd.DataFrame(cumulativeAllStats)
    dfAll = dfAll.fillna(0)

    if show_families:
        famlist = show_families.split(',')
        df = df.loc[famlist]

    if not allstats:
        if csv:
            print(df.to_csv())
        else:
            print(df)
    else:
        if csv:
            print(dfAll.to_csv())
        else:
            print(dfAll)

    if chart and not single:
        ax = df.plot(kind='bar', figsize=(6,6))
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig('chart.png')
        print('Chart saved as chart.png.')

if __name__ == '__main__':
    cli()
