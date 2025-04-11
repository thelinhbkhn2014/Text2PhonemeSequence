# Text2PhonemeSequence: A Python Library for Text to Phoneme Conversion

This repository is an enhanced version of the original Text2PhonemeSequence library, which converts text to phoneme sequences for [XPhoneBERT](https://github.com/VinAIResearch/XPhoneBERT).

## Key Improvements

### Vietnamese Pronunciation Fixes

- ✅ Fixed "uy" incorrectly pronounced as "ui"
- ✅ Fixed "gì" incorrectly pronounced as "ghì" 
- ✅ Fixed "oo" sound pronunciation
- ✅ Fixed "r", "d", "gi" being pronounced identically
- 🔄 In progress: Fixing "s" and "x" pronounced identically

### Performance & Architecture Enhancements

- ✅ Applied phoneme post-processing to the dataset inference method (improved consistency)
- ✅ Refactored codebase for better organization and maintainability
- ✅ Created a unique phoneme dictionary per word (instead of segmenting) for improved speed
- ✅ Allow saving words that have never appeared in the G2P dictionary before, so that they do not need to be processed again through the pretrained G2P model, which helps improve speed
- ✅ Merging Vietnamese and English TSV dictionaries for easier multilingual support (Prioritize Vietnamese in case of overlapping sounds, with an estimated 405 overlapping sounds).

## Installation <a name="install"></a>

To install **Text2PhonemeSequence**:

```
$ pip install text2phonemesequence
```

## Usage Examples <a name="example"></a>

This library uses [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main) and [segments](https://pypi.org/project/segments/) toolkits for text-to-phoneme conversion. Information about `pretrained_g2p_model` and `language` can be found in the [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main) repository.

**Note**: For languages where words are not separated by spaces (e.g., Vietnamese and Chinese), an external tokenizer should be used before feeding the text into the library.

```python
from text2phonemesequence import Text2PhonemeSequence

# Load Text2PhonemeSequence
model = Text2PhonemeSequence(pretrained_g2p_model='charsiu/g2p_multilingual_byT5_small_100', language='eng-us', is_cuda=False)

# Convert a raw corpus
model.infer_dataset(input_file="/absolute/path/to/input/file", output_file="/absolute/path/to/output/file", batch_size=64) # batch_size is the number of words fed into the CharsiuG2P toolkit per times. 

# Convert a raw sentence
model.infer_sentence("The overwhelming majority of people in this country know how to sift the wheat from the chaff in what they hear and what they read .")
##Output: "ˈθ i ▁ ˈo ʊ v ɝ ˌw ɛ ɫ m ɪ ŋ ▁ m ə ˈd ʒ ɔ ɹ ə t i ▁ ˈɑ f ▁ ˈp i p ə ɫ ▁ ˈɪ n ▁ ˈθ ɪ s ▁ ˈk a ʊ n t ɹ i ▁ ˈn o ʊ ▁ ˈh o ʊ ▁ ˈt o ʊ ▁ ˈs ɪ f t ▁ ˈθ i ▁ ˈw i t ▁ ˈf ɹ ɑ m ▁ ˈθ i ▁ ˈt ʃ æ f ▁ ˈɪ n ▁ ˈw æ t ▁ ˈθ e ɪ ▁ ˈh ɪ ɹ ▁ ˈæ n d ▁ ˈw æ t ▁ ˈθ e ɪ ▁ ˈɹ ɛ d ▁ ."
```
