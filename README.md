# Text2PhonemeSequence: A Python Library to convert text to phoneme sequences used for [XPhoneBERT](https://github.com/VinAIResearch/XPhoneBERT) 

- [Installation](#install)
- [Usage example](#example)

## Installation <a name="install"></a>

- To install **Text2PhonemeSequence**, users have to run the following command:

    `$ pip install text2phonemesequence` 

## Usage example <a name="example"></a>
The library uses CharsiuG2P to convert text to phoneme sequences. Users can find the information on `pretrained_g2p_model` and `language` in the [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main) repository. For languages where words are not separated by spaces such as Vietnamese and Chinese, users need to use an external tokenizer before feeding the dataset or sentences into our **Text2PhonemeSequence** library. 

```python
from text2phonemesequence import Text2PhonemeSequence

# Load Text2PhonemeSequence
model = Text2PhonemeSequence(pretrained_g2p_model='charsiu/g2p_multilingual_byT5_tiny_16_layers_100', language='eng-uk', is_cuda=False)


# Convert a raw corpus
model.infer_dataset(input_file="/absolute/path/to/input/file", output_file="/absolute/path/to/output/file")

# Convert a raw sentence
model.infer_sentence("The overwhelming majority of people in this country know how to sift the wheat from the chaff in what they hear and what they read .")
##Output: "ˈθ i ▁ ˈo ʊ v ɝ ˌw ɛ ɫ m ɪ ŋ ▁ m ə ˈd ʒ ɔ ɹ ə t i ▁ ˈɑ f ▁ ˈp i p ə ɫ ▁ ˈɪ n ▁ ˈθ ɪ s ▁ ˈk a ʊ n t ɹ i ▁ ˈn o ʊ ▁ ˈh o ʊ ▁ ˈt o ʊ ▁ ˈs ɪ f t ▁ ˈθ i ▁ ˈw i t ▁ ˈf ɹ ɑ m ▁ ˈθ i ▁ ˈt ʃ æ f ▁ ˈɪ n ▁ ˈw æ t ▁ ˈθ e ɪ ▁ ˈh ɪ ɹ ▁ ˈæ n d ▁ ˈw æ t ▁ ˈθ e ɪ ▁ ˈɹ ɛ d ▁ ."
```
