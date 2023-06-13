from transformers import T5ForConditionalGeneration, AutoTokenizer
from segments import Tokenizer
import os
from tqdm import tqdm
from more_itertools import chunked


class Text2PhonemeSequence:
    def __init__(self, 
                 pretrained_g2p_model: str = 'charsiu/g2p_multilingual_byT5_small_100', 
                 tokenizer: str = 'google/byt5-small', 
                 language: str ='vie-n', 
                 is_cuda: bool = True,
                 **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_g2p_model)
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.model = self.model.cuda()
        self.punctuation = list('.?!¡,:;-()[]{}<>"') + list("'/‘”“/&#~@^|") + ['...', '*']
        self.update_phone = {punc: [punc] for punc in self.punctuation}
        self.segment_tool = Tokenizer()
        self.language = language
        self.phoneme_length = {'msa': 27, 'amh': 28, 'urd': 152, 'mri': 30, 'glg': 22, 
                               'swa': 28, 'est': 47, 'hbs-latn': 31, 'spa': 28, 'pol': 38, 
                               'spa-latin': 24, 'fra': 43, 'uig': 33, 'aze': 34, 'nob': 52, 
                               'swe': 36, 'por-bz': 28, 'arg': 30, 'yue': 227, 'sqi': 42, 
                               'cat': 27, 'hbs-cyrl': 31, 'grc': 39, 'bak': 44, 'hun': 67, 
                               'lat-clas': 61, 'ita': 37, 'san': 38, 'arm-w': 41, 'afr': 81, 
                               'ind': 87, 'sme': 31, 'egy': 22, 'rus': 62, 'ady': 29, 'epo': 43, 
                               'srp': 42, 'zho-t': 135, 'tha': 295, 'vie-c': 184, 'kur': 37, 
                               'ice': 37, 'geo': 75, 'cze': 43, 'ara': 370, 'tgl': 27, 'nan': 175, 
                               'por-po': 35, 'bos': 52, 'enm': 24, 'ltz': 55, 'ina': 26, 'ukr': 39, 
                               'mac': 44, 'kaz': 44, 'slk': 73, 'ang': 33, 'khm': 47, 'syc': 24, 
                               'eus': 35, 'spa-me': 28, 'tts': 26, 'gle': 36, 'slv': 35, 'ido': 23, 
                               'zho-s': 135, 'vie-n': 174, 'gre': 20, 'eng-us': 119, 'fin': 44, 
                               'pap': 26, 'tuk': 49, 'jpn': 174, 'bel': 58, 'uzb': 36, 'dan': 42, 
                               'ori': 45, 'ron': 30, 'bul': 40, 'bur': 52, 'lit': 87, 'dut': 48, 
                               'fra-qu': 255, 'eng-uk': 129, 'hin': 110, 'isl': 41, 'arm-e': 38, 
                               'kor': 175, 'tur': 54, 'fas': 70, 'ger': 56, 'tam': 38, 'wel-sw': 44, 
                               'vie-s': 172, 'mlt': 47, 'wel-nw': 49, 'slo': 27, 'lat-eccl': 43, 
                               'snd': 60, 'tat': 103, 'alb': 40, 'hau': 45, 'pus': 34, 'tib': 61, 
                               'sga': 47, 'heb': 39, 'hrx': 27, 'fao': 42, 'dsb': 29
                               }
        if self.language not in self.phoneme_length.keys():
            self.phoneme_length[self.language] = 50

        self.phone_dict = {}
        
        if not os.path.exists('./' + language + ".tsv"):
            os.system("wget https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/" + language + ".tsv")
        
        if os.path.exists('./' + language + ".tsv"):
            with open("./" + language + ".tsv", "r", encoding="utf-8") as f:
                list_words = f.read().strip().split("\n")

            for word_phone in list_words:
                word, phone = word_phone.split("\t")
                self.phone_dict[word] = [phone.split(',')[0]]


    def infer_dataset(self, input_file: str ='', seperate_syllabel_token: str = "_", output_file: str = "", batch_size: int = 64):
        """Infer batch data
        """
        with open(input_file, 'r', encoding="utf-8") as f:
            list_lines = f.read().strip().split("\n")
        list_words = []
        
        print("\nStarting build vocabulary...")
        for line in list_lines:
            words = line.strip().split("|")[-1].lower().split(" ")
            for w in words:
                w = w.replace(seperate_syllabel_token, " ")
                if w not in self.phone_dict.keys():
                    list_words.append(w)

        list_words_p = {i: f'<{self.language}>: {i}' for i in list_words}

        batches = [dict(row) for row in list(chunked(list_words_p.items(), int(batch_size)))]
        # [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}, {'e': 5}]  # example batch_size = 2
        
        for batch in tqdm(batches, ncols=100):
            out = self.tokenizer(list(batch.values()), padding=True, add_special_tokens=False, return_tensors='pt')
            if self.is_cuda:
                out['input_ids'] = out['input_ids'].cuda()
                out['attention_mask'] = out['attention_mask'].cuda()

            preds = self.model.generate(**out, num_beams=1, max_length=self.phoneme_length[self.language])
            phones = self.tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
            phones = [[p] for p in phones]

            assert len(phones) == len(batch)

            batch_word_phone = dict(zip(batch.keys(), phones))
            batch_word_phone.update(self.update_phone)
            self.phone_dict.update(batch_word_phone) 

        for w, p in self.phone_dict.items():
            try:
                segmented_phone = self.segment_tool(p[0], ipa=True)
            except:
                segmented_phone = self.segment_tool(p[0])
            p.append(segmented_phone)
            self.phone_dict[w] = list(set(p))
        
        print(f"\nSaving vocabulary to {output_file}")
        output = []
        for line in tqdm(list_lines):
            line = line.strip().split("|")
            prefix = line[0]
            list_words = line[-1].lower().split(" ")
            list_phones = []
            for i in range(len(list_words)):
                w = list_words[i].replace(seperate_syllabel_token, " ")
                list_phones.append(self.phone_dict[w][-1])
            output.append(prefix + "|" + " ▁ ".join(list_phones))

        with open(output_file, 'w', encoding="utf-8") as f:
            for r in output:
                f.write(f"{r}\n")
        
        print("\nDone!")
    

    def infer_sentence(self, sentence: str = "", seperate_syllabel_token: str = "_"):
        """Infer a sentence
        """
        list_words = sentence.split(" ")
        list_phones = []

        for i in range(len(list_words)):
            list_words[i] = list_words[i].replace(seperate_syllabel_token, " ")
            if list_words[i] in self.phone_dict:
                list_phones.append(self.phone_dict[list_words[i]][0])
            elif list_words[i] in self.punctuation:
                list_phones.append(list_words[i])   
            else:
                out = self.tokenizer('<' + self.language + '>: ' + list_words[i], padding=True, add_special_tokens=False, return_tensors='pt')
                if self.is_cuda:
                    out['input_ids'] = out['input_ids'].cuda()
                    out['attention_mask'] = out['attention_mask'].cuda()
                preds = self.model.generate(**out, num_beams=1, max_length=self.phoneme_length[self.language])
                phones = self.tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)
                list_phones.append(phones[0])

        for i in range(len(list_phones)):
            try:
                segmented_phone = self.segment_tool(list_phones[i], ipa=True)
            except:
                segmented_phone = self.segment_tool(list_phones[i])
            list_phones[i] = segmented_phone
        return " ▁ ".join(list_phones)


if __name__ == '__main__':
    a = Text2PhonemeSequence(pretrained_g2p_model='charsiu/g2p_multilingual_byT5_tiny_16_layers_100', language='eng-us', is_cuda=False)
    print(a.infer_sentence("The overwhelming majority of people in this country know how to sift the wheat from the chaff in what they hear and what they read ."))
    a.infer_dataset(input_file="input.txt", output_file="output.txt")
