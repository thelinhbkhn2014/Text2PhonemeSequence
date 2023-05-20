from transformers import T5ForConditionalGeneration, AutoTokenizer
from segments import Tokenizer
import os
from tqdm import tqdm



class Text2PhonemeSequence:
    def __init__(self, pretrained_g2p_model='charsiu/g2p_multilingual_byT5_tiny_16_layers_100', language='vie-c', is_cuda=True):
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_g2p_model)
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.model = model.cuda()
        self.punctuation = list('.?!,:;-()[]{}<>"') + list("'/‘”“/&#~@^|") + ['...', '*']
        self.segment_tool = Tokenizer()
        self.language = language
        self.phoneme_length = {'msa.tsv': 27, 'amh.tsv': 28, 'urd.tsv': 152, 'mri.tsv': 30, 'glg.tsv': 22, 'swa.tsv': 28, 'est.tsv': 47, 'hbs-latn.tsv': 31, 'spa.tsv': 28, 'pol.tsv': 38, 'spa-latin.tsv': 24, 'fra.tsv': 43, 'uig.tsv': 33, 'aze.tsv': 34, 'nob.tsv': 52, 'swe.tsv': 36, 'por-bz.tsv': 28, 'arg.tsv': 30, 'yue.tsv': 227, 'sqi.tsv': 42, 'cat.tsv': 27, 'hbs-cyrl.tsv': 31, 'grc.tsv': 39, 'bak.tsv': 44, 'hun.tsv': 67, 'lat-clas.tsv': 61, 'ita.tsv': 37, 'san.tsv': 38, 'arm-w.tsv': 41, 'afr.tsv': 81, 'ind.tsv': 87, 'sme.tsv': 31, 'egy.tsv': 22, 'rus.tsv': 62, 'ady.tsv': 29, 'epo.tsv': 43, 'srp.tsv': 42, 'zho-t.tsv': 135, 'tha.tsv': 295, 'vie-c.tsv': 184, 'kur.tsv': 37, 'ice.tsv': 37, 'geo.tsv': 75, 'cze.tsv': 43, 'ara.tsv': 370, 'tgl.tsv': 27, 'nan.tsv': 175, 'por-po.tsv': 35, 'bos.tsv': 52, 'enm.tsv': 24, 'ltz.tsv': 55, 'ina.tsv': 26, 'ukr.tsv': 39, 'mac.tsv': 44, 'kaz.tsv': 44, 'slk.tsv': 73, 'ang.tsv': 33, 'khm.tsv': 47, 'syc.tsv': 24, 'eus.tsv': 35, 'spa-me.tsv': 28, 'tts.tsv': 26, 'gle.tsv': 36, 'slv.tsv': 35, 'ido.tsv': 23, 'zho-s.tsv': 135, 'vie-n.tsv': 174, 'gre.tsv': 20, 'eng-us.tsv': 119, 'fin.tsv': 44, 'pap.tsv': 26, 'tuk.tsv': 49, 'jpn.tsv': 174, 'bel.tsv': 58, 'uzb.tsv': 36, 'dan.tsv': 42, 'ori.tsv': 45, 'ron.tsv': 30, 'bul.tsv': 40, 'bur.tsv': 52, 'lit.tsv': 87, 'dut.tsv': 48, 'fra-qu.tsv': 255, 'eng-uk.tsv': 129, 'hin.tsv': 110, 'isl.tsv': 41, 'arm-e.tsv': 38, 'kor.tsv': 175, 'tur.tsv': 54, 'fas.tsv': 70, 'ger.tsv': 56, 'tam.tsv': 38, 'wel-sw.tsv': 44, 'vie-s.tsv': 172, 'mlt.tsv': 47, 'wel-nw.tsv': 49, 'slo.tsv': 27, 'lat-eccl.tsv': 43, 'snd.tsv': 60, 'tat.tsv': 103, 'alb.tsv': 40, 'hau.tsv': 45, 'pus.tsv': 34, 'tib.tsv': 61, 'sga.tsv': 47, 'heb.tsv': 39, 'hrx.tsv': 27, 'fao.tsv': 42, 'dsb.tsv': 29}
        self.phone_dict = {}
        if not os.path.exists('./' + language + ".tsv"):
            os.system("wget https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/" + language + ".tsv")
        if os.path.exists('./' + language + ".tsv"):
            f = open("./" + language + ".tsv", "r")
            list_words = f.read().strip().split("\n")
            f.close()
            for word_phone in list_words:
                w_p = word_phone.split("\t")
                assert len(w_p) == 2
                if "," not in w_p[1]:
                    self.phone_dict[w_p[0]] = [w_p[1]]
                else:
                    self.phone_dict[w_p[0]] = [w_p[1].split(',')[0]]

    def infer_dataset(self, input_file='', seperate_syllabel_token= "_", output_file=""):
        f = open(input_file, 'r')
        list_lines = f.readlines()
        f.close()
        list_words = []
        print("Building vocabulary!")
        for line in list_lines:
            words = line.strip().split(" ")
            for w in words:
                w = w.replace(seperate_syllabel_token, " ").lower()
                if w not in self.phone_dict.keys():
                    list_words.append(w)
        list_words_p = ['<' + self.language + '>: ' + i for i in list_words]
        out = self.tokenizer(list_words_p, padding=True, add_special_tokens=False, return_tensors='pt')
        if self.is_cuda:
            out['input_ids'] = out['input_ids'].cuda()
            out['attention_mask'] = out['attention_mask'].cuda()
        if self.language + '.tsv' not in self.phoneme_length.keys():
            self.phoneme_length[self.language + '.tsv'] = 50
        preds = self.model.generate(**out, num_beams=1, max_length=self.phoneme_length[self.language + '.tsv'])
        phones = self.tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)
        assert len(phones) == len(list_words)
        for i in range(len(phones)):
            if list_words[i] in self.punctuation:
                phones[i] = list_words[i]
            self.phone_dict[list_words[i]] = [phones[i]]
        for w in self.phone_dict.keys():
            try:
                segmented_phone = self.segment_tool(self.phone_dict[w][0], ipa=True)
            except:
                segmented_phone = self.segment_tool(self.phone_dict[w][0])
            self.phone_dict[w].append(segmented_phone)
        
        f = open(input_file, 'r')
        list_lines = f.readlines()
        f.close()
        f = open(output_file, 'w')
        for line in tqdm(list_lines):
            line = line.strip().split(" ")
            for i in range(len(line)):
                line[i] = self.phone_dict[line[i].replace(seperate_syllabel_token, " ").lower()][1]
            f.write(" ▁ ".join(line))
            f.write("\n")
        f.close()
    
    def infer_sentence(self, sentence="", seperate_syllabel_token="_"):
        list_words = sentence.split(" ")
        list_phones = []
        for i in range(len(list_words)):
            if list_words[i] in self.phone_dict:
                list_phones.append(self.phone_dict[list_words[i]][0])
            elif list_words[i] in self.punctuation:
                list_phones.append(list_words[i])   
            else:
                out = self.tokenizer('<' + self.language + '>: ' + list_words[i], padding=True, add_special_tokens=False, return_tensors='pt')
                if self.is_cuda:
                    out['input_ids'] = out['input_ids'].cuda()
                    out['attention_mask'] = out['attention_mask'].cuda()
                if self.language + '.tsv' not in self.phoneme_length.keys():
                    self.phoneme_length[self.language + '.tsv'] = 50
                preds = self.model.generate(**out, num_beams=1, max_length=self.phoneme_length[self.language + '.tsv'])
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