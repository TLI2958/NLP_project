## ref: https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/back_translation/transformation.py
## Modifications
## removed original submit format
## removed verbose
## from de_en to ru_en since de_en not rendering well. just returning special tokens.
## don't know if it is the problem of my system or model. 

from transformers import FSMTForConditionalGeneration, FSMTTokenizer

class BackTranslation():
    def __init__(self, seed=1011, max_outputs=1, num_beams=2):
        super().__init__()
        name_en_ru = "facebook/wmt19-en-ru"
        self.tokenizer_en_ru = FSMTTokenizer.from_pretrained(name_en_ru)
        self.model_en_ru = FSMTForConditionalGeneration.from_pretrained(
            name_en_ru
        )
        
        name_ru_en = "facebook/wmt19-ru-en"
        self.tokenizer_ru_en = FSMTTokenizer.from_pretrained(name_ru_en)
        self.model_ru_en = FSMTForConditionalGeneration.from_pretrained(
            name_ru_en
        )
        self.num_beams = num_beams

    def back_translate(self, en: str):
        try:
            ru = self.en2ru(en)
            en_new = self.ru2en(ru)
        except Exception:
            print("Returning Default due to Run Time Exception")
            en_new = en
        return en_new

    def en2ru(self, input):
        input_ids = self.tokenizer_en_ru.encode(input, return_tensors="pt")
        outputs = self.model_en_ru.generate(input_ids)
        decoded = self.tokenizer_en_ru.decode(
            outputs[0], skip_special_tokens=True
        )
      
        return decoded

    def ru2en(self, input):
        input_ids = self.tokenizer_ru_en.encode(input, return_tensors="pt")
        outputs = self.model_ru_en.generate(
            input_ids,
            num_return_sequences=self.max_outputs,
            num_beams=self.num_beams,
        )
        predicted_outputs = []
        for output in outputs:
            decoded = self.tokenizer_ru_en.decode(
                output, skip_special_tokens=True
            )
            # TODO: this should be able to return multiple sequences
            predicted_outputs.append(decoded)

        return predicted_outputs

    def generate(self, sentence: str):
        perturbs = self.back_translate(sentence)
        return perturbs