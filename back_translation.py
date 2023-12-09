## credit: https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/back_translation/transformation.py

## currently not in use. 
## constantly raise "Returning Default due to Run Time Exception"

from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from nlaugmenter.interfaces.SentenceOperation import SentenceOperation
from nlaugmenter.tasks.TaskTypes import TaskType


class BackTranslation(SentenceOperation):
    tasks = [TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_TO_TEXT_GENERATION]
    languages = ["en"]
    heavy = True
    keywords = ["lexical", "model-based", "syntactic", "high-coverage"]
    def __init__(self, seed=1011, max_outputs=1, num_beams=2):
        super().__init__()
        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(
            name_en_de
        )
        
        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(
            name_de_en
        )
        self.num_beams = num_beams

    def back_translate(self, en: str):
        try:
            de = self.en2de(en)
            en_new = self.de2en(de)
        except Exception:
            print("Returning Default due to Run Time Exception")
            en_new = en
        return en_new

    def en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt")
        outputs = self.model_en_de.generate(input_ids)
        decoded = self.tokenizer_en_de.decode(
            outputs[0], skip_special_tokens=True
        )
      
        return decoded

    def de2en(self, input):
        input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt")
        outputs = self.model_de_en.generate(
            input_ids,
            num_return_sequences=self.max_outputs,
            num_beams=self.num_beams,
        )
        predicted_outputs = []
        for output in outputs:
            decoded = self.tokenizer_de_en.decode(
                output, skip_special_tokens=True
            )
            # TODO: this should be able to return multiple sequences
            predicted_outputs.append(decoded)

        return predicted_outputs

    def generate(self, sentence: str):
        perturbs = self.back_translate(sentence)
        return perturbs