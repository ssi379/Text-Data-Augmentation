from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import util

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoSentenceFactory(PororoFactoryBase):

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": [
                "stsb-roberta-base",
                "stsb-roberta-large",
                "stsb-bert-base",
                "stsb-bert-large",
                "stsb-distillbert-base",
            ],
            "ko": ["brainsbert.base.ko.kornli.korsts"],
            "ja": ["jasbert.base.ja.nli.sts"],
            "zh": ["zhsbert.base.zh.nli.sts"],
        }

    def load(self, device: str):
        from sentence_transformers import SentenceTransformer

        model_path = self.config.n_model

        if self.config.lang != "en":
            model_path = download_or_load(
                f"sbert/{self.config.n_model}",
                self.config.lang,
            )
        model = SentenceTransformer(model_path).eval().to(device)
        return PororoSBertSentence(model, self.config)


class PororoSBertSentence(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def find_similar_sentences(
            self,
            query: str,
            cands: List[str],
    ) -> Dict:
        query_embedding = self._model.encode(query, convert_to_tensor=True)
        corpus_embeddings = self._model.encode(cands, convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        k = min(len(cos_scores), 5)
        top_results = np.argpartition(-cos_scores, range(k))[0:k]
        top_results = top_results.tolist()

        result = list()
        for idx in top_results:
            result.append(
                (idx, cands[idx].strip(), round(cos_scores[idx].item(), 2)))

        return {
            "query": query.strip(),
            "ranking": result,
        }

    def predict(self, sent: str):
        outputs = self._model.encode([sent])[0]
        return outputs


