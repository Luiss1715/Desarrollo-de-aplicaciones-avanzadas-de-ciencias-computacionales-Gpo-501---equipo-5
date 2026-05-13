from pathlib import Path

from suicidality.config import TrainConfig
from suicidality.features import FeatureExtractor
from suicidality.lexing import RiskLexer
from suicidality.model import SuicideRiskModel
from suicidality.pipeline import SuicidalityPipeline


def test_pipeline_fit_predict(tmp_path):
    lexicon = tmp_path / "lex.json"
    lexicon.write_text('{"critical": ["kill myself"], "warning": ["hopeless"]}', encoding="utf-8")
    config = TrainConfig(use_spacy=False)
    pipeline = SuicidalityPipeline(
        config=config,
        lexer=RiskLexer(Path(lexicon)),
        features=FeatureExtractor(min_df=1),
        model=SuicideRiskModel(),
    )
    texts = ["I feel hopeless", "I am fine today"]
    labels = [1, 0]
    pipeline.fit(texts, labels)
    proba = pipeline.predict_proba(["I am hopeless again"])
    assert len(proba) == 1
