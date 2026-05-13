from pathlib import Path

from suicidality.lexing import RiskLexer


def test_lexing_detects_critical(tmp_path):
    lexicon = tmp_path / "lex.json"
    lexicon.write_text('{"critical": ["kill myself"], "warning": []}', encoding="utf-8")
    lexer = RiskLexer(Path(lexicon))
    result = lexer.scan("I will kill myself soon")
    assert result.flags["critical"] == 1
    assert result.critical_matches
