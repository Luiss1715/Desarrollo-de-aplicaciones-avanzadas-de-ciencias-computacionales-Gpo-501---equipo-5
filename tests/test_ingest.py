from suicidality.ingest import load_dataset


def test_load_dataset(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("title,text,is_suicide\nHi,Hello,yes\n", encoding="utf-8")
    dataset = load_dataset(str(csv_path))
    assert dataset.texts[0].startswith("Hi")
    assert dataset.labels == [1]
