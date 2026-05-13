from suicidality.features import FeatureExtractor


def test_feature_shapes():
    fe = FeatureExtractor(min_df=1)
    texts = ["hello world", "hello there"]
    flags = [{"critical": 0}, {"critical": 1}]
    x = fe.fit_transform(texts, flags)
    assert x.shape[0] == 2
