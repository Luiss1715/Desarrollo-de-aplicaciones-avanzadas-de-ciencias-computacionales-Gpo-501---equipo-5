from suicidality.model import SuicideRiskModel


def test_model_fit_predict():
    model = SuicideRiskModel()
    x = [[0, 1], [1, 0], [0, 0], [1, 1]]
    y = [1, 0, 0, 1]
    model.fit(x, y)
    proba = model.predict_proba([[1, 0]])
    assert len(proba) == 1
