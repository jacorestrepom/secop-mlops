"""Regla de la compuerta de promoción (jobs/promover.py)."""

from jobs.promover import decidir

PROD = {"log_loss": 0.3825, "ece": 0.004, "log_loss_pymes": 0.39, "ece_pymes": 0.006}


def test_un_candidato_igual_no_se_promueve():
    pasa, motivo = decidir(dict(PROD), PROD, "log_loss", menor_es_mejor=True)
    assert not pasa and "no mejora" in motivo


def test_mejora_y_nada_empeora_se_promueve():
    assert decidir({**PROD, "log_loss": 0.38}, PROD, "log_loss", menor_es_mejor=True)[0]


def test_mejora_pero_empeora_en_pymes_no_se_promueve():
    pasa, motivo = decidir({**PROD, "log_loss": 0.38, "ece_pymes": 0.02}, PROD, "log_loss", menor_es_mejor=True)
    assert not pasa and "ece_pymes" in motivo


def test_metricas_mayor_es_mejor_y_faltantes():
    prod = {"recall_at_150": 0.789, "recall_at_150_pymes": 0.77}
    cand = {"recall_at_150": 0.80, "recall_at_150_pymes": 0.77}
    assert decidir(cand, prod, "recall_at_150", menor_es_mejor=False)[0]
    pasa, motivo = decidir({"recall_at_150": 0.80}, prod, "recall_at_150", menor_es_mejor=False)
    assert not pasa and "no se puede verificar" in motivo
