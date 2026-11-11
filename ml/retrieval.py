"""Retrieval M1 (notebook 03, celdas 18, 19 y 31): de todos los procesos abiertos a 150 candidatos por empresa.

``HistoriaRetrieval`` arma el estado de la celda 18 (llaves enteras ``pk``/``pid``, historial de
participaciones y contratos, TF-IDF de descripciones y de nombres, usuarios objetivo) y la parte fija de la
celda 19 (códigos de modalidad y tipo de contrato).

``recuperar(empresas, procesos_abiertos, fecha_corte, ...)`` es ``candidatos_mes2`` + la puntuación con M1 de
la celda 31, con dos cambios de interfaz (no de lógica):

- las empresas y los procesos se reciben como argumentos (el notebook los sacaba del mes: procesos publicados
  en el mes y usuarios objetivo que se presentaron a alguno);
- no se calcula la etiqueta ``y`` (solo servía para entrenar y evaluar).

Los ``pk`` y ``pid`` son los mismos de ``map_pk.parquet`` y ``map_pid.parquet``: se recalculan aquí con la
misma regla del notebook 03, así que esos archivos no hacen falta en inferencia.
"""

from __future__ import annotations

import gc
import re
import unicodedata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ml import config as C

COLS_PROCESOS = ["id_proceso", "fecha_publicacion", "unspsc_familia", "unspsc_segmento", "nit_entidad",
                 "departamento_entidad", "descripcion_limpia", "modalidad", "tipo_contrato_grupo", "log_precio_base",
                 "dias_para_ofertar", "proc_mes"]
COLS_PROPONENTES = ["id_proceso", "proveedor_key", "fecha_publicacion"]
COLS_CONTRATOS = ["proveedor_key", "fecha_firma", "unspsc_familia", "unspsc_segmento", "nit_entidad", "departamento"]

PESOS_GENERAN, PESOS_REFUERZAN, ATTRS = C.PESOS_GENERAN, C.PESOS_REFUERZAN, C.ATTRS
PESOS_TODOS = {**PESOS_GENERAN, **PESOS_REFUERZAN}
COMPONENTES = {"unspsc_familia": "f_fam", "nit_entidad": "f_ent",
               "unspsc_segmento": "f_seg", "departamento_entidad": "f_dep"}
FEATS, CATS = C.FEATS_M1, C.CATS_M1


def limpiar(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", x)).strip()


def onehot(serie, cats):
    idx = pd.Categorical(serie, categories=cats).codes
    filas = np.where(idx >= 0)[0]
    return sp.csr_matrix((np.ones(len(filas), dtype=np.float32), (filas, idx[filas])), shape=(len(serie), len(cats)))


def perfil(hist, llave):
    return hist.dropna(subset=[llave]).groupby(["pk", llave]).size().rename("n").reset_index()


def textos_de(p):
    # Desempate fijo (aprobado por ciencia de datos): el notebook usaba sort_values("fecha_publicacion"), que no es
    # estable; con fechas empatadas, los 50 textos elegidos dependían de la plataforma. Ahora: fecha y luego id_proceso.
    return (p.dropna(subset=["descripcion_limpia"])
              .sort_values(["fecha_publicacion", "id_proceso"], kind="mergesort")
              .groupby("pk")["descripcion_limpia"].apply(lambda s: " ".join(s.tail(50))))


def perfil_prestado(v, perf_pool):
    out = {}
    for llave, pp in perf_pool.items():
        m = v.merge(pp.rename(columns={"pk": "pk_vec"}), on="pk_vec")
        m["n"] = m["w"] * m["n"]
        out[llave] = m.groupby(["pk", llave], as_index=False)["n"].sum()
    return out


def afinidad_comp(perf, procm):
    gen = [perf[lv].merge(procm[["pid", lv]].dropna(), on=lv)[["pk", "pid"]] for lv in PESOS_GENERAN]
    a = pd.concat(gen).drop_duplicates()
    for llave, col in COMPONENTES.items():
        a = a.merge(procm[["pid", llave]], on="pid", how="left").merge(perf[llave], on=["pk", llave], how="left")
        a[col] = np.log1p(a["n"].fillna(0)).astype(np.float32)
        a = a.drop(columns=[llave, "n"])
    a["s_afin"] = sum(PESOS_TODOS[lv] * a[c] for lv, c in COMPONENTES.items())
    return a


def predecir_por_bloques(modelo, X, tam=1_000_000):
    salida = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), tam):
        salida[i:i + tam] = modelo.predict(X.iloc[i:i + tam])
    return salida


class HistoriaRetrieval:
    """Estado del retrieval (notebook 03, celdas 18 y 19).

    Tablas como DataFrames: procesos, proponentes, contratos y proveedores de data/clean/, y los nombres de los
    proveedores (``proveedor_key``, ``nombre_proveedor``) para el arranque en frío.
    """

    def __init__(self, procesos, proponentes, contratos, proveedores, nombres):
        # ---------- 1. Datos limpios ----------
        proc = procesos[["id_proceso", "fecha_publicacion", "unspsc_familia", "unspsc_segmento",
                         "nit_entidad", "departamento_entidad", "descripcion_limpia"]].copy()
        proponentes = proponentes[COLS_PROPONENTES].copy()
        contratos = contratos[COLS_CONTRATOS]
        contratos = contratos.dropna(subset=["proveedor_key"]).rename(columns={"departamento": "departamento_entidad"})

        # Llaves como enteros
        empresas_idx = pd.Index(pd.concat([proponentes["proveedor_key"], contratos["proveedor_key"]]).unique())
        proponentes["pk"] = empresas_idx.get_indexer(proponentes["proveedor_key"])
        contratos["pk"] = empresas_idx.get_indexer(contratos["proveedor_key"])
        proc["pid"] = np.arange(len(proc))
        proponentes["pid"] = pd.Index(proc["id_proceso"]).get_indexer(proponentes["id_proceso"])

        part_all = proponentes[["pk", "pid", "fecha_publicacion", "id_proceso"]].merge(
            proc[["pid", "descripcion_limpia"] + ATTRS], on="pid", how="left")

        # ---------- 2. TF-IDF de descripciones (solo procesos antes de 2025) ----------
        vec = TfidfVectorizer(min_df=5, max_df=0.3, max_features=100_000, ngram_range=(1, 2),
                              stop_words=C.STOP, sublinear_tf=True, dtype=np.float32)
        vec.fit(proc.loc[proc["fecha_publicacion"] < C.CORTE_TFIDF_RETRIEVAL, "descripcion_limpia"].fillna(""))

        # ---------- 4. Proveedores (usuarios objetivo) y similitud entre empresas ----------
        prov = proveedores.merge(nombres, on="proveedor_key", how="left")
        prov["pk"] = empresas_idx.get_indexer(prov["proveedor_key"])
        prov_obj = prov[(prov["es_usuario_objetivo"] == True) & (prov["pk"] >= 0)].drop_duplicates("pk").set_index("pk")  # noqa: E712
        objetivo = prov_obj.index.values

        vec_nombre = TfidfVectorizer(min_df=3, ngram_range=(1, 2), stop_words=C.STOP_NOMBRE,
                                     sublinear_tf=True, dtype=np.float32)
        vec_nombre.fit(prov_obj["nombre_proveedor"].dropna().map(limpiar))
        DEPTOS = sorted(prov_obj["departamento"].dropna().unique())
        TIPOS = sorted(prov_obj["tipo_empresa_grupo"].dropna().unique())

        # ---------- Celda 19: atributos del proceso que usan los modelos ----------
        extra = procesos[["id_proceso", "modalidad", "tipo_contrato_grupo", "log_precio_base", "dias_para_ofertar",
                          "proc_mes"]]
        proc = proc.merge(extra, on="id_proceso", how="left")
        proc["modalidad_cod"] = proc["modalidad"].astype("category").cat.codes
        proc["tipo_contrato_cod"] = proc["tipo_contrato_grupo"].astype("category").cat.codes

        self.proc, self.proponentes, self.contratos, self.part_all = proc, proponentes, contratos, part_all
        self.empresas_idx, self.vec, self.prov_obj, self.objetivo = empresas_idx, vec, prov_obj, objetivo
        self.vec_nombre, self.DEPTOS, self.TIPOS = vec_nombre, DEPTOS, TIPOS
        self.VENTANA = pd.Timedelta(days=C.VENTANA_DIAS)

    @classmethod
    def desde_carpeta(cls, clean_dir):
        """Lee las tablas de una carpeta con el formato de data/clean/ (incluye proveedores_nombres.parquet)."""
        d = Path(clean_dir)
        return cls(procesos=pd.read_parquet(d / "procesos.parquet", columns=COLS_PROCESOS),
                   proponentes=pd.read_parquet(d / "proponentes.parquet", columns=COLS_PROPONENTES),
                   contratos=pd.read_parquet(d / "contratos.parquet", columns=COLS_CONTRATOS),
                   proveedores=pd.read_parquet(d / "proveedores.parquet"),
                   nombres=pd.read_parquet(d / "proveedores_nombres.parquet"))

    # ---------- funciones base del retrieval (celda 18) ----------
    def historial(self, corte, empresas):
        part_all, contratos, VENTANA = self.part_all, self.contratos, self.VENTANA
        p = part_all[(part_all["fecha_publicacion"] < corte) & (part_all["fecha_publicacion"] >= corte - VENTANA)
                     & part_all["pk"].isin(empresas)]
        c = contratos[(contratos["fecha_firma"] < corte) & (contratos["fecha_firma"] >= corte - VENTANA)
                      & contratos["pk"].isin(empresas)]
        return p, pd.concat([p[["pk"] + ATTRS], c[["pk"] + ATTRS]])

    def top_texto(self, Xe, pks, procm):
        Xp = self.vec.transform(procm["descripcion_limpia"].fillna("").values)
        pids, filas = procm["pid"].values, []
        for i in range(0, Xe.shape[0], 2000):
            sim = (Xe[i:i + 2000] @ Xp.T).toarray()
            k = min(C.TOP_TEXTO, sim.shape[1])
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            filas.append(pd.DataFrame({"pk": np.repeat(pks[i:i + 2000], k), "pid": pids[top].ravel(),
                                       "s_texto": np.take_along_axis(sim, top, axis=1).ravel()}))
        t = pd.concat(filas)
        return t[t["s_texto"] > 0]

    def bloques(self, pks):
        d = self.prov_obj.reindex(pks)
        N = normalize(self.vec_nombre.transform(d["nombre_proveedor"].fillna("").map(limpiar)))
        D = onehot(d["departamento"], self.DEPTOS)
        T = onehot(d["tipo_empresa_grupo"], self.TIPOS)
        P = onehot(d["es_pyme"].map({True: "si", False: "no"}), ["si", "no"])
        return N, D, T, P

    # ---------- celda 19 ----------
    def vecinos2(self, pk_frio, pool):
        A, B = self.bloques(pk_frio), self.bloques(pool)
        S = sum(w * (a @ b.T) for w, a, b in zip(C.W_SIM_VECINOS, A, B, strict=True)).toarray()
        k = min(C.K_VECINOS, S.shape[1])
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        v = pd.DataFrame({"pk": np.repeat(pk_frio, k), "pk_vec": np.asarray(pool)[top].ravel(),
                          "w": np.take_along_axis(S, top, axis=1).ravel()})
        v = v[v["w"] > 0]
        sim_max = v.groupby("pk")["w"].max().rename("sim_vecinos")
        v["w"] = v["w"] / v.groupby("pk")["w"].transform("sum")
        return v, sim_max

    def ret_local(self, combos, Xe, fila_de, procm):
        loc = (combos[["pk", "modalidad", "departamento_entidad"]]
               .merge(procm[["pid", "modalidad", "departamento_entidad"]], on=["modalidad", "departamento_entidad"])
               [["pk", "pid"]].drop_duplicates())
        loc = loc[loc["pk"].isin(fila_de.index)]
        if loc.empty:
            return pd.DataFrame(columns=["pk", "pid", "s_loc"])
        Xp = self.vec.transform(procm["descripcion_limpia"].fillna("").values)
        col_de = pd.Series(np.arange(len(procm)), index=procm["pid"].values)
        sims = np.empty(len(loc), dtype=np.float32)
        for i in range(0, len(loc), 100_000):
            b = loc.iloc[i:i + 100_000]
            sims[i:i + 100_000] = np.asarray(
                Xe[fila_de[b["pk"]].values].multiply(Xp[col_de[b["pid"]].values]).sum(axis=1)).ravel()
        loc["s_loc"] = sims
        loc["_r"] = loc.groupby("pk")["s_loc"].rank(ascending=False, method="first")
        return loc[loc["_r"] <= C.TOP_LOCAL].drop(columns="_r")

    def vector_frio(self, v, pk_frio, textos_pool):
        pos_pool = pd.Series(np.arange(len(textos_pool)), index=textos_pool.index)
        pos_frio = pd.Series(np.arange(len(pk_frio)), index=pk_frio)
        Wn = sp.csr_matrix((v["w"].values.astype(np.float32),
                            (pos_frio[v["pk"]].values, pos_pool[v["pk_vec"]].values)),
                           shape=(len(pk_frio), len(textos_pool)))
        Xv = normalize(Wn @ self.vec.transform(textos_pool.values))
        Xn = normalize(self.vec.transform(self.prov_obj["nombre_proveedor"].reindex(pk_frio).fillna("").map(limpiar).values))
        return normalize(0.5 * Xv + 0.5 * Xn)

    def candidatos(self, activos, procm, mes):
        """``candidatos_mes2`` del notebook 03 sin la etiqueta: todos los candidatos con sus 22 atributos."""
        proc, part_all, prov_obj, objetivo = self.proc, self.part_all, self.prov_obj, self.objetivo
        historial, top_texto, ret_local = self.historial, self.top_texto, self.ret_local
        vec, VENTANA = self.vec, self.VENTANA

        # ---------- Con historial ----------
        p_act, hist_act = historial(mes, activos)
        con_hist = hist_act["pk"].unique()
        sin_hist = np.setdiff1d(activos, con_hist)
        a_w = afinidad_comp({k: perfil(hist_act, k) for k in ATTRS}, procm)
        tx = textos_de(p_act)
        Xe_w = vec.transform(tx.values)
        fila_w = pd.Series(np.arange(len(tx)), index=tx.index)
        t_w = top_texto(Xe_w, tx.index.values, procm)
        comb_w = (p_act.merge(proc[["pid", "modalidad"]], on="pid")
                       .groupby(["pk", "modalidad", "departamento_entidad"]).size().rename("n_local").reset_index())
        l_w = ret_local(comb_w, Xe_w, fila_w, procm)
        c_w = a_w.merge(t_w, on=["pk", "pid"], how="outer").merge(l_w, on=["pk", "pid"], how="outer")
        c_w["es_frio"] = 0

        # ---------- Sin historial ----------
        en_v = (part_all["fecha_publicacion"] < mes) & (part_all["fecha_publicacion"] >= mes - VENTANA)
        pool = np.intersect1d(part_all.loc[en_v, "pk"].unique(), objetivo)
        p_pool, hist_pool = historial(mes, pool)
        perf_pool = {k: perfil(hist_pool, k) for k in ATTRS}
        textos_pool = textos_de(p_pool).reindex(pool, fill_value="")
        v, sim_max = self.vecinos2(sin_hist, pool)
        a_c = afinidad_comp(perfil_prestado(v, perf_pool), procm)
        Xe_c = self.vector_frio(v, sin_hist, textos_pool)
        fila_c = pd.Series(np.arange(len(sin_hist)), index=sin_hist)
        t_c = top_texto(Xe_c, sin_hist, procm)

        mod_pool = (p_pool.merge(proc[["pid", "modalidad"]], on="pid")
                          .groupby(["pk", "modalidad"]).size().rename("n").reset_index())
        mod_c = (v.merge(mod_pool.rename(columns={"pk": "pk_vec"}), on="pk_vec")
                  .assign(n=lambda x: x["w"] * x["n"]).groupby(["pk", "modalidad"], as_index=False)["n"].sum()
                  .sort_values(["pk", "n"], ascending=[True, False]).groupby("pk").head(2))
        dep_c = pd.DataFrame({"pk": sin_hist, "departamento_entidad": prov_obj["departamento"].reindex(sin_hist).values})
        dep_vec = (v.merge(perf_pool["departamento_entidad"].rename(columns={"pk": "pk_vec"}), on="pk_vec")
                    .assign(n=lambda x: x["w"] * x["n"]).sort_values("n", ascending=False)
                    .drop_duplicates("pk")[["pk", "departamento_entidad"]]
                    .rename(columns={"departamento_entidad": "dep_vec"}))
        dep_c = dep_c.merge(dep_vec, on="pk", how="left")
        dep_c["departamento_entidad"] = dep_c["departamento_entidad"].fillna(dep_c["dep_vec"])
        comb_c = (mod_c.merge(dep_c[["pk", "departamento_entidad"]].dropna(), on="pk")
                       .rename(columns={"n": "n_local"}))
        l_c = ret_local(comb_c, Xe_c, fila_c, procm)
        c_c = a_c.merge(t_c, on=["pk", "pid"], how="outer").merge(l_c, on=["pk", "pid"], how="outer")
        c_c["es_frio"] = 1

        # ---------- Unir y construir features ----------
        c = pd.concat([c_w, c_c], ignore_index=True)
        del c_w, c_c, a_w, a_c, t_w, t_c, l_w, l_c
        c["f_local"] = c["s_loc"].notna().astype(np.float32)
        c["s_texto"] = c["s_texto"].fillna(c["s_loc"])
        c = c.drop(columns="s_loc")
        cols0 = list(COMPONENTES.values()) + ["s_afin", "s_texto"]
        c[cols0] = c[cols0].fillna(0)
        for s in ["s_afin", "s_texto"]:
            c[f"{s}_n"] = c[s] / c.groupby("pk")[s].transform("max").replace(0, 1)
        c["s_hib"] = 0.5 * c["s_afin_n"] + 0.5 * c["s_texto_n"]
        c["rank_afin"] = c.groupby("pk")["s_afin"].rank(ascending=False, method="first")
        c["rank_texto"] = c.groupby("pk")["s_texto"].rank(ascending=False, method="first")
        c["n_cand"] = c.groupby("pk")["pid"].transform("size")
        c = c.merge(p_act.groupby("pk").size().rename("n_part_hist"), on="pk", how="left")
        c["n_part_hist"] = c["n_part_hist"].fillna(0)
        c = c.merge(sim_max, on="pk", how="left")
        c = c.merge(procm[["pid", "modalidad", "departamento_entidad", "modalidad_cod", "tipo_contrato_cod",
                           "log_precio_base", "dias_para_ofertar", "proc_mes"]], on="pid", how="left")
        comb_all = pd.concat([comb_w, comb_c[["pk", "modalidad", "departamento_entidad", "n_local"]]])
        c = c.merge(comb_all, on=["pk", "modalidad", "departamento_entidad"], how="left")
        c["n_local"] = c["n_local"].fillna(0)
        c = c.drop(columns=["modalidad", "departamento_entidad"])
        num = [f for f in FEATS if f not in CATS]
        c[num] = c[num].astype(np.float32)
        gc.collect()
        return c

    def entradas_del_mes(self, mes):
        """Empresas y procesos que el notebook 03 usó para un mes (para la paridad y para evaluar).

        Procesos: publicados en el mes. Empresas: usuarios objetivo que se presentaron a alguno de ellos.
        Devuelve (lista de proveedor_key, lista de id_proceso).
        """
        mes = pd.Timestamp(mes)
        proc, proponentes = self.proc, self.proponentes
        procm = proc[(proc["fecha_publicacion"] >= mes) & (proc["fecha_publicacion"] < mes + pd.offsets.MonthBegin(1))]
        pos = proponentes.loc[proponentes["pid"].isin(procm["pid"]), ["pk", "pid"]].drop_duplicates()
        pos = pos[pos["pk"].isin(self.objetivo)]
        activos = pos["pk"].unique()
        return list(self.empresas_idx.values[activos]), list(procm["id_proceso"].values)

    def recuperar(self, empresas, procesos_abiertos, fecha_corte, modelo, k=C.K):
        """Top ``k`` candidatos por empresa con ``score_retrieval``, ``rank_retrieval`` y ``es_frio``.

        empresas: ``proveedor_key`` de las empresas a puntuar. El notebook usó los usuarios objetivo que se
            presentaron a algún proceso del mes; las que no tienen ``pk`` (nunca aparecieron en proponentes ni en
            contratos) no se pueden atender y se descartan.
        procesos_abiertos: ``id_proceso`` de los procesos candidatos (el notebook usó los publicados en el mes).
        fecha_corte: el historial se toma de los 2 años anteriores a esta fecha (el notebook usó el día 1 del mes).
        modelo: M1 como ``lgb.Booster`` o ruta a ``M1_final_unico.txt``.
        """
        modelo = modelo if isinstance(modelo, lgb.Booster) else lgb.Booster(model_file=str(modelo))
        mes = pd.Timestamp(fecha_corte)
        pk = self.empresas_idx.get_indexer(pd.Index(pd.unique(pd.Series(list(empresas), dtype=object))))
        activos = pk[pk >= 0]
        procm = self.proc[self.proc["id_proceso"].isin(set(procesos_abiertos))]
        c = self.candidatos(activos, procm, mes)
        c["score_retrieval"] = predecir_por_bloques(modelo, c[FEATS])
        c["rank_retrieval"] = c.groupby("pk")["score_retrieval"].rank(ascending=False, method="first")
        top = c.loc[c["rank_retrieval"] <= k, ["pk", "pid", "score_retrieval", "rank_retrieval", "es_frio"]]
        top = top.reset_index(drop=True)
        top.insert(2, "proveedor_key", self.empresas_idx.values[top.pk.values])
        top.insert(3, "id_proceso", self.proc["id_proceso"].values[top.pid.values])
        return top


def recuperar(empresas, procesos_abiertos, fecha_corte, *, historia: HistoriaRetrieval, modelo, k=C.K):
    """Interfaz fija del Paso 1 (ver ``HistoriaRetrieval.recuperar``)."""
    return historia.recuperar(empresas, procesos_abiertos, fecha_corte, modelo, k=k)
