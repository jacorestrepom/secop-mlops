"""Constructor de features (notebook final, secciones 3, 4 y 6).

``Historia`` arma, a partir de las tablas limpias, todo el estado global que el notebook deja listo antes de
llamar a ``construir()``: eventos (participaciones con su resultado), contratos, proponentes, priors de
rivales y entidades, y el texto (TF-IDF + SVD ajustados con procesos anteriores a 2024).

``construir(pares, fecha_corte, historia)`` recibe pares empresa–proceso y devuelve las 103 features del
constructor más las 3 de participación corregidas (``feats_participacion_ok``), igual que la sección 6.

El cuerpo de las funciones es el del notebook: solo cambia que las variables globales ahora son atributos.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ml import config as C
from ml.pit import Llave, a_ns, contar_antes, f32, logit, norm_txt, ultimo_evento

# Columnas de cada tabla limpia que usa este módulo
COLS_PROCESOS = sorted(set([c for c in C.COLS_Q if c != "fecha_corte"]
                           + ["fecha_publicacion", "fecha_adjudicacion", "fecha_limite_ofertas", "descripcion_limpia"]))
COLS_PROVEEDORES = ["proveedor_key", "es_pyme", "tipo_empresa_grupo", "departamento", "municipio_norm", "municipio",
                    "fecha_registro", "tipo_proveedor", "es_usuario_objetivo"]
COLS_CONTRATOS = ["proveedor_key", "fecha_firma", "valor_contrato", "nit_entidad"]

GR_COMP = {"ent": ["nit_entidad"], "moddep": ["modalidad", "departamento_entidad"], "fam": ["unspsc_familia"],
           "entmod": ["nit_entidad", "modalidad"], "modprecio": ["modalidad", "_pbin"]}
GR_RIV = {"ent": ["nit_entidad"], "entmod": ["nit_entidad", "modalidad"], "fam": ["unspsc_familia"],
          "moddep": ["modalidad", "departamento_entidad"]}
DIMS = {"fam": ["unspsc_familia"], "ent": ["nit_entidad"], "dep": ["departamento_entidad"],
        "mod": ["modalidad"], "moddep": ["modalidad", "departamento_entidad"]}


def armar_base_rk(base, procesos, proveedores):
    """Sección 3: base de participaciones (base_entrenamiento) + fecha de adjudicación, mes y usuario objetivo."""
    base = base.copy()
    pf = procesos[["id_proceso", "fecha_publicacion", "fecha_adjudicacion"]].set_index("id_proceso")
    assert pf.index.is_unique, "id_proceso repetido en procesos.parquet"
    pub = base.id_proceso.map(pf.fecha_publicacion)
    base["fecha_adjudicacion"] = base.id_proceso.map(pf.fecha_adjudicacion)
    assert pub.notna().all() and (base.fecha_corte == pub.dt.normalize()).all(), "fecha_corte debe ser el día de publicación"
    base["mes"] = pub.dt.to_period("M").astype(str)
    prov = proveedores[["proveedor_key", "es_usuario_objetivo"]]
    base["objetivo"] = (base.proveedor_key.map(prov.drop_duplicates("proveedor_key").set_index("proveedor_key")
                                               .es_usuario_objetivo).fillna(False).astype(bool))
    base["gano"] = base.gano.astype(np.int8)
    base["anio"] = base.anio.astype(np.int16)
    return base


def pit_basicas(base_rk, proponentes, proveedores):
    """Sección 3: features PIT básicas de cada participación. Solo las columnas que usa el constructor
    (récord de la empresa y global, y pyme); se guardan en float32 como en el notebook."""
    bb = base_rk[["proveedor_key", "fecha_corte", "fecha_adjudicacion", "gano"]]
    tq = a_ns(bb.fecha_corte)
    tadj = np.maximum(a_ns(bb.fecha_adjudicacion), tq)  # un resultado nunca se conoce antes de publicar
    g = bb.gano.values.astype(float)
    pit = pd.DataFrame({"fila": np.arange(len(bb), dtype=np.int32)})
    for nombre, col in [("emp", "proveedor_key"), ("glob", None)]:
        k = np.zeros(len(bb), np.int64) if col is None else pd.factorize(bb[col])[0].astype(np.int64)
        pit[f"{nombre}_dec"] = contar_antes(k, tq, k, tadj).astype(np.float32)
        pit[f"{nombre}_gan"] = contar_antes(k, tq, k, tadj, g).astype(np.float32)
    prov = proveedores[["proveedor_key", "es_pyme"]]
    s = bb.proveedor_key.map(prov.drop_duplicates("proveedor_key").set_index("proveedor_key").es_pyme)
    pit["es_pyme"] = np.where(s.isna(), -1, s.fillna(False).astype(bool)).astype(np.int8)
    return pit


class Historia:
    """Estado que necesita ``construir()``: la sección 4 del notebook (eventos, texto y priors).

    Recibe las tablas limpias como DataFrames:
      base:        base_entrenamiento (una fila por participación con su resultado)
      procesos, proponentes, contratos, proveedores: tablas de data/clean/
    """

    def __init__(self, base, procesos, proponentes, contratos, proveedores):
        procesos = procesos.drop_duplicates("id_proceso") if not procesos.id_proceso.is_unique else procesos
        base_rk = armar_base_rk(base, procesos, proveedores)
        pit = pit_basicas(base_rk, proponentes, proveedores)

        # ------------------------------------------------------------
        # 1. Eventos: todas las participaciones de la base (con su resultado y su fecha de adjudicación)
        # ------------------------------------------------------------
        EV = base_rk[C.COLS_Q + ["proveedor_key", "tipo_proveedor", "n_proponentes",
                                 "fecha_adjudicacion", "gano"]].copy()
        for c in C.LLAVES_TXT:
            EV[c] = EV[c].astype("string")
        NEV = len(EV)
        ev_tq = a_ns(EV.fecha_corte)
        ev_tadj = np.maximum(a_ns(EV.fecha_adjudicacion), ev_tq)  # el resultado se conoce al adjudicar
        ev_g = EV.gano.values.astype(float)
        ev_proc = pd.factorize(EV.id_proceso)[0].astype(np.int32)
        ev_lp = f32(EV.log_precio_base).astype(np.float64)
        ev_pyme = pit.es_pyme.values

        pg_ev = np.where(pit.glob_dec.values > 0, pit.glob_gan.values / np.maximum(pit.glob_dec.values, 1), 0.276)
        ev_P_EMP = (pit.emp_gan.values + C.A_EMP * pg_ev) / (pit.emp_dec.values + C.A_EMP)

        # Proveedores (registro) y tipo de proveedor de cada empresa
        disp = list(proveedores.columns)
        MUNI = "municipio_norm" if "municipio_norm" in disp else "municipio"
        PROV = (proveedores[[c for c in ["proveedor_key", "es_pyme", "tipo_empresa_grupo", "departamento", MUNI,
                                         "fecha_registro", "tipo_proveedor"] if c in disp]]
                .drop_duplicates("proveedor_key").copy())
        PROV["proveedor_key"] = PROV.proveedor_key.astype("string")
        PROV = PROV.set_index("proveedor_key")

        # Propiedades de cada evento que usan las features de la entidad (D3)
        _dp, _de = norm_txt(EV.proveedor_key.map(PROV.departamento)), norm_txt(EV.departamento_entidad)
        ev_mismo = np.where((_dp == "") | (_de == ""), np.nan, (_dp == _de).astype(float))  # ganador local
        _L = Llave(EV, ["proveedor_key", "nit_entidad"])
        ev_entgan = contar_antes(_L.ev, ev_tq, _L.ev, ev_tadj, ev_g)  # victorias previas de la empresa en esa entidad
        del _dp, _de, _L

        # Un registro por proceso (para competencia y rivales)
        primera = pd.Series(np.arange(NEV)).groupby(ev_proc).first().values
        t_p = ev_tadj[primera]
        n_p = EV.n_proponentes.astype("float64").values[primera]
        agg = (pd.DataFrame({"p": ev_proc, "f": ev_P_EMP, "x": np.log1p(pit.emp_dec.values)})
               .groupby("p").agg(f_media=("f", "mean"), f_max=("f", "max"), x_media=("x", "mean")))
        pre_p = EV.fecha_corte.values[primera] < C.CORTE_AJUSTE
        PRIOR_RIV = {c: float(agg[c].values[pre_p].mean()) for c in agg.columns}
        CORTES_P = np.nanpercentile(ev_lp[EV.fecha_corte.values < C.CORTE_AJUSTE], np.arange(10, 100, 10))
        self.CORTES_P = CORTES_P
        EV["_pbin"] = self.pbin(ev_lp).values

        # Ganadores (para el comportamiento de cada entidad)
        w_idx = np.flatnonzero(ev_g == 1)
        w_t = ev_tadj[w_idx]
        W_REP = (ev_entgan[w_idx] > 0).astype(float)
        W_PYME = (ev_pyme[w_idx] == 1).astype(float)
        W_LOK = ~np.isnan(ev_mismo[w_idx])
        W_LOC = np.nan_to_num(ev_mismo[w_idx])
        pre_w = EV.fecha_corte.values[w_idx] < C.CORTE_AJUSTE
        PR_REP, PR_PYME, PR_LOC = W_REP[pre_w].mean(), W_PYME[pre_w].mean(), W_LOC[pre_w & W_LOK].mean()

        # Participaciones (proponentes) y contratos firmados
        PROP = proponentes.dropna(subset=["proveedor_key", "fecha_publicacion"]).reset_index(drop=True)
        PROP["proveedor_key"] = PROP.proveedor_key.astype("string")
        prop_t = a_ns(PROP.fecha_publicacion)
        CON = contratos[COLS_CONTRATOS]
        CON = CON.dropna(subset=["proveedor_key", "fecha_firma"]).reset_index(drop=True)
        CON["proveedor_key"] = CON.proveedor_key.astype("string")
        CON["nit_entidad"] = CON.nit_entidad.astype("string")
        con_t = a_ns(CON.fecha_firma)
        con_v = np.log1p(np.clip(pd.to_numeric(CON.valor_contrato, errors="coerce").fillna(0).values.astype(float), 0, None))
        TIPO = (PROP.sort_values("fecha_publicacion").groupby("proveedor_key").tipo_proveedor.last().astype("string")
                if "tipo_proveedor" in PROP else EV.groupby("proveedor_key").tipo_proveedor.last())

        # Fecha en que cada participación se hace PÚBLICA: cierre de ofertas (sin fecha válida: publicación + 15 días)
        pl = (procesos[["id_proceso", "fecha_publicacion", "fecha_limite_ofertas"]]
              .drop_duplicates("id_proceso").set_index("id_proceso"))
        pl.index = pl.index.astype(str)
        _PUB = pd.to_datetime(pl.fecha_publicacion).dt.normalize()
        _LIM = pd.to_datetime(pl.fecha_limite_ofertas, errors="coerce").dt.normalize()
        CONOCIDO = _LIM.where(_LIM >= _PUB, _PUB + pd.Timedelta(days=15))
        _t = PROP.id_proceso.astype(str).map(CONOCIDO)
        PROP_T_OK = np.where(_t.isna().values, prop_t + 15 * C.DIAS, a_ns(_t.fillna(pd.Timestamp("2000-01-01"))))
        del pl, _PUB, _LIM, _t

        # ------------------------------------------------------------
        # 2. Texto: TF-IDF + SVD (ajustados solo con procesos publicados antes de 2024)
        # ------------------------------------------------------------
        tx = base_rk.descripcion_limpia
        docs = tx.iloc[primera].fillna("").astype(str).values
        del tx
        ajuste = EV.fecha_corte.values[primera] < C.CORTE_AJUSTE
        VEC = TfidfVectorizer(min_df=5, max_df=0.3, max_features=100_000, ngram_range=(1, 2),
                              stop_words=C.STOP, sublinear_tf=True, dtype=np.float32).fit(docs[ajuste])
        Xd = VEC.transform(docs)
        muestra_svd = np.random.default_rng(C.SEMILLA).choice(np.flatnonzero(ajuste), min(150_000, int(ajuste.sum())),
                                                               replace=False)
        SVD = TruncatedSVD(64, random_state=C.SEMILLA).fit(Xd[muestra_svd])
        ZP = normalize(SVD.transform(Xd)).astype(np.float32)  # un vector por proceso de la base
        del Xd, docs
        gc.collect()

        # Procesos (sección 6): datos de cada proceso candidato
        PROC = (procesos[[c for c in C.COLS_Q if c != "fecha_corte"] + ["fecha_publicacion", "descripcion_limpia"]]
                .drop_duplicates("id_proceso").set_index("id_proceso"))
        PROC["fecha_corte"] = pd.to_datetime(PROC.fecha_publicacion).dt.normalize()

        self.__dict__.update(
            EV=EV, NEV=NEV, ev_tq=ev_tq, ev_tadj=ev_tadj, ev_g=ev_g, ev_proc=ev_proc, ev_lp=ev_lp,
            PROV=PROV, MUNI=MUNI, primera=primera, t_p=t_p, n_p=n_p, agg=agg, PRIOR_RIV=PRIOR_RIV,
            w_idx=w_idx, w_t=w_t, W_REP=W_REP, W_PYME=W_PYME, W_LOK=W_LOK, W_LOC=W_LOC,
            PR_REP=PR_REP, PR_PYME=PR_PYME, PR_LOC=PR_LOC, PROP=PROP, prop_t=prop_t, CON=CON, con_t=con_t,
            con_v=con_v, TIPO=TIPO, PROP_T_OK=PROP_T_OK, VEC=VEC, SVD=SVD, ZP=ZP, PROC=PROC,
        )
        self.FRAMES = {"EV": EV, "PROP": PROP, "CON": CON}
        self._LL = {}

    @classmethod
    def desde_carpeta(cls, clean_dir):
        """Lee las tablas de una carpeta con el formato de data/clean/ (base_entrenamiento, procesos, ...)."""
        d = Path(clean_dir)
        prov_cols = [c for c in COLS_PROVEEDORES if c in _columnas(d / "proveedores.parquet")]
        return cls(base=pd.read_parquet(d / "base_entrenamiento.parquet"),
                   procesos=pd.read_parquet(d / "procesos.parquet", columns=COLS_PROCESOS),
                   proponentes=pd.read_parquet(d / "proponentes.parquet"),
                   contratos=pd.read_parquet(d / "contratos.parquet", columns=COLS_CONTRATOS),
                   proveedores=pd.read_parquet(d / "proveedores.parquet", columns=prov_cols))

    # ------------------------------------------------------------
    # Ayudas del notebook
    # ------------------------------------------------------------
    def pbin(self, lp):
        return pd.Series(np.where(np.isnan(lp), None, np.digitize(lp, self.CORTES_P).astype(str)), dtype="string")

    def llave(self, frame, cols):
        clave = (frame, tuple(cols))
        if clave not in self._LL:
            self._LL[clave] = Llave(self.FRAMES[frame], cols)
        return self._LL[clave]

    def texto_a_vec(self, textos):
        return normalize(self.SVD.transform(self.VEC.transform(pd.Series(textos).fillna("").astype(str).values))
                         ).astype(np.float32)

    # ------------------------------------------------------------
    # 3. El constructor de features (notebook final, sección 4)
    # ------------------------------------------------------------
    def _construir(self, Q, Zq):
        """Q: una fila por par (proveedor_key + columnas del proceso + fecha_corte + tipo_proveedor).
        Zq: vector de texto de cada proceso. Devuelve las features en el orden ORDEN_D + ORDEN_D3.
        Todo se calcula con eventos ESTRICTAMENTE anteriores a fecha_corte (point-in-time)."""
        llave, pbin = self.llave, self.pbin
        NEV, ev_tq, ev_tadj, ev_g, ev_proc, ev_lp = self.NEV, self.ev_tq, self.ev_tadj, self.ev_g, self.ev_proc, self.ev_lp
        PROV, MUNI, primera, t_p, n_p, agg = self.PROV, self.MUNI, self.primera, self.t_p, self.n_p, self.agg
        PRIOR_RIV, w_idx, w_t = self.PRIOR_RIV, self.w_idx, self.w_t
        W_REP, W_PYME, W_LOK, W_LOC = self.W_REP, self.W_PYME, self.W_LOK, self.W_LOC
        PR_REP, PR_PYME, PR_LOC = self.PR_REP, self.PR_PYME, self.PR_LOC
        prop_t, con_t, con_v, ZP = self.prop_t, self.con_t, self.con_v, self.ZP
        A_ENT, A_EMP, A2, A_PRIOR, DIAS, SEMILLA = C.A_ENT, C.A_EMP, C.A2, C.A_PRIOR, C.DIAS, C.SEMILLA

        Q = Q.reset_index(drop=True).copy()
        for c in C.LLAVES_TXT:
            Q[c] = Q[c].astype("string")
        n = len(Q)
        tq = a_ns(Q.fecha_corte)
        lp = f32(Q.log_precio_base).astype(np.float64)
        Q["_pbin"] = pbin(lp).values
        F = {}

        def cev(cols, et, w=None, frame="EV", sub=None, t=None):
            L = llave(frame, cols)
            ek = L.ev if sub is None else L.ev[sub]
            return contar_antes(L.q(Q), tq if t is None else t, ek, et, w)

        # récord y conteos básicos
        emp_dec, emp_gan = cev(["proveedor_key"], ev_tadj), cev(["proveedor_key"], ev_tadj, ev_g)
        ent_dec, ent_gan = cev(["nit_entidad"], ev_tadj), cev(["nit_entidad"], ev_tadj, ev_g)
        mod_dec, mod_gan = cev(["modalidad"], ev_tadj), cev(["modalidad"], ev_tadj, ev_g)
        z_q, z_e = np.zeros(n, np.int64), np.zeros(NEV, np.int64)
        glob_dec, glob_gan = contar_antes(z_q, tq, z_e, ev_tadj), contar_antes(z_q, tq, z_e, ev_tadj, ev_g)
        pg = np.where(glob_dec > 0, glob_gan / np.maximum(glob_dec, 1), 0.276)
        pm = (mod_gan + A_ENT * pg) / (mod_dec + A_ENT)
        P_ENT = (ent_gan + A_ENT * pm) / (ent_dec + A_ENT)
        P_EMP = (emp_gan + A_EMP * pg) / (emp_dec + A_EMP)
        n_part = cev(["proveedor_key"], prop_t, frame="PROP")
        s = Q.proveedor_key.map(PROV.es_pyme)
        es_pyme = np.where(s.isna(), -1, s.fillna(False).astype(bool)).astype(np.int8)

        # proceso
        for c in C.NUM_P:
            F[c] = f32(Q[c])
        for c in C.CAT_P + ["tipo_proveedor"]:
            F[c] = Q[c].fillna("sin_dato").astype(str).values

        # competencia esperada (procesos ya adjudicados)
        gq, ge = np.zeros(n, np.int64), np.zeros(len(primera), np.int64)
        g_cnt, g_sum = contar_antes(gq, tq, ge, t_p), contar_antes(gq, tq, ge, t_p, n_p)
        media_glob = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1), 3.8)
        for nom, cols in GR_COMP.items():
            cnt = cev(cols, t_p, sub=primera)
            sm = cev(cols, t_p, n_p, sub=primera)
            un = cev(cols, t_p, (n_p == 1).astype(float), sub=primera)
            F[f"comp_{nom}_media"] = (sm + 3 * media_glob) / (cnt + 3)
            F[f"comp_{nom}_unico"] = (un + 3 * 0.4) / (cnt + 3)
            F[f"comp_{nom}_nproc"] = np.log1p(cnt)

        # empresa
        F["P_ENT"], F["P_EMP"] = P_ENT, P_EMP
        for nom, v in [("emp_dec", emp_dec), ("emp_gan", emp_gan), ("ent_dec", ent_dec), ("ent_gan", ent_gan),
                       ("n_part_prev", n_part)]:
            F[nom] = np.log1p(v)
        F["es_pyme"] = np.where(es_pyme < 0, np.nan, es_pyme)
        Le = llave("EV", ["proveedor_key"])
        qe = Le.q(Q)
        F["emp_dec_365"] = np.log1p(emp_dec - contar_antes(qe, tq - 365 * DIAS, Le.ev, ev_tadj))
        F["emp_gan_365"] = np.log1p(emp_gan - contar_antes(qe, tq - 365 * DIAS, Le.ev, ev_tadj, ev_g))
        Lp = llave("PROP", ["proveedor_key"])
        qp = Lp.q(Q)
        ip = ultimo_evento(qp, tq, Lp.ev, prop_t)
        F["dias_ult_part"] = np.where(ip >= 0, (tq - prop_t[np.maximum(ip, 0)]) / DIAS, np.nan)
        F["n_part_365"] = np.log1p(n_part - contar_antes(qp, tq - 365 * DIAS, Lp.ev, prop_t))
        Lc = llave("CON", ["proveedor_key"])
        qc_ = Lc.q(Q)
        n_con = contar_antes(qc_, tq, Lc.ev, con_t)
        F["con_n"] = np.log1p(n_con)
        F["con_n_365"] = np.log1p(n_con - contar_antes(qc_, tq - 365 * DIAS, Lc.ev, con_t))
        F["con_logvalor_medio"] = np.where(n_con > 0, contar_antes(qc_, tq, Lc.ev, con_t, con_v) / np.maximum(n_con, 1),
                                           np.nan)
        F["con_misma_entidad"] = np.log1p(cev(["proveedor_key", "nit_entidad"], con_t, frame="CON"))
        F["tipo_empresa_grupo"] = (Q.proveedor_key.map(PROV.tipo_empresa_grupo).astype("string").fillna("sin_dato")
                                   .astype(str).values)
        dp, de = norm_txt(Q.proveedor_key.map(PROV.departamento)), norm_txt(Q.departamento_entidad)
        F["mismo_depto"] = np.where((dp == "") | (de == ""), np.nan, (dp == de).astype(float))
        mp, me = norm_txt(Q.proveedor_key.map(PROV[MUNI])), norm_txt(Q.ciudad_norm)
        F["mismo_municipio"] = np.where((mp == "") | (me == ""), np.nan, (mp == me).astype(float))
        fr = pd.to_datetime(Q.proveedor_key.map(PROV.fecha_registro), errors="coerce")
        ant = (Q.fecha_corte - fr).dt.days.astype("float64").values
        F["antiguedad_dias"] = np.where(ant < 0, np.nan, ant)
        has_e = (~np.isnan(ev_lp)).astype(float)
        lp_e = np.nan_to_num(ev_lp)
        n_h, s_h = contar_antes(qe, tq, Le.ev, ev_tq, has_e), contar_antes(qe, tq, Le.ev, ev_tq, lp_e * has_e)
        n_g, s_g = (contar_antes(qe, tq, Le.ev, ev_tadj, has_e * ev_g),
                    contar_antes(qe, tq, Le.ev, ev_tadj, lp_e * has_e * ev_g))
        F["precio_vs_historial"] = np.where(n_h > 0, lp - s_h / np.maximum(n_h, 1), np.nan)
        F["precio_vs_ganados"] = np.where(n_g > 0, lp - s_g / np.maximum(n_g, 1), np.nan)

        # perfil por tema
        for nom, cols in DIMS.items():
            dec, gan = cev(["proveedor_key"] + cols, ev_tadj), cev(["proveedor_key"] + cols, ev_tadj, ev_g)
            F[f"perf_{nom}_tasa"] = (gan + A2 * P_EMP) / (dec + A2)
            F[f"perf_{nom}_n"] = np.log1p(dec)

        # procesos parecidos por texto (top 50 del historial ya decidido de la empresa)
        kn = {"knn_n": np.zeros(n), "knn_max_sim": np.full(n, np.nan), "knn_media_sim10": np.full(n, np.nan),
              "knn_tasa5": P_ENT.copy(), "knn_tasa50": P_ENT.copy(),
              "knn_max_sim_ganado": np.full(n, np.nan), "knn_max_sim_perdido": np.full(n, np.nan)}
        oe = np.argsort(Le.ev, kind="stable")
        ee_s = Le.ev[oe]
        oq = np.argsort(qe, kind="stable")
        qs = qe[oq]
        for grupo in np.split(oq, np.flatnonzero(np.diff(qs)) + 1):
            c = qe[grupo[0]]
            if c < 0:
                continue
            E = oe[np.searchsorted(ee_s, c):np.searchsorted(ee_s, c, side="right")]
            if len(E) == 0:
                continue
            tj, Ze, yj = ev_tadj[E], ZP[ev_proc[E]], ev_g[E]
            paso = int(max(64, min(1024, 2e7 // len(E))))
            for s0 in range(0, len(grupo), paso):
                qc = grupo[s0:s0 + paso]
                S = Zq[qc] @ Ze.T
                S[tj[None, :] >= tq[qc][:, None]] = -np.inf
                top = (np.argpartition(-S, 49, axis=1)[:, :50] if len(E) > 50
                       else np.broadcast_to(np.arange(len(E)), (len(qc), len(E))))
                st = np.take_along_axis(S, top, 1)
                o = np.argsort(-st, 1)
                top, st = np.take_along_axis(top, o, 1), np.take_along_axis(st, o, 1)
                st = st.astype(np.float16).astype(np.float32)
                val = np.isfinite(st)
                s0_ = np.where(val, st, 0.0)
                y_ = np.where(val, yj[top], 0.0)
                w = np.clip(s0_, 0, None) * val
                kn["knn_n"][qc] = val.sum(1)
                kn["knn_max_sim"][qc] = np.where(val[:, 0], s0_[:, 0], np.nan)
                v10 = val[:, :10]
                kn["knn_media_sim10"][qc] = np.where(v10.sum(1) > 0, (s0_[:, :10] * v10).sum(1) / np.maximum(v10.sum(1), 1),
                                                     np.nan)
                kn["knn_tasa5"][qc] = ((w[:, :5] * y_[:, :5]).sum(1) + P_ENT[qc]) / (w[:, :5].sum(1) + 1)
                kn["knn_tasa50"][qc] = ((w * y_).sum(1) + P_ENT[qc]) / (w.sum(1) + 1)
                for nom, cond in [("ganado", y_ == 1), ("perdido", y_ == 0)]:
                    mx = np.max(np.where(val & cond, s0_, -2), axis=1)
                    kn[f"knn_max_sim_{nom}"][qc] = np.where(mx > -2, mx, np.nan)
        F.update(kn)

        # D3 · rivales esperados
        for nom, cols in GR_RIV.items():
            cnt = cev(cols, t_p, sub=primera)
            for c in agg.columns:
                F[f"riv_{nom}_{c}"] = (cev(cols, t_p, agg[c].values, sub=primera) + A_PRIOR * PRIOR_RIV[c]) / (cnt + A_PRIOR)
        for nom in ["ent", "entmod"]:
            F[f"ventaja_vs_rivales_{nom}"] = logit(P_EMP) - logit(np.asarray(F[f"riv_{nom}_f_media"]))
        # D3 · comportamiento de la entidad
        for nom in ["ent", "entmod"]:
            cols = GR_RIV[nom]
            cnt = cev(cols, w_t, sub=w_idx)
            F[f"ent_{nom}_repite_ganador"] = (cev(cols, w_t, W_REP, sub=w_idx) + A_PRIOR * PR_REP) / (cnt + A_PRIOR)
            F[f"ent_{nom}_gana_pyme"] = (cev(cols, w_t, W_PYME, sub=w_idx) + A_PRIOR * PR_PYME) / (cnt + A_PRIOR)
            nk = cev(cols, w_t, W_LOK.astype(float), sub=w_idx)
            F[f"ent_{nom}_gana_local"] = (cev(cols, w_t, W_LOC, sub=w_idx) + A_PRIOR * PR_LOC) / (nk + A_PRIOR)
            F[f"ent_{nom}_n_ganadores"] = np.log1p(cnt)
        # D3 · incumbencia y rachas
        i1 = ultimo_evento(qe, tq, Le.ev, ev_tadj)
        F["emp_ult_resultado"] = np.where(i1 >= 0, ev_g[np.maximum(i1, 0)], np.nan)
        Lpe = llave("EV", ["proveedor_key", "nit_entidad"])
        i2 = ultimo_evento(Lpe.q(Q), tq, Lpe.ev, ev_tadj)
        F["emp_ent_ult_resultado"] = np.where(i2 >= 0, ev_g[np.maximum(i2, 0)], np.nan)
        i3 = ultimo_evento(qe, tq, Le.ev[w_idx], w_t)
        F["dias_ult_victoria"] = np.where(i3 >= 0, (tq - w_t[np.maximum(i3, 0)]) / DIAS, np.nan)
        for d in (90, 365):
            dec = emp_dec - contar_antes(qe, tq - d * DIAS, Le.ev, ev_tadj)
            gan = emp_gan - contar_antes(qe, tq - d * DIAS, Le.ev, ev_tadj, ev_g)
            F[f"emp_tasa_{d}"] = (gan + 2 * P_EMP) / (dec + 2)
            F[f"emp_dec_{d}d"] = np.log1p(dec)
        F["emp_tendencia"] = logit(np.asarray(F["emp_tasa_365"])) - logit(P_EMP)
        # D3 · grafo (embedding espectral del mes con los 24 meses previos)
        meses_q = Q.fecha_corte.dt.to_period("M")
        NE = len(Le.idx)
        for nom in ["grafo_part_ent", "grafo_gana_ent", "grafo_gana_fam"]:
            F[nom] = np.full(n, np.nan)
        for M in meses_q.unique():
            q = np.flatnonzero((meses_q == M).values)
            tM = M.start_time.value
            ventana = (ev_tq < tM) & (ev_tq >= tM - 730 * DIAS)
            for nom, colx, vic in [("grafo_part_ent", "nit_entidad", False), ("grafo_gana_ent", "nit_entidad", True),
                                   ("grafo_gana_fam", "unspsc_familia", True)]:
                Lx = llave("EV", [colx])
                NX = len(Lx.idx)
                sel = ventana & (Lx.ev >= 0) & ((ev_tadj < tM) & (ev_g == 1) if vic else True)
                A = sparse.csr_matrix((np.ones(sel.sum(), np.float32), (Le.ev[sel], Lx.ev[sel])), shape=(NE, NX))
                A.sum_duplicates()
                A.data = np.log1p(A.data)
                r = np.sqrt(np.asarray(A.sum(1)).ravel())
                cc = np.sqrt(np.asarray(A.sum(0)).ravel())
                A = (sparse.diags(np.where(r > 0, 1 / np.maximum(r, 1e-12), 0)) @ A
                     @ sparse.diags(np.where(cc > 0, 1 / np.maximum(cc, 1e-12), 0)))
                svd = TruncatedSVD(16, random_state=SEMILLA).fit(A)
                Z, V = svd.transform(A), svd.components_
                e_q, x_q = qe[q], Lx.q(Q)[q]
                ok = (e_q >= 0) & (x_q >= 0)
                sc = np.zeros(len(q))
                sc[ok] = (Z[e_q[ok]] * V[:, x_q[ok]].T).sum(1)
                tiene = ok & (np.asarray(A.getnnz(axis=1))[np.maximum(e_q, 0)] > 0)
                F[nom][q] = np.where(tiene, sc, np.nan)

        out = pd.DataFrame({c: F[c] for c in C.ORDEN_D + C.ORDEN_D3})
        for c in out.columns:
            out[c] = out[c].astype(str) if c in C.CAT_OUT else out[c].astype(np.float32)
        return out

    # ------------------------------------------------------------
    # 4. Participación con la fecha CORREGIDA (una participación se hace pública al cierre de ofertas)
    # ------------------------------------------------------------
    def feats_participacion_ok(self, Q):
        """Participaciones previas contadas SOLO si ya eran públicas (cierre de ofertas) en fecha_corte."""
        PROP_T_OK, DIAS = self.PROP_T_OK, C.DIAS
        tq = a_ns(Q.fecha_corte)
        qk = self.llave("PROP", ["proveedor_key"]).q(Q.assign(proveedor_key=Q.proveedor_key.astype("string")))
        ek = self.llave("PROP", ["proveedor_key"]).ev
        n = contar_antes(qk, tq, ek, PROP_T_OK)
        n365 = n - contar_antes(qk, tq - 365 * DIAS, ek, PROP_T_OK)
        i = ultimo_evento(qk, tq, ek, PROP_T_OK)
        dias = np.where(i >= 0, (tq - PROP_T_OK[np.maximum(i, 0)]) / DIAS, np.nan)
        return pd.DataFrame({"n_part_prev_ok": np.log1p(n), "n_part_365_ok": np.log1p(n365),
                             "dias_ult_part_ok": dias}).astype(np.float32)

    # ------------------------------------------------------------
    # Sección 6: de pares (empresa, proceso) a la tabla de features
    # ------------------------------------------------------------
    def construir(self, pares, fecha_corte=None):
        """Features de cada par (proveedor_key, id_proceso), como la sección 6 del notebook.

        pares: DataFrame con ``proveedor_key`` e ``id_proceso`` (las demás columnas se conservan en la salida).
        fecha_corte: None (como el notebook) usa el día de publicación de cada proceso; si se da una fecha,
            todas las features se calculan con la información anterior a esa fecha.
        Devuelve ``pares`` (con índice 0..n-1) seguido de las 103 features del constructor y las 3 corregidas.
        """
        PROC, TIPO = self.PROC, self.TIPO
        top = pares.reset_index(drop=True)
        Q = PROC.reindex(top.id_proceso.values)
        faltan = Q.fecha_corte.isna()
        assert not faltan.any(), f"Hay {int(faltan.sum())} candidatos sin datos de proceso"
        Q = Q.reset_index()[C.COLS_Q + ["descripcion_limpia"]].copy()
        if fecha_corte is not None:
            Q["fecha_corte"] = pd.Timestamp(fecha_corte)
        Q["proveedor_key"] = top.proveedor_key.astype("string").values
        Q["tipo_proveedor"] = top.proveedor_key.astype("string").map(TIPO).astype("string").values
        u, inv = np.unique(Q.id_proceso.astype(str).values, return_inverse=True)
        Zq = self.texto_a_vec(PROC.reindex(u).descripcion_limpia.values)[inv]  # texto: una vez por proceso
        Q = Q.drop(columns="descripcion_limpia")
        TROZO = C.TROZO
        partes = [pd.concat([self._construir(Q.iloc[s0:s0 + TROZO], Zq[s0:s0 + TROZO]),
                             self.feats_participacion_ok(Q.iloc[s0:s0 + TROZO].reset_index(drop=True))], axis=1)
                  for s0 in range(0, len(Q), TROZO)]
        Fm = pd.concat(partes, ignore_index=True)
        for c in C.CAT_OUT:
            Fm[c] = Fm[c].astype("category")
        return pd.concat([top, Fm], axis=1)


def construir(pares, fecha_corte=None, *, historia: Historia):
    """Interfaz fija del Paso 1: pares empresa–proceso → 103 features + 3 corregidas (ver ``Historia.construir``)."""
    return historia.construir(pares, fecha_corte)


def _columnas(ruta):
    import pyarrow.parquet as pq

    return pq.read_schema(ruta).names
