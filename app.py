"""
App básica de Streamlit — Nivel de ríos/quebradas
Marinilla, Quebrada La Bolsa (Red Agua - Cód. 31) / MARCO
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ------------------------------------------------------------------
# Parámetros fijos de la consulta
# ------------------------------------------------------------------

CODIGO_ESTACION = "31"
FECHA_DESDE = "2026-02-19"
FECHA_HASTA = "2026-02-25"

# Coordenadas por defecto
LAT_DEFECTO = 6.1806
LON_DEFECTO = -75.32974

API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"

LLAVE_FECHA = "level_date"
LLAVE_VALOR = "level"

CANDIDATOS_LAT = ["lat", "latitude", "latitud"]
CANDIDATOS_LON = ["lng", "lon", "longitude", "longitud"]


st.set_page_config(
    page_title="Nivel de estación — Quebrada La Bolsa",
    page_icon="🌊",
    layout="wide"
)


# ------------------------------------------------------------------
# Funciones de consulta
# ------------------------------------------------------------------

def obtener_serie_nivel(
    codigo_estacion,
    desde,
    hasta,
    calidad=1,
    timeout=30
):
    url = f"{API_BASE_URL}/{codigo_estacion}/nivel"

    params = {
        "desde": desde,
        "hasta": hasta,
        "calidad": calidad
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify=False
        )

        if resp.status_code == 200:
            return resp.json(), None

        return None, f"HTTP {resp.status_code} — {resp.text[:300]}"

    except requests.exceptions.RequestException as e:
        return None, f"Error de red: {e}"


def obtener_todas_las_paginas(datos_json, timeout=30):

    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")

    while siguiente_url:

        try:
            resp = requests.get(
                siguiente_url,
                timeout=timeout,
                verify=False
            )

        except requests.exceptions.RequestException:
            break

        if resp.status_code != 200:
            break

        pagina = resp.json()

        registros.extend(
            pagina.get("values", [])
        )

        siguiente_url = pagina.get("next")

    return registros


def detectar_coordenadas(datos_json):

    """
    Busca lat/lon en las llaves raíz de la respuesta.
    Si no las encuentra, usa el valor por defecto.
    """

    if not isinstance(datos_json, dict):
        return LAT_DEFECTO, LON_DEFECTO, False

    lat = next(
        (
            datos_json[k]
            for k in CANDIDATOS_LAT
            if k in datos_json
        ),
        None
    )

    lon = next(
        (
            datos_json[k]
            for k in CANDIDATOS_LON
            if k in datos_json
        ),
        None
    )

    if lat is not None and lon is not None:

        try:
            return float(lat), float(lon), True

        except (TypeError, ValueError):
            pass

    return LAT_DEFECTO, LON_DEFECTO, False


# ------------------------------------------------------------------
# Índice de calidad
# ------------------------------------------------------------------

def calcular_indice_calidad(df):

    """
    Índice simple (0-100) combinando completitud
    de la serie y proporción de outliers.
    """

    if df.empty or len(df) < 2:
        return 0.0, 0, 0

    df_idx = df.set_index("fecha")

    frecuencia_tipica = (
        df["fecha"]
        .diff()
        .dropna()
        .mode()
    )

    if len(frecuencia_tipica) == 0:
        return 0.0, 0, 0

    frecuencia_tipica = frecuencia_tipica[0]

    rango_completo = pd.date_range(
        start=df_idx.index.min(),
        end=df_idx.index.max(),
        freq=frecuencia_tipica
    )

    esperados = len(rango_completo)

    huecos = esperados - len(df_idx)

    completitud = (
        max(0.0, 1 - (huecos / esperados))
        if esperados > 0
        else 0.0
    )

    Q1 = df["nivel"].quantile(0.25)
    Q3 = df["nivel"].quantile(0.75)

    IQR = Q3 - Q1

    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    es_outlier = (
        (df["nivel"] < lim_inf)
        | (df["nivel"] > lim_sup)
        | (df["nivel"] < 0)
    )

    proporcion_outliers = es_outlier.mean()

    indice = (
        completitud * 0.7
        + (1 - proporcion_outliers) * 0.3
    ) * 100

    return (
        round(indice, 1),
        int(huecos),
        int(es_outlier.sum())
    )


# ------------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------------

def analizar_missing_values(df):

    """
    Missing values reales: huecos en la serie de tiempo.

    En una serie minuto a minuto, un "missing value" casi nunca
    aparece como NaN explícito — aparece como un hueco:
    el sensor debía reportar y no lo hizo.

    Para detectarlo correctamente comparamos contra la frecuencia
    esperada, no solo contamos NaN directos.
    """

    resultado = {
        "nan_directos": 0,
        "frecuencia_tipica": None,
        "esperadas": 0,
        "recibidas": 0,
        "huecos": 0,
        "porcentaje_huecos": 0.0,
        "df_regular": None,
        "huecos_detalle": None,
        "bloques_huecos": None,
    }

    if df.empty or len(df) < 2:
        return resultado

    # --------------------------------------------------------------
    # 1. NaN directos
    # --------------------------------------------------------------

    resultado["nan_directos"] = int(
        df["nivel"].isna().sum()
    )

    # --------------------------------------------------------------
    # 2. Frecuencia típica
    # --------------------------------------------------------------

    diferencias = (
        df["fecha"]
        .diff()
        .dropna()
    )

    modas = diferencias.mode()

    if len(modas) == 0:
        return resultado

    frecuencia_tipica = modas[0]

    resultado["frecuencia_tipica"] = frecuencia_tipica

    # --------------------------------------------------------------
    # 3. Crear serie regular
    # --------------------------------------------------------------

    df_indexado = df.set_index("fecha")

    rango_completo = pd.date_range(
        start=df_indexado.index.min(),
        end=df_indexado.index.max(),
        freq=frecuencia_tipica
    )

    df_regular = df_indexado.reindex(
        rango_completo
    )

    # --------------------------------------------------------------
    # 4. Calcular huecos
    # --------------------------------------------------------------

    huecos = int(
        df_regular["nivel"].isna().sum()
    )

    esperadas = len(rango_completo)

    resultado["esperadas"] = esperadas
    resultado["recibidas"] = df_indexado.shape[0]
    resultado["huecos"] = huecos

    resultado["porcentaje_huecos"] = (
        round((huecos / esperadas) * 100, 2)
        if esperadas > 0
        else 0.0
    )

    resultado["df_regular"] = df_regular

    # --------------------------------------------------------------
    # 5. Detalle de huecos
    # --------------------------------------------------------------

    huecos_detalle = df_regular[
        df_regular["nivel"].isna()
    ].copy()

    huecos_detalle.index.name = "fecha_esperada"

    resultado["huecos_detalle"] = (
        huecos_detalle
        .reset_index()[["fecha_esperada"]]
    )

    # --------------------------------------------------------------
    # 6. Agrupar huecos consecutivos
    # --------------------------------------------------------------

    es_hueco = df_regular["nivel"].isna()

    if es_hueco.any():

        grupo = (
            es_hueco != es_hueco.shift()
        ).cumsum()

        bloques = []

        for _, tramo in df_regular[
            es_hueco
        ].groupby(grupo[es_hueco]):

            inicio = tramo.index.min()
            fin = tramo.index.max()

            duracion = (
                fin - inicio
            ) + frecuencia_tipica

            bloques.append(
                {
                    "inicio": inicio,
                    "fin": fin,
                    "duracion": duracion,
                    "lecturas_perdidas": len(tramo),
                }
            )

        bloques_df = (
            pd.DataFrame(bloques)
            .sort_values(
                "duracion",
                ascending=False
            )
            .reset_index(drop=True)
        )

        resultado["bloques_huecos"] = bloques_df

    else:

        resultado["bloques_huecos"] = pd.DataFrame(
            columns=[
                "inicio",
                "fin",
                "duracion",
                "lecturas_perdidas"
            ]
        )

    return resultado


# ------------------------------------------------------------------
# Renderizar resultados
# ------------------------------------------------------------------

def mostrar_resultados(
    df,
    lat,
    lon,
    coords_reales,
    codigo_estacion
):

    """
    Renderiza todas las métricas, gráficos y análisis.
    """

    indice_calidad, huecos_idx, n_outliers = (
        calcular_indice_calidad(
            df.dropna(subset=["nivel"])
        )
    )

    mv = analizar_missing_values(df)

    # ==============================================================
    # INFORMACIÓN DE LA ESTACIÓN
    # ==============================================================

    st.title("🌊 Nivel de ríos y quebradas — CORNARE")

    st.caption(
        f"Estación: **{codigo_estacion}** · "
        f"Quebrada La Bolsa · "
        f"{FECHA_DESDE} → {FECHA_HASTA}"
    )

    st.divider()

    # ==============================================================
    # MÉTRICAS
    # ==============================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Lecturas",
        len(df)
    )

    col2.metric(
        "Nivel promedio",
        f"{df['nivel'].mean():.2f}"
    )

    col3.metric(
        "Índice de calidad",
        f"{indice_calidad} / 100"
    )

    col4.metric(
        "Outliers detectados",
        n_outliers
    )

    st.divider()

    # ==============================================================
    # GRÁFICO + MAPA
    # ==============================================================
    
    grafico_col, mapa_col = st.columns(2)

    # --------------------------------------------------------------
    # GRÁFICO
    # --------------------------------------------------------------

    with grafico_col:

        st.subheader("📈 Serie de nivel")

        if (
            mv.get("df_regular") is not None
            and not mv["df_regular"].empty
        ):
            serie = mv["df_regular"]["nivel"]

        else:
            serie = (
                df
                .set_index("fecha")["nivel"]
            )

        st.line_chart(
            serie,
            use_container_width=True
        )

        if (
            mv.get("bloques_huecos") is not None
            and not mv["bloques_huecos"].empty
        ):

            st.caption(
                "Los tramos vacíos de la línea corresponden "
                "a los huecos detectados en la serie."
            )

    # --------------------------------------------------------------
    # MAPA
    # --------------------------------------------------------------

    with mapa_col:

        st.subheader("📍 Ubicación de la estación")

        if coords_reales:

            st.caption(
                f"Coordenadas reportadas por la API: "
                f"`{lat}, {lon}`"
            )

        else:

            st.caption(
                "La API no trajo las coordenadas reales. "
                "Se utiliza la coordenada configurada "
                "por defecto."
            )

        mapa_df = pd.DataFrame(
            {
                "lat": [lat],
                "lon": [lon]
            }
        )

        st.map(
            mapa_df,
            zoom=10
        )

    st.divider()

    # ==============================================================
    # MISSING VALUES
    # ==============================================================

    st.subheader("🕳️ Missing Values")

    st.caption(
        "En una serie minuto a minuto, un missing value "
        "casi nunca aparece como NaN explícito — aparece "
        "como un hueco: el sensor debía reportar y no lo hizo. "
        "Por eso comparamos la serie contra su frecuencia "
        "esperada en lugar de solo contar NaN directos."
    )

    if mv["frecuencia_tipica"] is None:

        st.info(
            "No hay suficientes datos para estimar "
            "la frecuencia típica de reporte."
        )

    else:

        # ----------------------------------------------------------
        # Métricas de missing values
        # ----------------------------------------------------------

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        mcol1.metric(
            "NaN directos",
            mv["nan_directos"]
        )

        mcol2.metric(
            "Frecuencia típica",
            str(mv["frecuencia_tipica"])
        )

        mcol3.metric(
            "Lecturas esperadas",
            mv["esperadas"]
        )

        mcol4.metric(
            "Huecos reales",
            mv["huecos"],
            delta=f"{mv['porcentaje_huecos']}%",
            delta_color="inverse"
        )

        # ----------------------------------------------------------
        # Tabla de cortes
        # ----------------------------------------------------------

        bloques = mv.get(
            "bloques_huecos"
        )

        if (
            bloques is not None
            and not bloques.empty
        ):

            st.markdown(
                "**Cortes detectados "
                "(huecos consecutivos agrupados "
                "en un solo bloque):**"
            )

            bloques_mostrar = bloques.copy()

            bloques_mostrar["duracion"] = (
                bloques_mostrar[
                    "duracion"
                ].astype(str)
            )

            st.dataframe(
                bloques_mostrar.rename(
                    columns={
                        "inicio": "Sin datos desde",
                        "fin": "Sin datos hasta",
                        "duracion": "Duración del corte",
                        "lecturas_perdidas": "Lecturas perdidas",
                    }
                ),
                use_container_width=True
            )

            corte_mas_largo = bloques.iloc[0]

            st.warning(
                f"⚠️ El corte más largo va de "
                f"**{corte_mas_largo['inicio']}** a "
                f"**{corte_mas_largo['fin']}** "
                f"({corte_mas_largo['duracion']}, "
                f"{int(corte_mas_largo['lecturas_perdidas'])} "
                f"lecturas perdidas)."
            )

        # ----------------------------------------------------------
        # Detalle minuto a minuto
        # ----------------------------------------------------------

        with st.expander(
            "Ver detalle minuto a minuto de los huecos"
        ):

            st.write(
                f"- NaN directos en `nivel`: "
                f"**{mv['nan_directos']}**"
            )

            st.write(
                f"- Frecuencia típica entre lecturas: "
                f"**{mv['frecuencia_tipica']}**"
            )

            st.write(
                f"- Lecturas esperadas a frecuencia regular: "
                f"**{mv['esperadas']}**"
            )

            st.write(
                f"- Lecturas realmente recibidas: "
                f"**{mv['recibidas']}**"
            )

            st.write(
                f"- Huecos reales en la serie: "
                f"**{mv['huecos']} "
                f"({mv['porcentaje_huecos']}%)**"
            )

            if mv["huecos"] > 0:

                st.markdown(
                    "**Timestamps esperados donde "
                    "no llegó reporte:**"
                )

                st.dataframe(
                    mv["huecos_detalle"],
                    use_container_width=True
                )

            else:

                st.success(
                    "No se detectaron huecos: "
                    "la serie está completa a la "
                    "frecuencia esperada."
                )

    # ==============================================================
    # ÍNDICE DE CALIDAD
    # ==============================================================

    with st.expander(
        "Detalle del índice de calidad"
    ):

        st.write(
            f"- Huecos de reporte detectados "
            f"(índice de calidad): **{huecos_idx}**"
        )

        st.write(
            f"- Outliers (IQR + nivel negativo): "
            f"**{n_outliers}** de "
            f"{len(df.dropna(subset=['nivel']))} lecturas"
        )

        st.write(
            "El índice combina completitud de la serie "
            "(70%) y proporción de datos sin outliers (30%)."
        )

    # ==============================================================
    # DATOS CRUDOS
    # ==============================================================

    st.subheader("📄 Datos crudos")

    st.dataframe(
        df,
        use_container_width=True
    )

    csv_salida = (
        df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Descargar CSV",
        csv_salida,
        file_name=(
            f"nivel_estacion_"
            f"{codigo_estacion}.csv"
        ),
        mime="text/csv"
    )


# ------------------------------------------------------------------
# CONSULTA AUTOMÁTICA DE LA API
# ------------------------------------------------------------------

st.info(
    f"🔄 Consultando automáticamente la estación "
    f"`{CODIGO_ESTACION}`..."
)

with st.spinner("Consultando la API de CORNARE..."):

    datos_crudos, error = obtener_serie_nivel(
        CODIGO_ESTACION,
        FECHA_DESDE,
        FECHA_HASTA,
        calidad=1
    )


# ------------------------------------------------------------------
# Procesamiento de respuesta
# ------------------------------------------------------------------

if error:

    st.error(
        f"❌ No se pudo conectar con el servidor: {error}"
    )

    st.caption(
        "Verifica que el servidor de CORNARE "
        "esté disponible en este momento."
    )

else:

    registros = obtener_todas_las_paginas(
        datos_crudos
    )

    if not registros:

        st.warning(
            "No hay registros para esta estación "
            "y rango de fechas."
        )

    else:

        df = pd.DataFrame(registros)

        # ----------------------------------------------------------
        # Renombrar columnas
        # ----------------------------------------------------------

        df = df.rename(
            columns={
                LLAVE_FECHA: "fecha",
                LLAVE_VALOR: "nivel"
            }
        )

        # ----------------------------------------------------------
        # Conversión de tipos
        # ----------------------------------------------------------

        df["fecha"] = pd.to_datetime(
            df["fecha"],
            errors="coerce"
        )

        df["nivel"] = pd.to_numeric(
            df["nivel"],
            errors="coerce"
        )

        df = (
            df
            .dropna(subset=["fecha"])
            .sort_values("fecha")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------------
        # Coordenadas
        # ----------------------------------------------------------

        lat, lon, coords_reales = (
            detectar_coordenadas(
                datos_crudos
            )
        )

        # ----------------------------------------------------------
        # Mostrar resultados
        # ----------------------------------------------------------

        mostrar_resultados(
            df,
            lat,
            lon,
            coords_reales,
            CODIGO_ESTACION
        )
