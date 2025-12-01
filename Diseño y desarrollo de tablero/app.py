import numpy as np
import pandas as pd
import joblib

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px

# 1. Carga de datos y modelos

DATA_PATH = "base_final_con_recomendacion.csv"
df = pd.read_csv(DATA_PATH)

TARGET_REG = "price"
TARGET_CLF = "recomended"
FEATURE_COLS = [c for c in df.columns if c not in [TARGET_REG, TARGET_CLF]]

# scaler y modelos entrenados
scaler = joblib.load("scaler_airbnb_london.pkl")
reg_model = joblib.load("mlp_regresion_price_airbnb.pkl")
clf_model = joblib.load("mlp_clasificacion_recomended_airbnb.pkl")

# Valores por defecto: medianas para numéricas, 0 para dummies
default_values: dict[str, float] = {}
for col in FEATURE_COLS:
    # si todos los valores son 0/1 asumimos dummy
    uniques = set(df[col].dropna().unique())
    if uniques <= {0, 1}:
        default_values[col] = 0.0
    else:
        default_values[col] = float(df[col].median())


# Mapeos de dropdowns -> columnas dummy
ZONA_MAP = {
    "Central": None,  # categoría base (todas las dummies en 0)
    "East": "neighbourhood_cleansed_East",
    "North": "neighbourhood_cleansed_North",
    "South": "neighbourhood_cleansed_South",
    "West": "neighbourhood_cleansed_West",
}

PROPERTY_MAP = {
    "Entire home/apt": None,  # base
    "Private room": "property_type_Private room",
    "Other": "property_type_Other",
}

ROOM_MAP = {
    "Entire home/apt": None,  # base
    "Private room": "room_type_Private room",
    "Shared room": "room_type_Shared room",
    "Hotel room": "room_type_Hotel room",
}


def build_feature_vector(
    accommodates,
    bedrooms,
    bathrooms,
    beds,
    minimum_nights,
    availability_365,
    zona,
    property_type,
    room_type,
):
    """
    Construye un vector de features en el mismo orden de FEATURE_COLS
    a partir de los inputs del usuario y valores por defecto.
    """
    # empezamos de la configuración "típica": medianas y dummies en 0
    x = default_values.copy()

    # numéricas principales
    if accommodates is not None:
        x["accommodates"] = accommodates
    if bedrooms is not None and "bedrooms" in x:
        x["bedrooms"] = bedrooms
    if bathrooms is not None and "bathrooms" in x:
        x["bathrooms"] = bathrooms
    if beds is not None and "beds" in x:
        x["beds"] = beds
    if minimum_nights is not None and "minimum_nights" in x:
        x["minimum_nights"] = minimum_nights
    if availability_365 is not None and "availability_365" in x:
        x["availability_365"] = availability_365

    # resetear dummies de zona/property/room
    for col in ZONA_MAP.values():
        if col is not None and col in x:
            x[col] = 0
    for col in PROPERTY_MAP.values():
        if col is not None and col in x:
            x[col] = 0
    for col in ROOM_MAP.values():
        if col is not None and col in x:
            x[col] = 0

    # zona
    col_z = ZONA_MAP.get(zona)
    if col_z is not None and col_z in x:
        x[col_z] = 1

    # tipo de propiedad
    col_p = PROPERTY_MAP.get(property_type)
    if col_p is not None and col_p in x:
        x[col_p] = 1

    # tipo de habitación
    col_r = ROOM_MAP.get(room_type)
    if col_r is not None and col_r in x:
        x[col_r] = 1

    # ordenar según FEATURE_COLS
    x_vec = np.array([[x[c] for c in FEATURE_COLS]], dtype=float)
    return x_vec


# 2. Inicialización de la app

app = Dash(__name__)


# 3. Gráficas estáticas mejoradas

# Preparación de datos de visualización
df_viz = df.copy()
df_viz["zona"] = np.select(
    [
        df_viz.get("neighbourhood_cleansed_East", 0) == 1,
        df_viz.get("neighbourhood_cleansed_North", 0) == 1,
        df_viz.get("neighbourhood_cleansed_South", 0) == 1,
        df_viz.get("neighbourhood_cleansed_West", 0) == 1,
    ],
    ["East", "North", "South", "West"],
    default="Central",
)
df_viz["recom_label"] = np.where(
    df_viz[TARGET_CLF] == 1, "Recomendado", "No recomendado"
)


def _estilo_figura(fig, titulo: str):
    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor="center"),
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=60),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # leyenda debajo de la gráfica
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


# 3.1 Boxplot de precios por zona con puntos individuales
fig_box_zona = px.box(
    df_viz,
    x="zona",
    y=TARGET_REG,
    points="all",
    category_orders={"zona": ["Central", "East", "North", "South", "West"]},
    color="zona",
    labels={"zona": "Zona de Londres", TARGET_REG: "Precio por noche (GBP)"},
)
fig_box_zona = _estilo_figura(fig_box_zona, "Distribución de precios por zona")
# el color ya está codificado en el eje X, podemos ocultar la leyenda
fig_box_zona.update_layout(showlegend=False)

# 3.2 Conteo de anuncios recomendados vs no recomendados
recom_counts = (
    df_viz.groupby("recom_label", as_index=False)[TARGET_CLF]
    .count()
    .rename(columns={TARGET_CLF: "conteo"})
)

fig_hist_recom = px.bar(
    recom_counts,
    x="recom_label",
    y="conteo",
    text="conteo",
    labels={"recom_label": "Estado de recomendación", "conteo": "Número de anuncios"},
)
fig_hist_recom.update_traces(textposition="outside")
fig_hist_recom = _estilo_figura(
    fig_hist_recom, "Anuncios recomendados vs no recomendados"
)

# 3.3 Dispersión precio vs capacidad coloreado por zona
if {"accommodates", TARGET_REG}.issubset(df_viz.columns):
    hover_cols = []
    if "minimum_nights" in df_viz.columns:
        hover_cols.append("minimum_nights")
    if "availability_365" in df_viz.columns:
        hover_cols.append("availability_365")

    fig_scatter_capacidad = px.scatter(
        df_viz,
        x="accommodates",
        y=TARGET_REG,
        color="zona",
        hover_data=hover_cols,
        labels={
            "accommodates": "Capacidad (huéspedes)",
            TARGET_REG: "Precio por noche (GBP)",
            "zona": "Zona",
        },
    )
    fig_scatter_capacidad = _estilo_figura(
        fig_scatter_capacidad, "Precio vs capacidad del anuncio"
    )
else:
    fig_scatter_capacidad = None


# 4. Layout del tablero

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "margin": "0",
        "padding": "20px 40px",
        "backgroundColor": "#f3f4f6",
    },
    children=[
        html.Div(
            children=[
                html.H1(
                    "Tablero Airbnb Londres",
                    style={"textAlign": "center", "marginBottom": "4px"},
                ),
                html.P(
                    "Explora el comportamiento de precios y recomendaciones en alojamientos de Airbnb en Londres. "
                    "Ajusta las características de un anuncio y obtén un precio sugerido y la probabilidad de ser recomendado.",
                    style={
                        "textAlign": "center",
                        "maxWidth": "900px",
                        "margin": "0 auto 24px auto",
                        "color": "#4b5563",
                    },
                ),
            ]
        ),
        html.Div(
            style={
                "display": "flex",
                "gap": "24px",
                "alignItems": "flex-start",
            },
            children=[
                # Panel de entrada
                html.Div(
                    style={
                        "flex": "1",
                        "backgroundColor": "#ffffff",
                        "borderRadius": "12px",
                        "padding": "18px 18px 24px 18px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.H3(
                            "Características del anuncio",
                            style={"marginTop": "0", "marginBottom": "12px"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "12px 16px",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Label("Acomoda (huéspedes)"),
                                        dcc.Input(
                                            id="input-accommodates",
                                            type="number",
                                            min=1,
                                            step=1,
                                            value=2,
                                            style={"width": "100%"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Habitaciones"),
                                        dcc.Input(
                                            id="input-bedrooms",
                                            type="number",
                                            min=0,
                                            step=1,
                                            value=1,
                                            style={"width": "100%"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Baños"),
                                        dcc.Input(
                                            id="input-bathrooms",
                                            type="number",
                                            min=0,
                                            step=0.5,
                                            value=1,
                                            style={"width": "100%"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Camas"),
                                        dcc.Input(
                                            id="input-beds",
                                            type="number",
                                            min=0,
                                            step=1,
                                            value=1,
                                            style={"width": "100%"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Noches mínimas"),
                                        dcc.Input(
                                            id="input-minimum-nights",
                                            type="number",
                                            min=1,
                                            step=1,
                                            value=2,
                                            style={"width": "100%"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Disponibilidad (365 días)"),
                                        dcc.Input(
                                            id="input-availability-365",
                                            type="number",
                                            min=0,
                                            step=1,
                                            value=180,
                                            style={"width": "100%"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Zona"),
                                        dcc.Dropdown(
                                            id="input-zona",
                                            options=[
                                                {
                                                    "label": "Central",
                                                    "value": "Central",
                                                },
                                                {"label": "East", "value": "East"},
                                                {"label": "North", "value": "North"},
                                                {"label": "South", "value": "South"},
                                                {"label": "West", "value": "West"},
                                            ],
                                            value="Central",
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Tipo de propiedad"),
                                        dcc.Dropdown(
                                            id="input-property-type",
                                            options=[
                                                {
                                                    "label": "Entire home/apt",
                                                    "value": "Entire home/apt",
                                                },
                                                {
                                                    "label": "Private room",
                                                    "value": "Private room",
                                                },
                                                {"label": "Other", "value": "Other"},
                                            ],
                                            value="Entire home/apt",
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Tipo de habitación"),
                                        dcc.Dropdown(
                                            id="input-room-type",
                                            options=[
                                                {
                                                    "label": "Entire home/apt",
                                                    "value": "Entire home/apt",
                                                },
                                                {
                                                    "label": "Private room",
                                                    "value": "Private room",
                                                },
                                                {
                                                    "label": "Shared room",
                                                    "value": "Shared room",
                                                },
                                                {
                                                    "label": "Hotel room",
                                                    "value": "Hotel room",
                                                },
                                            ],
                                            value="Entire home/apt",
                                            clearable=False,
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Button(
                            "Calcular precio y recomendación",
                            id="btn-calcular",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "marginTop": "16px",
                                "padding": "10px",
                                "borderRadius": "8px",
                                "border": "none",
                                "backgroundColor": "#2563eb",
                                "color": "white",
                                "fontWeight": "600",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                ),
                # Panel de resultados y gráficas
                html.Div(
                    style={
                        "flex": "1.6",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": "#ffffff",
                                "borderRadius": "12px",
                                "padding": "16px 20px",
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                            },
                            children=[
                                html.H3(
                                    "Resultados del modelo",
                                    style={"marginTop": 0},
                                ),
                                html.Div(
                                    id="output-precio",
                                    style={
                                        "fontSize": "18px",
                                        "marginBottom": "8px",
                                        "fontWeight": "600",
                                    },
                                ),
                                html.Div(
                                    id="output-recomendacion",
                                    style={
                                        "fontSize": "16px",
                                        "color": "#374151",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "16px"},
                            children=[
                                html.Div(
                                    style={
                                        "flex": "1",
                                        "backgroundColor": "#ffffff",
                                        "borderRadius": "12px",
                                        "padding": "8px",
                                        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                                    },
                                    children=[dcc.Graph(figure=fig_box_zona)],
                                ),
                                html.Div(
                                    style={
                                        "flex": "1",
                                        "backgroundColor": "#ffffff",
                                        "borderRadius": "12px",
                                        "padding": "8px",
                                        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                                    },
                                    children=[dcc.Graph(figure=fig_hist_recom)],
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "backgroundColor": "#ffffff",
                                "borderRadius": "12px",
                                "padding": "8px",
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                            },
                            children=[
                                dcc.Graph(figure=fig_scatter_capacidad)
                                if fig_scatter_capacidad is not None
                                else html.Div(
                                    "No hay información suficiente para mostrar el gráfico de precio vs capacidad."
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# 5. Callback de predicción


@app.callback(
    [Output("output-precio", "children"), Output("output-recomendacion", "children")],
    inputs=[Input("btn-calcular", "n_clicks")],
    state=[
        State("input-accommodates", "value"),
        State("input-bedrooms", "value"),
        State("input-bathrooms", "value"),
        State("input-beds", "value"),
        State("input-minimum-nights", "value"),
        State("input-availability-365", "value"),
        State("input-zona", "value"),
        State("input-property-type", "value"),
        State("input-room-type", "value"),
    ],
)
def actualizar_prediccion(
    n_clicks,
    accommodates,
    bedrooms,
    bathrooms,
    beds,
    minimum_nights,
    availability_365,
    zona,
    property_type,
    room_type,
):
    # Solo calculamos cuando el usuario haga clic (n_clicks > 0)
    if not n_clicks:
        return (
            "Precio sugerido: —",
            "Probabilidad de ser recomendado: —",
        )

    # Construir vector de features
    x = build_feature_vector(
        accommodates,
        bedrooms,
        bathrooms,
        beds,
        minimum_nights,
        availability_365,
        zona,
        property_type,
        room_type,
    )

    # Escalar
    x_scaled = scaler.transform(x)

    # Modelo de regresión: precio
    price_pred = float(reg_model.predict(x_scaled)[0])

    # Modelo de clasificación: probabilidad de ser recomendado
    proba = clf_model.predict_proba(x_scaled)[0]

    # Asegurarnos de tomar la columna de la clase "1"
    if 1 in clf_model.classes_:
        idx_pos = list(clf_model.classes_).index(1)
        prob_recom = float(proba[idx_pos])
    else:
        # Por seguridad: tomamos la clase con mayor probabilidad
        idx_pos = int(np.argmax(proba))
        prob_recom = float(proba[idx_pos])

    # Recortar por si acaso
    prob_recom = max(0.0, min(1.0, prob_recom))

    texto_precio = f"Precio sugerido: {price_pred:,.0f} GBP por noche."
    etiqueta = "Recomendado" if prob_recom >= 0.5 else "No recomendado"
    texto_recom = (
        f"Probabilidad de ser recomendado: {prob_recom*100:,.1f}% ({etiqueta})."
    )

    return texto_precio, texto_recom


# 6. Main

if __name__ == "__main__":
    # host 0.0.0.0 y puerto 8050 para Docker/AWS
    app.run(host="0.0.0.0", port=8050, debug=False)
