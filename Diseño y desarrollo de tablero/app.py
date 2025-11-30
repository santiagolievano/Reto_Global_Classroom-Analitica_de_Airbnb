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

# scaler y modelos entrenados (ajusta nombres si usaste otros)
scaler = joblib.load("scaler_airbnb_london.pkl")
reg_model = joblib.load("mlp_regresion_price_airbnb.pkl")
clf_model = joblib.load("mlp_clasificacion_recomended_airbnb.pkl")

# Valores por defecto: medianas para numéricas, 0 para dummies
default_values = {}
for col in FEATURE_COLS:
    if set(df[col].unique()) <= {0, 1}:   # dummy
        default_values[col] = 0
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


def build_feature_vector(accommodates, bedrooms, bathrooms, beds,
                         minimum_nights, availability_365,
                         zona, property_type, room_type):
    """
    Construye un vector de features en el mismo orden de FEATURE_COLS
    a partir de los inputs del usuario y valores por defecto.
    """
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

# Gráficas estáticas
fig_box_zona = px.box(
    df.assign(
        zona=np.select(
            [
                df.get("neighbourhood_cleansed_East", 0) == 1,
                df.get("neighbourhood_cleansed_North", 0) == 1,
                df.get("neighbourhood_cleansed_South", 0) == 1,
                df.get("neighbourhood_cleansed_West", 0) == 1,
            ],
            ["East", "North", "South", "West"],
            default="Central",
        )
    ),
    x="zona",
    y="price",
    title="Distribución de precios por zona",
)

fig_hist_recom = px.histogram(
    df,
    x="recomended",
    title="Distribución de anuncios recomendados",
    labels={"recomended": "Recomendado (1) / No (0)"},
)


# 3. Layout del tablero

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "margin": "20px"},
    children=[
        html.H1("Tablero Airbnb Londres", style={"textAlign": "center"}),

        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[
                # Panel de entrada
                html.Div(
                    style={
                        "flex": "1",
                        "border": "1px solid #ccc",
                        "borderRadius": "8px",
                        "padding": "15px",
                    },
                    children=[
                        html.H3("Características del anuncio"),

                        html.Label("Acomoda (huéspedes)"),
                        dcc.Input(
                            id="input-accommodates",
                            type="number",
                            min=1,
                            step=1,
                            value=2,
                            style={"width": "100%"},
                        ),

                        html.Label("Habitaciones"),
                        dcc.Input(
                            id="input-bedrooms",
                            type="number",
                            min=0,
                            step=1,
                            value=1,
                            style={"width": "100%"},
                        ),

                        html.Label("Baños"),
                        dcc.Input(
                            id="input-bathrooms",
                            type="number",
                            min=0,
                            step=0.5,
                            value=1,
                            style={"width": "100%"},
                        ),

                        html.Label("Camas"),
                        dcc.Input(
                            id="input-beds",
                            type="number",
                            min=0,
                            step=1,
                            value=1,
                            style={"width": "100%"},
                        ),

                        html.Label("Noches mínimas"),
                        dcc.Input(
                            id="input-minimum-nights",
                            type="number",
                            min=1,
                            step=1,
                            value=2,
                            style={"width": "100%"},
                        ),

                        html.Label("Disponibilidad 365 días"),
                        dcc.Input(
                            id="input-availability-365",
                            type="number",
                            min=0,
                            step=1,
                            value=180,
                            style={"width": "100%"},
                        ),

                        html.Br(),
                        html.Label("Zona"),
                        dcc.Dropdown(
                            id="input-zona",
                            options=[
                                {"label": "Central", "value": "Central"},
                                {"label": "East", "value": "East"},
                                {"label": "North", "value": "North"},
                                {"label": "South", "value": "South"},
                                {"label": "West", "value": "West"},
                            ],
                            value="Central",
                            clearable=False,
                        ),

                        html.Label("Tipo de propiedad"),
                        dcc.Dropdown(
                            id="input-property-type",
                            options=[
                                {"label": "Entire home/apt", "value": "Entire home/apt"},
                                {"label": "Private room", "value": "Private room"},
                                {"label": "Other", "value": "Other"},
                            ],
                            value="Entire home/apt",
                            clearable=False,
                        ),

                        html.Label("Tipo de habitación"),
                        dcc.Dropdown(
                            id="input-room-type",
                            options=[
                                {"label": "Entire home/apt", "value": "Entire home/apt"},
                                {"label": "Private room", "value": "Private room"},
                                {"label": "Shared room", "value": "Shared room"},
                                {"label": "Hotel room", "value": "Hotel room"},
                            ],
                            value="Entire home/apt",
                            clearable=False,
                        ),

                        html.Br(),
                        html.Button(
                            "Calcular precio y recomendación",
                            id="btn-calcular",
                            n_clicks=0,
                            style={"width": "100%", "marginTop": "10px"},
                        ),
                    ],
                ),

                # Panel de resultados y gráficas
                html.Div(
                    style={"flex": "2", "display": "flex", "flexDirection": "column", "gap": "20px"},
                    children=[
                        html.Div(
                            style={
                                "border": "1px solid #ccc",
                                "borderRadius": "8px",
                                "padding": "15px",
                            },
                            children=[
                                html.H3("Resultados del modelo"),
                                html.Div(id="output-precio", style={"fontSize": "18px", "marginBottom": "8px"}),
                                html.Div(id="output-recomendacion", style={"fontSize": "18px"}),
                            ],
                        ),

                        html.Div(
                            style={"display": "flex", "gap": "20px"},
                            children=[
                                html.Div(
                                    style={"flex": "1"},
                                    children=[dcc.Graph(figure=fig_box_zona)],
                                ),
                                html.Div(
                                    style={"flex": "1"},
                                    children=[dcc.Graph(figure=fig_hist_recom)],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# 4. Callback de predicción

@app.callback(
    [Output("output-precio", "children"),
     Output("output-recomendacion", "children")],
    Input("btn-calcular", "n_clicks"),
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
def actualizar_prediccion(n_clicks,
                          accommodates, bedrooms, bathrooms, beds,
                          minimum_nights, availability_365,
                          zona, property_type, room_type):
    if n_clicks is None or n_clicks == 0:
        return (
            "Ingrese los datos del anuncio y presione el botón para ver el precio sugerido.",
            "",
        )

    x = build_feature_vector(
        accommodates, bedrooms, bathrooms, beds,
        minimum_nights, availability_365,
        zona, property_type, room_type,
    )

    x_scaled = scaler.transform(x)

    # modelos sklearn
    price_pred = reg_model.predict(x_scaled)[0]
    prob_recom = clf_model.predict_proba(x_scaled)[0, 1]

    texto_precio = f"Precio sugerido: {price_pred:,.0f} GBP por noche."
    etiqueta = "Recomendado" if prob_recom >= 0.5 else "No recomendado"
    texto_recom = f"Probabilidad de ser recomendado: {prob_recom:,.1%} ({etiqueta})."

    return texto_precio, texto_recom


# 5. Main

if __name__ == "__main__":
    # host 0.0.0.0 y puerto 8050 para Docker/AWS
    app.run_server(host="0.0.0.0", port=8050, debug=False)
