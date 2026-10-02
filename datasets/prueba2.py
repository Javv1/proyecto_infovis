import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

df_pip = pd.read_csv('pip.csv')
df_pip = df_pip.drop_duplicates(subset=['country_name', 'reporting_year'])
df_pip['headcount_pct'] = df_pip['headcount'] * 100

df_pip['headcount_pct_log'] = df_pip['headcount_pct'].clip(lower=0.1)

df_pip = df_pip[~((df_pip['country_name'] == 'Costa Rica') & (df_pip['reporting_year'] < 1989))]

paises_objetivo = ['China', 'Viet Nam', 'Indonesia', 'Costa Rica', 'Panama', 'Colombia']
df_trayectorias = df_pip[df_pip['country_name'].isin(paises_objetivo)].sort_values(['country_name', 'reporting_year'])

colores = {
    'China': '#e74c3c', 'Viet Nam': '#f39c12', 'Indonesia': '#27ae60',
    'Costa Rica': '#e10eaf', 'Panama': '#8e44ad', 'Colombia': '#d4ac0d'
}

df_pov = pd.read_csv('share-of-population-living-in-extreme-poverty.csv')
df_efw = pd.read_csv('efw_cc.csv')

df_pov.columns = df_pov.columns.str.strip()
df_efw.columns = df_efw.columns.str.strip()

df_pov['Country'] = df_pov['Country'].astype(str).str.replace(r' \(urban\)', '', regex=True)
df_efw = df_efw.rename(columns={'countries': 'Country', 'year': 'Year'})

for d in (df_pov, df_efw):
    d['Year'] = pd.to_numeric(d['Year'], errors='coerce')
    d.dropna(subset=['Year'], inplace=True)
    d['Year'] = d['Year'].astype(int)
    d['Country'] = d['Country'].astype(str).str.strip()
    d['_key'] = d['Country'].str.lower()

df_merged = pd.merge(
    df_pov, df_efw.drop(columns=['Country']),
    on=['_key', 'Year'], how='inner'
)

df_merged['ECONOMIC FREEDOM'] = pd.to_numeric(df_merged['ECONOMIC FREEDOM'], errors='coerce')
df_merged['Share below $3 a day'] = pd.to_numeric(df_merged['Share below $3 a day'], errors='coerce')
df_merged = df_merged.dropna(subset=['ECONOMIC FREEDOM', 'Share below $3 a day'])

print(f"Filas tras el merge (Gráfico 3): {len(df_merged)}")
if df_merged.empty:
    print("⚠ El merge quedó vacío. Revisa nombres de países/años en ambos CSV.")
    print("Ejemplo pov:", df_pov['Country'].unique()[:10])
    print("Ejemplo efw:", df_efw['Country'].unique()[:10])


def L(v):
    return np.log10(v)

Y_TICKVALS = [0.1, 1, 5, 15, 30, 60, 100]
Y_TICKTEXT = ['0%', '1%', '5%', '15%', '30%', '60%', '100%']
Y_RANGE_LOG = [L(0.06), L(130)]



fig1 = go.Figure()

for pais in paises_objetivo:
    df_p = df_trayectorias[df_trayectorias['country_name'] == pais].dropna(subset=['gini', 'headcount_pct'])
    if df_p.empty:
        continue

    x = df_p['gini'].values
    y_log = df_p['headcount_pct_log'].values
    y_real = df_p['headcount_pct'].values
    anios = df_p['reporting_year'].astype(int).values

    fig1.add_trace(go.Scatter(
        x=x.tolist(), y=y_log.tolist(), mode='markers', name=pais,
        marker=dict(size=6, color=colores[pais]),
        customdata=np.stack((anios, y_real), axis=-1).tolist(),
        hovertemplate="<b>" + pais + "</b><br><br>Año: %{customdata[0]}<br><br>Gini: %{x:.3f}<br><br>Pobreza: %{customdata[1]:.2f}%<extra></extra>"
    ))

    for i in range(len(x) - 1):
        fig1.add_annotation(
            x=x[i + 1], y=L(y_log[i + 1]), ax=x[i], ay=L(y_log[i]),
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=colores[pais]
        )


    fig1.add_annotation(x=x[0], y=L(y_log[0]), text=str(anios[0]), showarrow=False, yshift=10,
                        font=dict(color=colores[pais], size=11, family="Arial Black"))
    fig1.add_annotation(x=x[-1], y=L(y_log[-1]), text=str(anios[-1]), showarrow=False, yshift=-10,
                        font=dict(color=colores[pais], size=11, family="Arial Black"))

fig1.update_layout(
    title="1. Apertura Económica: Impacto en la Pobreza Absoluta y Desigualdad",
    xaxis_title="Índice de Gini (Mayor = Más desigual)",
    yaxis_title="Población bajo la línea de pobreza extrema (%)",
    yaxis=dict(type="log", tickvals=Y_TICKVALS, ticktext=Y_TICKTEXT, range=Y_RANGE_LOG),
    template="plotly_white",
    height=700,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title_text=""),
    margin=dict(t=110, b=70)
)


fig2 = go.Figure()

for pais in paises_objetivo:
    df_p = df_trayectorias[df_trayectorias['country_name'] == pais].dropna(subset=['mean', 'headcount_pct'])
    if df_p.empty:
        continue

    x = df_p['mean'].values  # INGRESO MEDIO DIARIO (Crecimiento Económico)
    y_log = df_p['headcount_pct_log'].values
    y_real = df_p['headcount_pct'].values
    anios = df_p['reporting_year'].astype(int).values

    fig2.add_trace(go.Scatter(
        x=x.tolist(), y=y_log.tolist(), mode='markers', name=pais,
        marker=dict(size=6, color=colores[pais]),
        customdata=np.stack((anios, y_real), axis=-1).tolist(),
        hovertemplate="<b>" + pais + "</b><br><br>Año: %{customdata[0]}<br><br>Ingreso Medio Diario: $%{x:.2f}<br><br>Pobreza: %{customdata[1]:.2f}%<extra></extra>"
    ))

    for i in range(len(x) - 1):
        fig2.add_annotation(
            x=x[i + 1], y=L(y_log[i + 1]), ax=x[i], ay=L(y_log[i]),
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=colores[pais]
        )

    fig2.add_annotation(x=x[0], y=L(y_log[0]), text=str(anios[0]), showarrow=False, xshift=-15,
                        font=dict(color=colores[pais], size=11, family="Arial Black"))
    fig2.add_annotation(x=x[-1], y=L(y_log[-1]), text=str(anios[-1]), showarrow=False, xshift=15,
                        font=dict(color=colores[pais], size=11, family="Arial Black"))

fig2.update_layout(
    title="2. El Motor del Bienestar: Ingreso Medio vs Pobreza Extrema",
    xaxis_title="Ingreso Medio Diario (Dólares PPA)",
    yaxis_title="Población bajo la línea de pobreza extrema (%)",
    yaxis=dict(type="log", tickvals=Y_TICKVALS, ticktext=Y_TICKTEXT, range=Y_RANGE_LOG),
    template="plotly_white",
    height=700,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title_text=""),
    margin=dict(t=110, b=70)
)


fig3 = px.scatter(
    df_merged, x='ECONOMIC FREEDOM', y='Share below $3 a day',
    hover_name='Country',
    hover_data={'Year': True, 'ECONOMIC FREEDOM': ':.2f', 'Share below $3 a day': ':.2f'},
    trendline='ols', trendline_color_override='#e74c3c', opacity=0.6,
    title='3. Impacto de la Libertad Económica en la Pobreza Extrema',
    labels={'ECONOMIC FREEDOM': 'Índice de Libertad Económica',
            'Share below $3 a day': 'Población bajo $3 al día (%)'},
    template='plotly_white'
)
fig3.update_traces(marker=dict(size=7, color='#2c3e50'), selector=dict(mode='markers'))
fig3.update_layout(height=650, showlegend=False)


html_fig1 = fig1.to_html(full_html=False, include_plotlyjs='cdn')
html_fig2 = fig2.to_html(full_html=False, include_plotlyjs=False)
html_fig3 = fig3.to_html(full_html=False, include_plotlyjs=False)

html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Interactivo: Pobreza y Apertura</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 40px 20px;
            background-color: #f8f9fa;
            color: #2c3e50;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            font-size: 2.2em;
            margin-bottom: 10px;
            color: #1a252f;
        }}
        .subtitle {{
            text-align: center;
            font-size: 1.1em;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .countries {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .countries span {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 6px 14px;
            margin: 4px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }}
        .chart-container {{
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }}
        .note {{
            text-align: center;
            font-style: italic;
            color: #7f8c8d;
            margin: 10px 0 30px 0;
            font-size: 0.95em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>El Libre Mercado, la Erradicación de la Pobreza y la mentira de la igualdad</h1>
        <p class="subtitle">Análisis empírico de trayectorias nacionales</p>

        <div class="countries">
            <strong>Países Analizados: </strong>
            <span>China</span>
            <span>Viet Nam</span>
            <span>Indonesia</span>
            <span>Costa Rica</span>
            <span>Panamá</span>
            <span>Colombia</span>
        </div>

        <div class="chart-container">
            {html_fig1}
        </div>

        <p class="note">El ingreso medio diario opera como el mejor indicador del tamaño de la economía familiar real (crecimiento).</p>

        <div class="chart-container">
            {html_fig2}
        </div>

        <div class="chart-container">
            {html_fig3}
        </div>
    </div>
</body>
</html>
"""

with open('reporte_desarrollo.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("Dashboard generado exitosamente: reporte_desarrollo.html")