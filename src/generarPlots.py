import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# --- 1. CARGA Y PROCESAMIENTO DE DATOS ---
df_pip = pd.read_csv('datasets/pip.csv').drop_duplicates(
    subset=['country_name', 'reporting_year']
)
df_pip['headcount_pct'] = df_pip['headcount'] * 100
df_pip['headcount_pct_log'] = df_pip['headcount_pct'].clip(lower=0.1)
df_pip = df_pip[
    ~(
        (df_pip['country_name'] == 'Costa Rica')
        & (df_pip['reporting_year'] < 1989)
    )
]

paises_objetivo = [
    'China',
    'Viet Nam',
    'Indonesia',
    'Costa Rica',
    'Panama',
    'Colombia',
]
df_trayectorias = df_pip[df_pip['country_name'].isin(paises_objetivo)].sort_values(
    ['country_name', 'reporting_year']
)

colores = {
    'China': '#e74c3c',
    'Viet Nam': '#f39c12',
    'Indonesia': '#27ae60',
    'Costa Rica': '#e10eaf',
    'Panama': '#8e44ad',
    'Colombia': '#d4ac0d',
}

df_pov = pd.read_csv('datasets/share-of-population-living-in-extreme-poverty.csv')
df_efw = pd.read_csv('datasets/efw_cc.csv')
df_pov.columns = df_pov.columns.str.strip()
df_efw.columns = df_efw.columns.str.strip()
df_pov['Country'] = (
    df_pov['Country'].astype(str).str.replace(r' \(urban\)', '', regex=True)
)
df_efw = df_efw.rename(columns={'countries': 'Country', 'year': 'Year'})

for d in (df_pov, df_efw):
  d['Year'] = pd.to_numeric(d['Year'], errors='coerce')
  d.dropna(subset=['Year'], inplace=True)
  d['Year'] = d['Year'].astype(int)
  d['Country'] = d['Country'].astype(str).str.strip()
  d['_key'] = d['Country'].str.lower()

df_merged = pd.merge(
    df_pov, df_efw.drop(columns=['Country']), on=['_key', 'Year'], how='inner'
)
df_merged['ECONOMIC FREEDOM'] = pd.to_numeric(
    df_merged['ECONOMIC FREEDOM'], errors='coerce'
)
df_merged['Share below $3 a day'] = pd.to_numeric(
    df_merged['Share below $3 a day'], errors='coerce'
)
df_merged = df_merged.dropna(
    subset=['ECONOMIC FREEDOM', 'Share below $3 a day']
)

def L(v):
  return np.log10(v)

Y_TICKVALS = [0.1, 1, 5, 15, 30, 60, 100]
Y_TICKTEXT = ['0%', '1%', '5%', '15%', '30%', '60%', '100%']
Y_RANGE_LOG = [L(0.06), L(130)]

# ==========================================
# GRÁFICO 1: LÍNEAS DE TRAYECTORIA
# ==========================================
fig1 = go.Figure()

for pais in paises_objetivo:
  df_p = df_trayectorias[df_trayectorias['country_name'] == pais].dropna(
      subset=['gini', 'headcount_pct']
  )
  if df_p.empty:
    continue

  x = df_p['gini'].values
  y_log = df_p['headcount_pct_log'].values
  y_real = df_p['headcount_pct'].values
  anios = df_p['reporting_year'].astype(int).values

  fig1.add_trace(
      go.Scatter(
          x=x.tolist(),
          y=y_log.tolist(),
          mode='lines+markers',
          name=pais,
          line=dict(width=1.5, color=colores[pais]),
          marker=dict(size=6, color=colores[pais]),
          customdata=np.stack((anios, y_real), axis=-1).tolist(),
          hovertemplate=(
              '<b>'
              + pais
              + '</b><br><br>Año: %{customdata[0]}<br><br>Gini:'
              ' %{x:.3f}<br><br>Pobreza: %{customdata[1]:.2f}%<extra></extra>'
          ),
      )
  )

  fig1.add_annotation(
      x=x[0],
      y=L(y_log[0]),
      text=str(anios[0]),
      showarrow=False,
      yshift=10,
      font=dict(color=colores[pais], size=11, family='Arial Black'),
  )
  fig1.add_annotation(
      x=x[-1],
      y=L(y_log[-1]),
      text=str(anios[-1]),
      showarrow=False,
      yshift=-10,
      font=dict(color=colores[pais], size=11, family='Arial Black'),
  )
  
  fig1.add_annotation( 
      x=x[-1],
      y=L(y_log[-1]),
      text=f"<b>{pais}</b>",
      showarrow=True,
      arrowhead=1,
      arrowsize=1,
      arrowwidth=1.5,
      arrowcolor=colores[pais],
      ax=50,             
      ay=0,              
      font=dict(color=colores[pais], size=12),
      bgcolor='white',   
      bordercolor=colores[pais], 
      borderwidth=2,
      borderpad=4
  )

fig1.update_layout( 
    title='1. Apertura Económica: Impacto en la Pobreza Absoluta y Desigualdad',
    xaxis_title='Índice de Gini (Mayor = Más desigual)',
    yaxis_title='Población bajo la línea de pobreza extrema (%)',
    yaxis=dict(
        type='log', tickvals=Y_TICKVALS, ticktext=Y_TICKTEXT, range=Y_RANGE_LOG
    ),
    template='plotly_white',
    height=700,
    showlegend=False,
    margin=dict(t=80, b=70, r=80),
)

# ==========================================
# GRÁFICO 2: LÍNEAS DE TRAYECTORIA
# ==========================================
fig2 = go.Figure()

for pais in paises_objetivo:
  df_p = df_trayectorias[df_trayectorias['country_name'] == pais].dropna(
      subset=['mean', 'headcount_pct']
  )
  if df_p.empty:
    continue

  x = df_p['mean'].values
  y_log = df_p['headcount_pct_log'].values
  y_real = df_p['headcount_pct'].values
  anios = df_p['reporting_year'].astype(int).values

  fig2.add_trace(
      go.Scatter(
          x=x.tolist(),
          y=y_log.tolist(),
          mode='lines+markers',
          name=pais,
          line=dict(width=1.5, color=colores[pais]),
          marker=dict(size=6, color=colores[pais]),
          customdata=np.stack((anios, y_real), axis=-1).tolist(),
          hovertemplate=(
              '<b>'
              + pais
              + '</b><br><br>Año: %{customdata[0]}<br><br>Ingreso Medio'
              ' Diario: $%{x:.2f}<br><br>Pobreza:'
              ' %{customdata[1]:.2f}%<extra></extra>'
          ),
      )
  )

  fig2.add_annotation(
      x=x[0],
      y=L(y_log[0]),
      text=str(anios[0]),
      showarrow=False,
      xshift=-15,
      font=dict(color=colores[pais], size=11, family='Arial Black'),
  )
  fig2.add_annotation(
      x=x[-1],
      y=L(y_log[-1]),
      text=str(anios[-1]),
      showarrow=False,
      xshift=15,
      font=dict(color=colores[pais], size=11, family='Arial Black'),
  )

  fig2.add_annotation( 
      x=x[-1],
      y=L(y_log[-1]),
      text=f"<b>{pais}</b>",
      showarrow=True,
      arrowhead=1,
      arrowsize=1,
      arrowwidth=1.5,
      arrowcolor=colores[pais],
      ax=50,             
      ay=0,              
      font=dict(color=colores[pais], size=12),
      bgcolor='white',   
      bordercolor=colores[pais], 
      borderwidth=2,
      borderpad=4
  )

fig2.update_layout(
    title='2. El Motor del Bienestar: Ingreso Medio vs Pobreza Extrema',
    xaxis_title='Ingreso Medio Diario (Dólares PPA)',
    yaxis_title='Población bajo la línea de pobreza extrema (%)',
    yaxis=dict(
        type='log', tickvals=Y_TICKVALS, ticktext=Y_TICKTEXT, range=Y_RANGE_LOG
    ),
    template='plotly_white',
    height=700,
    showlegend=False,
    margin=dict(t=80, b=70, r=80),
)

# ==========================================
# GRÁFICO 3: LIBERTAD ECONÓMICA
# ==========================================
fig3 = px.scatter(
    df_merged,
    x='ECONOMIC FREEDOM',
    y='Share below $3 a day',
    hover_name='Country',
    hover_data={
        'Year': True,
        'ECONOMIC FREEDOM': ':.2f',
        'Share below $3 a day': ':.2f',
    },
    trendline='ols',
    trendline_color_override='#e74c3c',
    opacity=0.6,
    title='3. Impacto de la Libertad Económica en la Pobreza Extrema',
    labels={
        'ECONOMIC FREEDOM': 'Índice de Libertad Económica',
        'Share below $3 a day': 'Población bajo $3 al día (%)',
    },
    template='plotly_white',
)
fig3.update_traces(
    marker=dict(size=7, color='#2c3e50'), selector=dict(mode='markers')
)
fig3.update_layout(height=650, showlegend=False)


# ==========================================
# GRÁFICO 4: REGRESIÓN GINI VS CRECIMIENTO
# ==========================================
df_wiid = pd.read_excel('datasets/WIID-08SEP2026.xlsx')
df_clean_gini = df_wiid[['country', 'year', 'gini', 'gdp']].dropna()

df_sorted_gini = df_clean_gini.sort_values('year')
df_first_gini = df_sorted_gini.groupby('country').first().reset_index()
df_last_gini = df_sorted_gini.groupby('country').last().reset_index()

df_tendencias_gini = pd.DataFrame({
    'country': df_first_gini['country'],
    'crecimiento_gdp_pct': ((df_last_gini['gdp'] - df_first_gini['gdp']) / df_first_gini['gdp']) * 100,
    'cambio_gini': df_last_gini['gini'] - df_first_gini['gini']
}).dropna()

fig4 = px.scatter(
    df_tendencias_gini,
    x='crecimiento_gdp_pct',
    y='cambio_gini',
    hover_name='country',
    hover_data={'crecimiento_gdp_pct': ':.1f', 'cambio_gini': ':.2f'},
    trendline='ols', 
    trendline_color_override='#e74c3c',
    title='4. Relación entre Crecimiento Económico y Cambio en la Desigualdad',
    labels={
        'crecimiento_gdp_pct': 'Crecimiento del GDP (%) entre el primer y último año',
        'cambio_gini': 'Cambio neto en el Gini (Positivo = Más desigual)'
    },
    opacity=0.6,
    template='plotly_white'
)

resultados_modelo4 = px.get_trendline_results(fig4)
if not resultados_modelo4.empty:
    r_cuadrado4 = resultados_modelo4.iloc[0]["px_fit_results"].rsquared
    fig4.add_annotation(
        x=0.98, y=0.98, 
        xref='paper', yref='paper',
        text=f"<b>Correlación:</b><br>R² = {r_cuadrado4:.4f}",
        showarrow=False,
        font=dict(size=14, color="#c0392b"),
        bgcolor="white",
        bordercolor="#e74c3c",
        borderwidth=2,
        borderpad=10,
        xanchor='right',
        yanchor='top'
    )

fig4.update_layout(height=600, margin=dict(t=80, b=50, l=50, r=50))


# ==========================================
# GRÁFICO 5: REGRESIÓN INGRESO VS POBREZA
# ==========================================
df_clean_pov = df_pip[['country_name', 'reporting_year', 'mean', 'headcount']].dropna()

df_sorted_pov = df_clean_pov.sort_values('reporting_year')
df_first_pov = df_sorted_pov.groupby('country_name').first().reset_index()
df_last_pov = df_sorted_pov.groupby('country_name').last().reset_index()

df_tendencias_pov = pd.DataFrame({
    'country': df_first_pov['country_name'],
    'crecimiento_ingreso_pct': ((df_last_pov['mean'] - df_first_pov['mean']) / df_first_pov['mean']) * 100,
    'cambio_pobreza_puntos': (df_last_pov['headcount'] - df_first_pov['headcount']) * 100 
}).dropna()

fig5 = px.scatter(
    df_tendencias_pov,
    x='crecimiento_ingreso_pct',
    y='cambio_pobreza_puntos',
    hover_name='country',
    hover_data={'crecimiento_ingreso_pct': ':.1f', 'cambio_pobreza_puntos': ':.2f'},
    trendline='ols', 
    trendline_color_override='#27ae60',
    title='5. Relación entre Crecimiento del Ingreso Medio y Cambio en la Pobreza Extrema',
    labels={
        'crecimiento_ingreso_pct': 'Crecimiento del Ingreso Medio (%) entre primer y último año',
        'cambio_pobreza_puntos': 'Cambio neto en Pobreza Extrema (Puntos %, Negativo = Menos pobreza)'
    },
    opacity=0.6,
    template='plotly_white'
)

resultados5 = px.get_trendline_results(fig5)
if not resultados5.empty:
    r_cuadrado5 = resultados5.iloc[0]["px_fit_results"].rsquared
    fig5.add_annotation(
        x=0.98, y=0.98, 
        xref='paper', yref='paper',
        text=f"<b>Correlación:</b><br>R² = {r_cuadrado5:.4f}",
        showarrow=False,
        font=dict(size=14, color="#2c3e50"),
        bgcolor="white",
        bordercolor="#27ae60",
        borderwidth=2,
        borderpad=10,
        xanchor='right',
        yanchor='top'
    )

fig5.update_layout(height=600, margin=dict(t=80, b=50, l=50, r=50))


# --- EXPORTAR A UNA SOLA PÁGINA HTML ---

html_fig1 = fig1.to_html(full_html=False, include_plotlyjs='cdn')
html_fig2 = fig2.to_html(full_html=False, include_plotlyjs=False)
html_fig3 = fig3.to_html(full_html=False, include_plotlyjs=False)
html_fig4 = fig4.to_html(full_html=False, include_plotlyjs=False)
html_fig5 = fig5.to_html(full_html=False, include_plotlyjs=False)

plantilla_html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Reporte Desigualdad vs Crecimiento</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f4f7f6;
            margin: 0;
            padding: 40px;
            color: #333;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .contenedor-grafico {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 40px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }}
        .disclaimer {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            padding: 20px;
            margin-bottom: 20px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            font-size: 0.95em;
            color: #555;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <h1>Reporte: Desigualdad vs Crecimiento</h1>
    
    <div class="contenedor-grafico">
        {html_fig1}
    </div>
    
    <div class="contenedor-grafico">
        {html_fig2}
    </div>

    
    
    <div class="contenedor-grafico">
        {html_fig3}
    </div>

    <div class="contenedor-grafico">
        {html_fig4}
    </div>

    <div class="contenedor-grafico">
        {html_fig5}
    </div>
    <div class="disclaimer">
        <p><strong>Nota sobre la visualización (Gráficos 1 y 2):</strong> El eje Y se expande a medida que desciende. Esto ocurre porque muchos países llevan bastante tiempo con poca pobreza, lo que aplanaría visualmente los gráficos dificultando su lectura. Además, refleja la dificultad real del progreso: es estadísticamente más difícil pasar de 50% a 10% de pobreza, que de 10% a 0%. Cada punto porcentual reducido en los niveles bajos requiere un esfuerzo monumental, por lo que resaltarlo visualmente aporta contexto crítico.</p>
        <p>Los países seleccionados corresponden a modelos de apertura al mercado internacional y alta libertad económica adoptados alrededor de los años 70s-80s (época donde inicia la disponibilidad de datos de este dataset). Países como Nueva Zelanda, Dinamarca, Suecia o Alemania de posguerra siguieron modelos similares, pero sus transformaciones fueron mucho más antiguas o carecían de los datos estandarizados necesarios para esta serie temporal.</p>
    </div>

    <div class="disclaimer">
        <p><strong>El Motor del Bienestar:</strong> En el Gráfico 2 y 5, puede parecer una obviedad que "a más ingresos, menos pobreza". Sin embargo, el propósito fundamental de esta métrica es evidenciar que una variable como el ingreso medio —el cual está correlacionado casi en un 1:1 con el crecimiento económico general— es el verdadero motor en la erradicación de la pobreza absoluta a nivel nacional.</p>
    </div>
    
</body>
</html>
"""

with open('reporte_desigualdad_vs_crecimiento.html', 'w', encoding='utf-8') as archivo:
    archivo.write(plantilla_html)

print("¡Reporte completo generado con éxito! Abre 'reporte_desigualdad_vs_crecimiento.html' para ver los 5 gráficos.")