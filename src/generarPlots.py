import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CARGA Y PROCESAMIENTO DE DATOS ---
df_pip = pd.read_csv('pip.csv').drop_duplicates(
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

df_pov = pd.read_csv('share-of-population-living-in-extreme-poverty.csv')
df_efw = pd.read_csv('efw_cc.csv')
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


# --- FUNCIÓN POST-PROCESADORA XML DE SVG ---
def inyectar_ids_en_svg(svg_filename, datos_puntos):
  """Asigna atributos id='puntosonoro_...' únicamente a los puntos de datos

  (sin afectar las flechas de trayectoria).
  """
  ET.register_namespace('', 'http://www.w3.org/2000/svg')
  tree = ET.parse(svg_filename)
  root = tree.getroot()

  idx_punto = 0
  for elem in root.iter():
    tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
    # Filtrar estrictamente solo los marcadores de puntos
    if tag_name in ['path', 'circle'] and 'point' in elem.attrib.get(
        'class', ''
    ):
      if idx_punto < len(datos_puntos):
        dp = datos_puntos[idx_punto]
        id_str = f"puntosonoro_pobreza-{dp['pobreza']:.2f}_pais-{dp['pais']}_anio-{dp['anio']}"
        elem.set('id', id_str)
        idx_punto += 1

  tree.write(svg_filename, encoding='utf-8', xml_declaration=True)


# ==========================================
# GRÁFICO 1: CON FLECHAS DE TRAYECTORIA
# ==========================================
fig1 = go.Figure()
puntos_fig1 = []

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
          mode='markers',
          name=pais,
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

  # Flechas de trayectoria
  for i in range(len(x) - 1):
    fig1.add_annotation(
        x=x[i + 1],
        y=L(y_log[i + 1]),
        ax=x[i],
        ay=L(y_log[i]),
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=colores[pais],
    )

  # Etiquetas de año inicial y final
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

  for i in range(len(x)):
    puntos_fig1.append({'pobreza': y_real[i], 'pais': pais, 'anio': anios[i]})

fig1.update_layout(
    title='1. Apertura Económica: Impacto en la Pobreza Absoluta y Desigualdad',
    xaxis_title='Índice de Gini (Mayor = Más desigual)',
    yaxis_title='Población bajo la línea de pobreza extrema (%)',
    yaxis=dict(
        type='log', tickvals=Y_TICKVALS, ticktext=Y_TICKTEXT, range=Y_RANGE_LOG
    ),
    template='plotly_white',
    height=700,
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        title_text='',
    ),
    margin=dict(t=110, b=70),
)

fig1.write_image('grafico_1_pobreza_desigualdad.svg', format='svg')
inyectar_ids_en_svg('grafico_1_pobreza_desigualdad.svg', puntos_fig1)


# ==========================================
# GRÁFICO 2: CON FLECHAS DE TRAYECTORIA
# ==========================================
fig2 = go.Figure()
puntos_fig2 = []

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
          mode='markers',
          name=pais,
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

  # Flechas de trayectoria
  for i in range(len(x) - 1):
    fig2.add_annotation(
        x=x[i + 1],
        y=L(y_log[i + 1]),
        ax=x[i],
        ay=L(y_log[i]),
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=colores[pais],
    )

  # Etiquetas de año inicial y final
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

  for i in range(len(x)):
    puntos_fig2.append({'pobreza': y_real[i], 'pais': pais, 'anio': anios[i]})

fig2.update_layout(
    title='2. El Motor del Bienestar: Ingreso Medio vs Pobreza Extrema',
    xaxis_title='Ingreso Medio Diario (Dólares PPA)',
    yaxis_title='Población bajo la línea de pobreza extrema (%)',
    yaxis=dict(
        type='log', tickvals=Y_TICKVALS, ticktext=Y_TICKTEXT, range=Y_RANGE_LOG
    ),
    template='plotly_white',
    height=700,
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        title_text='',
    ),
    margin=dict(t=110, b=70),
)

fig2.write_image('grafico_2_ingreso_pobreza.svg', format='svg')
inyectar_ids_en_svg('grafico_2_ingreso_pobreza.svg', puntos_fig2)


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

puntos_fig3 = []
for _, row in df_merged.iterrows():
  puntos_fig3.append({
      'pobreza': row['Share below $3 a day'],
      'pais': str(row['Country']).replace(' ', '_'),
      'anio': row['Year'],
  })

# Reemplaza la exportación SVG y la función XML por esto:

# Guardar Gráfico 1
fig1.write_html(
    'grafico_1_pobreza_desigualdad.html',
    include_plotlyjs='cdn',  # Carga la librería Plotly desde red para ahorrar espacio
    full_html=True,
)

# Guardar Gráfico 2
fig2.write_html('grafico_2_ingreso_pobreza.html', include_plotlyjs='cdn')

# Guardar Gráfico 3
fig3.write_html(
    'grafico_3_libertad_economica.html', include_plotlyjs='cdn'
)