import pandas as pd
import plotly.express as px
import statsmodels.api as sm

df_wiid = pd.read_excel('datasets/WIID-08SEP2026.xlsx')


df_clean = df_wiid[['country', 'year', 'gini', 'gdp']].dropna()

df_sorted = df_clean.sort_values('year')
df_first = df_sorted.groupby('country').first().reset_index()
df_last = df_sorted.groupby('country').last().reset_index()

df_tendencias = pd.DataFrame({
    'country': df_first['country'],
    'crecimiento_gdp_pct': ((df_last['gdp'] - df_first['gdp']) / df_first['gdp']) * 100,
    'cambio_gini': df_last['gini'] - df_first['gini']
}).dropna()

fig = px.scatter(
    df_tendencias,
    x='crecimiento_gdp_pct',
    y='cambio_gini',
    hover_name='country',
    hover_data={'crecimiento_gdp_pct': ':.1f', 'cambio_gini': ':.2f'},
    trendline='ols', 
    trendline_color_override='#e74c3c',
    title='Relación entre Crecimiento Económico y Cambio en la Desigualdad',
    labels={
        'crecimiento_gdp_pct': 'Crecimiento del GDP (%) entre el primer y último año',
        'cambio_gini': 'Cambio neto en el Gini (Positivo = Más desigual)'
    },
    opacity=0.6,
    template='plotly_white'
)
resultados_modelo = px.get_trendline_results(fig)
if not resultados_modelo.empty:
    r_cuadrado = resultados_modelo.iloc[0]["px_fit_results"].rsquared
    
    fig.add_annotation(
        x=0.98, y=0.98, 
        xref='paper', yref='paper',
        text=f"<b>Correlación:</b><br>R² = {r_cuadrado:.4f}",
        showarrow=False,
        font=dict(size=14, color="#c0392b"),
        bgcolor="white",
        bordercolor="#e74c3c",
        borderwidth=2,
        borderpad=10,
        xanchor='right',
        yanchor='top'
    )

fig.update_layout(
    height=600,
    margin=dict(t=80, b=50, l=50, r=50)
)

fig.write_html('regresion_gini_gdp.html', include_plotlyjs='cdn')
print("¡Gráfico generado! Abre 'regresion_gini_gdp.html' para ver el R².")


df_pip = pd.read_csv('datasets/pip.csv').drop_duplicates(subset=['country_name', 'reporting_year'])
df_clean = df_pip[['country_name', 'reporting_year', 'mean', 'headcount']].dropna()

df_sorted = df_clean.sort_values('reporting_year')
df_first = df_sorted.groupby('country_name').first().reset_index()
df_last = df_sorted.groupby('country_name').last().reset_index()

df_tendencias = pd.DataFrame({
    'country': df_first['country_name'],
    'crecimiento_ingreso_pct': ((df_last['mean'] - df_first['mean']) / df_first['mean']) * 100,
    'cambio_pobreza_puntos': (df_last['headcount'] - df_first['headcount']) * 100 
}).dropna()

fig = px.scatter(
    df_tendencias,
    x='crecimiento_ingreso_pct',
    y='cambio_pobreza_puntos',
    hover_name='country',
    hover_data={'crecimiento_ingreso_pct': ':.1f', 'cambio_pobreza_puntos': ':.2f'},
    trendline='ols', 
    title='Relación entre Crecimiento del Ingreso Medio y Cambio en la Pobreza Extrema',
    labels={
        'crecimiento_ingreso_pct': 'Crecimiento del Ingreso Medio (%) entre primer y último año',
        'cambio_pobreza_puntos': 'Cambio neto en Pobreza Extrema (Puntos %, Negativo = Menos pobreza)'
    },
    opacity=0.6,
    template='plotly_white'
)

# 5. Extraer y mostrar el R^2
resultados = px.get_trendline_results(fig)
if not resultados.empty:
    r_cuadrado = resultados.iloc[0]["px_fit_results"].rsquared
    
    fig.add_annotation(
        x=0.98, y=0.98, 
        xref='paper', yref='paper',
        text=f"<b>Correlación:</b><br>R² = {r_cuadrado:.4f}",
        showarrow=False,
        font=dict(size=14, color="#2c3e50"),
        bgcolor="white",
        bordercolor="#27ae60",
        borderwidth=2,
        borderpad=10,
        xanchor='right',
        yanchor='top'
    )

fig.update_layout(
    height=600,
    margin=dict(t=80, b=50, l=50, r=50)
)

fig.write_html('regresion_ingreso_pobreza.html', include_plotlyjs='cdn')
print("¡Gráfico generado! Abre 'regresion_ingreso_pobreza.html' para ver la correlación.")