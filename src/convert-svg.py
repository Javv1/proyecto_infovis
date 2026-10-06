import json
import re
import xml.etree.ElementTree as ET
import plotly.graph_objects as go


def agregar_ids_a_svg(svg_filepath, data):
    """Añade los atributos id='puntosonoro_...' a los elementos del SVG exportado."""
    # Registrar namespace SVG para no romper la estructura del archivo
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(svg_filepath)
    root = tree.getroot()

    # Extraer puntos de las trazas de Plotly
    puntos_datos = []
    for trace in data:
        x_vals = trace.get("x", [])
        y_vals = trace.get("y", [])
        text_vals = trace.get(
            "text", trace.get("hovertext", [])
        )  # Para país/año

        for i in range(len(y_vals)):
            pobreza = y_vals[i]
            # Si text es una lista usará la posición, si no usa un valor por defecto
            info_extra = (
                text_vals[i]
                if isinstance(text_vals, list) and i < len(text_vals)
                else "Dato"
            )
            puntos_datos.append(
                {"pobreza": pobreza, "pais": info_extra, "anio": x_vals[i]}
            )

    # Buscar elementos gráficos de puntos dentro del SVG (clase 'point' en Plotly)
    idx_punto = 0
    for elem in root.iter():
        # Plotly guarda los puntos de dispersión en elementos <path class="point">
        if elem.tag.endswith("path") and "point" in elem.attrib.get(
            "class", ""
        ):
            if idx_punto < len(puntos_datos):
                d = puntos_datos[idx_punto]
                # Formatear el ID compatible con tu HTML
                node_id = f"puntosonoro_pobreza-{d['pobreza']}_pais-{d['pais']}_anio-{d['anio']}"
                elem.set("id", node_id)
                idx_punto += 1

    tree.write(svg_filepath, encoding="utf-8", xml_declaration=True)


def extraer_graficos_svg(html_filepath):
    with open(html_filepath, "r", encoding="utf-8") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    pattern = re.compile(r'Plotly\.newPlot\s*\(\s*["\']([^"\']+)["\']\s*,\s*')

    count = 0
    for match in pattern.finditer(content):
        div_id = match.group(1)
        start_pos = match.end()

        try:
            data, data_len = decoder.raw_decode(content[start_pos:])
            remaining = content[start_pos + data_len :].lstrip()

            if remaining.startswith(","):
                remaining = remaining[1:].lstrip()

            layout, _ = decoder.raw_decode(remaining)

            count += 1
            fig = go.Figure(data=data, layout=layout)

            # Quitar títulos del gráfico
            fig.update_layout(title="")

            output_filename = f"grafico_{count}_{div_id}.svg"

            # 1. Exportar gráfico estático
            fig.write_image(output_filename, format="svg")

            # 2. Inyectar IDs para interacción con el sonido en tu HTML
            agregar_ids_a_svg(output_filename, data)

            print(
                f"✔ Guardado y preparado para sonido: {output_filename}"
            )

        except Exception as e:
            print(f"✖ Error procesando el gráfico '{div_id}': {e}")


if __name__ == "__main__":
    extraer_graficos_svg("reporte_desarrollo.html")