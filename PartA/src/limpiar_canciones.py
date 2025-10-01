import re
import os

import re

def limpiar_letra(texto: str) -> str:
    """
    Limpia una letra de canción preservando saltos de línea y estructura.
    - Quita notas escénicas (*...*).
    - Quita encabezados [Verso], [Coro], [Puente: ...], etc.
    - Quita líneas basura ('Embed', 'You might also like', 'Letra de ...').
    - Normaliza espacios solo dentro de cada línea.
    - Convierte a minúsculas.
    - Preserva saltos de línea dobles para estrofas.
    """
    # 1. Eliminar notas entre *...*
    texto = re.sub(r"\*[^*]+\*", " ", texto)

    # 2. Eliminar encabezados de secciones tipo [Verso 1], [Coro], [Puente: ...], etc.
    texto = re.sub(r"\[.*?\]", "", texto)

    # 3. Eliminar líneas basura típicas de Genius y créditos
    bad_tokens = [
        "embed", "you might also like", "letra de", "lyrics", "here we go"
    ]
    # Preserva saltos dobles
    lineas = texto.splitlines()
    lineas_limpias = []
    for linea in lineas:
        if not any(bt in linea.lower() for bt in bad_tokens):
            # Normaliza espacios en cada línea
            linea_limpia = re.sub(r"\s+", " ", linea).strip()
            lineas_limpias.append(linea_limpia)
    # Reconstruye el texto, preservando saltos dobles
    texto_limpio = "\n".join(lineas_limpias)
    texto_limpio = re.sub(r'\n{3,}', '\n\n', texto_limpio)  # Máximo dos saltos seguidos

    return texto_limpio


def limpiar_archivo(entrada="canciones.txt", salida="canciones_clean.txt", min_len=20):
    """
    Limpia todas las canciones en un archivo con delimitadores <|startsong|> y <|endsong|>.
    Guarda el resultado en un nuevo archivo.
    """
    with open(entrada, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = re.split(r"<\|startsong\|>", contenido)
    canciones_limpias = []

    for bloque in bloques:
        if "<|endsong|>" not in bloque:
            continue
        letra = bloque.split("<|endsong|>")[0].strip()
        letra_limpia = limpiar_letra(letra)

        # Descartar canciones demasiado cortas
        if len(letra_limpia) >= min_len:
            canciones_limpias.append(
                "<|startsong|>\n" + letra_limpia + "\n<|endsong|>\n"
            )

    with open(salida, "w", encoding="utf-8") as f:
        f.write("\n".join(canciones_limpias))

    print(f"✅ Archivo limpio guardado en {salida}")
    print(f"📊 Canciones finales: {len(canciones_limpias)}")


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PATH = os.path.join(base_path, "data", "canciones.txt")
    print(f"Usando archivo de entrada: {PATH}")
    limpiar_archivo(entrada=PATH)
