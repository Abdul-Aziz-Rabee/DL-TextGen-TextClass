import re

def limpiar_letra(texto: str) -> str:
    """
    Aplica reglas de limpieza a una letra de canción.
    - Quita notas escénicas (*...*).
    - Quita secciones [Verso], [Coro], etc.
    - Quita 'Embed', 'You might also like', números sueltos.
    - Normaliza espacios y saltos de línea.
    - Convertir a minúsculas.
    """
    # 1. Eliminar notas entre *...*
    texto = re.sub(r"\*[^*]+\*", " ", texto)

    # 2. Eliminar encabezados de secciones tipo [Verso 1], [Coro]
    texto = re.sub(r"\[.*?\]", " ", texto)

    # 3. Eliminar líneas basura típicas de Genius
    bad_tokens = ["embed", "you might also like"]
    lineas = []
    for linea in texto.splitlines():
        if not any(bt in linea.lower() for bt in bad_tokens):
            lineas.append(linea)
    texto = "\n".join(lineas)

    # 4. Normalizar espacios en blanco
    texto = re.sub(r"\s+", " ", texto).strip()
    # 5. Convertir a minúsculas
    texto = texto.lower()

    return texto


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
    limpiar_archivo()
