
import os
from dotenv import load_dotenv
import lyricsgenius

# Cargar variables de entorno desde .env
load_dotenv()
GENIUS_TOKEN = os.getenv("GENIUS_TOKEN")


# =============== CONFIG ===============
OUTPUT_FILE = "canciones.txt"
ARTISTS = [
    ("Enjambre", 20),
    ("Caifanes", 20),
    ("Los Bunkers", 20),
    ("El cuarteto de Nos", 20),
    ("Pxndx", 20),
    ("Zoé", 20),
    ("Café Tacvba", 20),
    ("Soda Stereo", 20),
    ("Héroes del Silencio", 20),
    ("León Larregui", 20),
    ("Julieta Venegas", 20),
    ("Natalia Lafourcade", 20),
    ("Maná", 20),
    ("Siddhartha", 20),
    ("José José", 20),
]
# ======================================

def main():
    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Cover)"],
        remove_section_headers=True,
        timeout=15,
    )
    genius.verbose = False  # menos spam en consola

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for artist_name, n_songs in ARTISTS:
            print(f"🔎 Descargando {n_songs} canciones de {artist_name}...")
            artist = genius.search_artist(artist_name, max_songs=n_songs, sort="popularity")
            if not artist:
                print(f"⚠️ No se pudo obtener canciones de {artist_name}")
                continue

            for song in artist.songs:
                f.write("<|startsong|>\n")
                f.write(song.lyrics.strip() + "\n")
                f.write("<|endsong|>\n\n")

    print(f"\n✅ Letras guardadas en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()