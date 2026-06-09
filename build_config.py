#!/usr/bin/env python3
"""
build_config.py
Parsea todos los .pgw en /layers, calcula bounding boxes y genera config.json.
Convierte UTM EPSG:32614 -> WGS84 sin dependencias externas.
Uso: python build_config.py
"""

import json
import math
import struct
from pathlib import Path
import datetime

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ── UTM → WGS84 (sin pyproj) ──────────────────────────────────────────────
# Parámetros GRS80 / WGS84
_a  = 6378137.0          # semieje mayor
_f  = 1 / 298.257223563  # achatamiento
_b  = _a * (1 - _f)
_e2 = 1 - (_b/_a)**2
_e  = math.sqrt(_e2)
_k0 = 0.9996             # factor de escala UTM
_E0 = 500000.0           # false easting

def utm_to_wgs84(easting, northing, zone=14, northern=True):
    """
    Convierte coordenadas UTM (zona 14N por defecto) a WGS84 (lon, lat) en grados.
    Implementación directa de la formula de Karney / USGS.
    """
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)  # meridiano central

    x = easting  - _E0
    y = northing if northern else northing - 10_000_000.0

    M  = y / _k0
    mu = M / (_a * (1 - _e2/4 - 3*_e2**2/64 - 5*_e2**3/256))

    e1 = (1 - math.sqrt(1 - _e2)) / (1 + math.sqrt(1 - _e2))
    phi1 = (mu
            + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96) * math.sin(6*mu)
            + (1097*e1**4/512) * math.sin(8*mu))

    N1   = _a / math.sqrt(1 - _e2 * math.sin(phi1)**2)
    T1   = math.tan(phi1)**2
    C1   = (_e2 / (1 - _e2)) * math.cos(phi1)**2
    R1   = _a * (1 - _e2) / (1 - _e2 * math.sin(phi1)**2)**1.5
    D    = x / (N1 * _k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2/2
        - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*(_e2/(1-_e2))) * D**4/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*(_e2/(1-_e2)) - 3*C1**2) * D**6/720
    )
    lon = lon0 + (
        D
        - (1 + 2*T1 + C1) * D**3/6
        + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*(_e2/(1-_e2)) + 24*T1**2) * D**5/120
    ) / math.cos(phi1)

    return math.degrees(lon), math.degrees(lat)


# ── PNG DIMENSIONS ─────────────────────────────────────────────────────────
def read_png_dimensions(path):
    with open(path, 'rb') as f:
        if f.read(8) != b'\x89PNG\r\n\x1a\n':
            raise ValueError(f"No es PNG: {path}")
        f.read(4)
        if f.read(4) != b'IHDR':
            raise ValueError(f"Sin IHDR: {path}")
        w = struct.unpack('>I', f.read(4))[0]
        h = struct.unpack('>I', f.read(4))[0]
    return w, h


# ── PGW PARSER ─────────────────────────────────────────────────────────────
def parse_pgw(pgw_path):
    lines = Path(pgw_path).read_text().strip().splitlines()
    if len(lines) < 6:
        raise ValueError(f"PGW inválido: {pgw_path}")
    return (
        float(lines[0]),  # pixel_size_x
        float(lines[3]),  # pixel_size_y (negativo)
        float(lines[4]),  # x_origen (centro pixel sup-izq)
        float(lines[5]),  # y_origen
    )


# ── BBOX CALCULATION ───────────────────────────────────────────────────────
def compute_bbox(pixel_size_x, pixel_size_y, x0, y0, width_px, height_px):
    """
    Calcula las 4 esquinas en el SRC nativo del PGW (borde exterior del píxel).
    """
    half_x = pixel_size_x / 2.0
    half_y = pixel_size_y / 2.0   # negativo

    nw = (x0 - half_x,                            y0 - half_y)
    ne = (x0 - half_x + pixel_size_x * width_px,  y0 - half_y)
    se = (x0 - half_x + pixel_size_x * width_px,  y0 - half_y + pixel_size_y * height_px)
    sw = (x0 - half_x,                             y0 - half_y + pixel_size_y * height_px)
    return nw, ne, se, sw


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    script_dir = Path(__file__).parent
    layers_dir = script_dir / "layers"
    shapes_dir = script_dir / "shapes"
    out_path   = script_dir / "config.json"

    if not layers_dir.exists():
        print(f"[error] Carpeta no encontrada: {layers_dir}")
        return

    layers = {}

    for pgw_path in sorted(layers_dir.glob("*.pgw")):
        stem     = pgw_path.stem
        png_path = pgw_path.with_suffix(".png")

        if not png_path.exists():
            print(f"[skip] {stem}.pgw — falta {stem}.png")
            continue

        try:
            pixel_size_x, pixel_size_y, x0, y0 = parse_pgw(pgw_path)

            if PIL_AVAILABLE:
                with Image.open(png_path) as img:
                    width_px, height_px = img.size
            else:
                width_px, height_px = read_png_dimensions(png_path)

            corners_native = compute_bbox(pixel_size_x, pixel_size_y, x0, y0, width_px, height_px)

            # ¿UTM o ya WGS84?
            is_utm = abs(x0) > 1000 or abs(y0) > 1000

            if is_utm:
                print(f"[info] {stem}: UTM EPSG:32614 detectado → convirtiendo a WGS84")
                corners_wgs84 = [utm_to_wgs84(x, y, zone=14, northern=True)
                                 for x, y in corners_native]
            else:
                corners_wgs84 = list(corners_native)

            lons = [c[0] for c in corners_wgs84]
            lats = [c[1] for c in corners_wgs84]

            bbox = {
                "west":  round(min(lons), 8),
                "east":  round(max(lons), 8),
                "south": round(min(lats), 8),
                "north": round(max(lats), 8),
            }

            layers[stem] = {
                "path":      f"layers/{stem}.png",
                "bbox":      bbox,
                "epsg_src":  32614 if is_utm else 4326,
                "width_px":  width_px,
                "height_px": height_px,
            }

            print(f"[ok] {stem}.png  {width_px}×{height_px}px  "
                  f"W:{bbox['west']:.6f}  E:{bbox['east']:.6f}  "
                  f"S:{bbox['south']:.6f}  N:{bbox['north']:.6f}")

        except Exception as e:
            print(f"[error] {stem}: {e}")
            import traceback; traceback.print_exc()

    # Shapes
    shapes = {}
    for gj in sorted(shapes_dir.glob("*.geojson")):
        shapes[gj.stem] = f"shapes/{gj.name}"
        print(f"[ok] shape: {gj.name}")

    config = {
        "layers":    layers,
        "shapes":    shapes,
        "generated": datetime.datetime.now().isoformat(),
    }

    out_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"\n[done] {len(layers)} capa(s) · {len(shapes)} shape(s) → config.json")
    print(f"\nSiguiente paso:")
    print(f"  python -m http.server 8080")
    print(f"  http://localhost:8080")


if __name__ == "__main__":
    main()