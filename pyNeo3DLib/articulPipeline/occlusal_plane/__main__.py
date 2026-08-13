"""교합평면 CLI 진입점.

사용 예::

    python -m core.occlusal_plane landmarks.mrk.json --canal Mandibular_canal.stl
"""

from core.occlusal_plane.cli import main

if __name__ == "__main__":
    main()
