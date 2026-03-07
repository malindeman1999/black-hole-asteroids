# Web Renderer Prototype

Minimal `FastAPI + Three.js` scaffold that calls Python visibility computation and renders an inertial tetrahedron.

## Install

```powershell
python -m pip install fastapi uvicorn
```

## Run

From project root:

```powershell
python web_renderer/server.py
```

Then open:

- http://127.0.0.1:8000

## API

- `GET /api/health`
- `POST /api/frame`

Example request:

```json
{
  "t": 12.5,
  "use_gpu": false
}
```

Response includes:

- tetra center
- tetra vertices and triangle indices
- first-visible corner/face branch direction and gamma angles at observer B

## Notes

- Uses `InertialTetrahedron` from `inertial_objects.py`.
- Uses precomputed interpolation table (`data/earliest_angles_precompute_10rs.npz` by default with fallbacks).
- Current tetra vertex offsets are simple rigid offsets (no relativistic contraction yet).
