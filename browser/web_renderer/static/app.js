import * as THREE from "https://unpkg.com/three@0.161.0/build/three.module.js";

const app = document.getElementById("app");
const statusEl = document.getElementById("status");
const timeEl = document.getElementById("time");
const dirsEl = document.getElementById("dirs");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x06101a, 1e8, 2e10);

const camera = new THREE.PerspectiveCamera(54, window.innerWidth / window.innerHeight, 1e5, 5e10);
camera.position.set(2.2e9, 1.6e9, 1.8e9);
camera.lookAt(0, 0, 0);

const hemi = new THREE.HemisphereLight(0xa8ccff, 0x0a1422, 1.1);
scene.add(hemi);

const dir = new THREE.DirectionalLight(0x99e6ff, 0.75);
dir.position.set(0.4, 1.0, 0.7);
scene.add(dir);

const grid = new THREE.GridHelper(2.5e10, 32, 0x244460, 0x16324a);
grid.position.y = -1.0e9;
scene.add(grid);

const horizon = new THREE.Mesh(
  new THREE.SphereGeometry(1.0, 48, 32),
  new THREE.MeshStandardMaterial({ color: 0x0e141b, roughness: 0.7, metalness: 0.1 })
);
scene.add(horizon);

const tetraGeometry = new THREE.BufferGeometry();
const tetraMaterial = new THREE.MeshStandardMaterial({
  color: 0x61c9ff,
  roughness: 0.3,
  metalness: 0.2,
  transparent: true,
  opacity: 0.9,
  side: THREE.DoubleSide,
});
const tetraMesh = new THREE.Mesh(tetraGeometry, tetraMaterial);
scene.add(tetraMesh);

const wire = new THREE.LineSegments(
  new THREE.EdgesGeometry(tetraGeometry),
  new THREE.LineBasicMaterial({ color: 0xd7f3ff })
);
scene.add(wire);

const observerMesh = new THREE.Mesh(
  new THREE.SphereGeometry(5.0e7, 16, 16),
  new THREE.MeshBasicMaterial({ color: 0xff59d6 })
);
scene.add(observerMesh);

let rs = 1.0;
let lastFrame = null;
let t = 0.0;

async function loadFrame(tt) {
  const payload = {
    t: tt,
    use_gpu: false,
  };
  const res = await fetch("/api/frame", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return await res.json();
}

function applyFrame(frame) {
  lastFrame = frame;
  rs = frame.rs_m;

  const verts = frame.vertices_m;
  const tris = frame.triangles;
  const pos = new Float32Array(verts.flat());
  const idx = new Uint16Array(tris.flat());

  tetraGeometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  tetraGeometry.setIndex(new THREE.BufferAttribute(idx, 1));
  tetraGeometry.computeVertexNormals();

  wire.geometry.dispose();
  wire.geometry = new THREE.EdgesGeometry(tetraGeometry);

  horizon.scale.setScalar(rs);
  observerMesh.position.set(...frame.observer_point_b);

  const dirs = frame.corner_first_direction || [];
  const plusCount = dirs.filter((v) => v > 0).length;
  const minusCount = dirs.filter((v) => v < 0).length;
  const noneCount = dirs.length - plusCount - minusCount;

  statusEl.textContent = "status: connected";
  timeEl.textContent = `t: ${frame.t.toFixed(3)} s`;
  dirsEl.textContent = `corner dir (+/-/0): ${plusCount}/${minusCount}/${noneCount}`;
}

function animate() {
  requestAnimationFrame(animate);
  tetraMesh.rotation.y += 0.002;
  wire.rotation.y = tetraMesh.rotation.y;

  if (lastFrame) {
    const c = lastFrame.center_m;
    const target = new THREE.Vector3(c[0], c[1], c[2]);
    camera.lookAt(target);
  }
  renderer.render(scene, camera);
}

let inflight = false;
async function tick() {
  if (inflight) return;
  inflight = true;
  try {
    t += 0.25;
    const frame = await loadFrame(t);
    applyFrame(frame);
  } catch (err) {
    statusEl.textContent = `status: error (${err})`;
  } finally {
    inflight = false;
  }
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

setInterval(tick, 120);
tick();
animate();
