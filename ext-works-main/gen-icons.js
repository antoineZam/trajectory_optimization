const { createCanvas } = require('canvas');
const fs = require('fs');
const path = require('path');

function drawIcon(size) {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  const cx = size / 2, cy = size / 2, r = size / 2;

  // Background
  const bg = ctx.createRadialGradient(cx, cy * 0.8, 0, cx, cy, r);
  bg.addColorStop(0, '#1e1e30');
  bg.addColorStop(1, '#0d0d18');
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = bg;
  ctx.fill();

  // Outer glow ring
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.78, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(108, 99, 255, 0.5)';
  ctx.lineWidth = size * 0.055;
  ctx.stroke();

  // Second ring (faint)
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.58, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(167, 139, 250, 0.2)';
  ctx.lineWidth = size * 0.03;
  ctx.stroke();

  // Inner dot with gradient
  const dotGrad = ctx.createRadialGradient(cx - r*0.08, cy - r*0.08, 0, cx, cy, r * 0.32);
  dotGrad.addColorStop(0, '#c4b5fd');
  dotGrad.addColorStop(0.6, '#6c63ff');
  dotGrad.addColorStop(1, '#4338ca');
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.32, 0, Math.PI * 2);
  ctx.fillStyle = dotGrad;
  ctx.fill();

  return canvas;
}

const iconsDir = path.join(__dirname, 'icons');
if (!fs.existsSync(iconsDir)) fs.mkdirSync(iconsDir);

for (const size of [16, 48, 128]) {
  const canvas = drawIcon(size);
  const buf = canvas.toBuffer('image/png');
  fs.writeFileSync(path.join(iconsDir, `icon${size}.png`), buf);
  console.log(`Generated icon${size}.png`);
}
