// Renders explainer.html to frames with Chromium, then ffmpeg encodes an mp4.
//   node render.js [outdir] [fps] [only=t1,t2,...]   (only: write just those timestamps as PNG for a quick look)
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path'), { execFileSync } = require('child_process');

(async () => {
  const out = path.resolve(process.argv[2] || 'frames'), fps = +(process.argv[3] || 15);
  const only = process.argv[4] && process.argv[4].startsWith('only=') ? process.argv[4].slice(5).split(',').map(Number) : null;
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto('file://' + path.resolve(__dirname, 'explainer.html'));
  const dur = await page.evaluate(() => window.DURATION);
  const grab = async (t, file, type) => {
    const url = await page.evaluate(([t, type]) => { draw(t); return document.getElementById('c').toDataURL(type, 0.92); }, [t, type]);
    fs.writeFileSync(file, Buffer.from(url.split(',')[1], 'base64'));
  };
  if (only) { for (const t of only) await grab(t, path.join(out, `t${t}.png`), 'image/png'); }
  else {
    const n = Math.round(dur * fps);
    for (let i = 0; i < n; i++) await grab(i / fps, path.join(out, `f${String(i).padStart(5, '0')}.jpg`), 'image/jpeg');
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', path.join(out, 'f%05d.jpg'),
      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', path.resolve(__dirname, 'inkling2_explainer.mp4')]);
  }
  await browser.close();
})();
