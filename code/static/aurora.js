// ────────────────────────────────────────────────────────────
// 时空呼吸 v3 Aurora · 粒子背景(streamlit components 兼容版)
//
// 工作原理:
//   - 此脚本在 streamlit components.html 创建的 iframe 里运行
//   - 通过 window.parent.document 访问主页面 body
//   - 在主页面创建一个 fixed 全屏 canvas
//   - 粒子动画在主页面上下文运行,覆盖整页
//
// 同源安全:streamlit iframe 与主页面同源(都在 localhost:8501),
// parent.document 访问合法,无 CORS 问题
// ────────────────────────────────────────────────────────────
(function () {
  // ── 拿到主页面 document ──
  var doc;
  try {
    doc = window.parent.document;
  } catch (e) {
    // 极端情况:无法访问 parent(开发环境/iframe隔离),静默退出
    console.warn('[particles] cannot access parent.document, skip');
    return;
  }

  // ── 防止重复注入 ──
  var existing = doc.getElementById('particle-bg-canvas');
  if (existing) existing.remove();

  // ── 创建主页面 canvas ──
  var canvas = doc.createElement('canvas');
  canvas.id = 'particle-bg-canvas';
  canvas.style.cssText = [
    'position:fixed',
    'top:0', 'left:0',
    'width:100vw', 'height:100vh',
    'z-index:0',
    'pointer-events:none',
    'opacity:0.55'
  ].join(';');
  doc.body.appendChild(canvas);

  var ctx = canvas.getContext('2d');
  var W, H;
  var particles = [];
  var N;

  function resize() {
    W = canvas.width = window.parent.innerWidth;
    H = canvas.height = window.parent.innerHeight;
    N = W < 768 ? 60 : 130;  // 增加密度
  }
  resize();
  window.parent.addEventListener('resize', resize);

  // ── 粒子工厂(分3层:近/中/远,模拟景深) ──
  // tier 0: 远 (60%) - 小、快、淡
  // tier 1: 中 (30%) - 中等
  // tier 2: 近 (10%) - 大、慢、亮、有光晕
  function spawn(tier) {
    if (tier === undefined) {
      var rnd = Math.random();
      tier = rnd < 0.6 ? 0 : (rnd < 0.9 ? 1 : 2);
    }
    var profiles = [
      { rMin: 0.3, rMax: 0.9,  vyMin: 0.18, vyMax: 0.45, oMin: 0.10, oMax: 0.35, glow: 4  }, // 远
      { rMin: 0.8, rMax: 1.6,  vyMin: 0.12, vyMax: 0.30, oMin: 0.20, oMax: 0.50, glow: 6  }, // 中
      { rMin: 1.6, rMax: 2.8,  vyMin: 0.06, vyMax: 0.18, oMin: 0.35, oMax: 0.70, glow: 10 }  // 近
    ];
    var pr = profiles[tier];
    return {
      tier: tier,
      x: Math.random() * W,
      y: H + Math.random() * 40,
      r: pr.rMin + Math.random() * (pr.rMax - pr.rMin),
      vy: -(pr.vyMin + Math.random() * (pr.vyMax - pr.vyMin)),
      vx: (Math.random() - 0.5) * (0.08 + tier * 0.06),  // 近层横向漂移更明显
      o: pr.oMin + Math.random() * (pr.oMax - pr.oMin),
      ph: Math.random() * Math.PI * 2,
      glow: pr.glow,
      life: 0,
      maxLife: 600 + Math.random() * 400
    };
  }

  for (var i = 0; i < (N || 130); i++) {
    var p = spawn();
    p.y = Math.random() * H;
    p.life = Math.random() * p.maxLife;
    particles.push(p);
  }

  // ── 颜色:从主页面 CSS 变量 --bhi-particle-color 读取 ──
  function getColor() {
    try {
      var c = window.parent.getComputedStyle(doc.documentElement)
                  .getPropertyValue('--bhi-particle-color').trim();
      if (c) return c;
    } catch (e) {}
    return '#00d4ff';  // 默认色
  }

  // ── 主循环 ──
  function tick() {
    if (!doc.body.contains(canvas)) return;  // canvas 被移除则停止
    ctx.clearRect(0, 0, W, H);

    var color = getColor();
    var r = 0, g = 212, b = 255;
    if (color && color[0] === '#' && color.length === 7) {
      r = parseInt(color.slice(1, 3), 16);
      g = parseInt(color.slice(3, 5), 16);
      b = parseInt(color.slice(5, 7), 16);
    }

    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.life++;
      p.x += p.vx + Math.sin(p.life * 0.008 + p.ph) * 0.25;
      p.y += p.vy;
      var fade = Math.min(1, Math.min(p.life, p.maxLife - p.life) / 80);
      var a = p.o * fade;

      // 主点
      ctx.beginPath();
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();

      // 光晕(近层粒子光晕更大,营造景深)
      var glowR = p.r * (p.glow || 6);
      var glowA = p.tier === 2 ? a * 0.55 : (p.tier === 1 ? a * 0.4 : a * 0.25);
      var grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowR);
      grad.addColorStop(0, 'rgba(' + r + ',' + g + ',' + b + ',' + glowA + ')');
      grad.addColorStop(1, 'rgba(' + r + ',' + g + ',' + b + ',0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2);
      ctx.fill();

      if (p.y < -10 || p.life > p.maxLife) {
        particles[i] = spawn();
      }
    }

    window.parent.requestAnimationFrame(tick);
  }
  window.parent.requestAnimationFrame(tick);

  console.log('[particles] OK, N=' + N + ' on ' + W + 'x' + H);
})();

// 侧边栏：默认折叠 + 隐藏箭头
(function(){
  var done = false;
  function handleSidebar(){
    // 隐藏折叠箭头（侧边栏关闭时的浮动箭头）
    ['[data-testid="collapsedControl"]','.stApp>button[aria-label="Open sidebar"]'].forEach(function(s){
      try{ var el=parent.document.querySelector(s); if(el){el.style.display='none';el.style.visibility='hidden';el.style.opacity='0';} }catch(e){}
    });
    // 首次加载强制折叠侧边栏
    if(!done){
      var closeBtn = parent.document.querySelector('[data-testid="stSidebarCollapseButton"] button');
      if(closeBtn){ closeBtn.click(); done=true; }
    }
  }
  handleSidebar();
  new MutationObserver(handleSidebar).observe(parent.document.body,{childList:true,subtree:true});
})();
