// ────────────────────────────────────────────────────────────
// 时空呼吸 v3 Aurora · 音效系统(Web Audio API)
//
// 提供柔和的 UI 交互音:
// - tab 切换 / 按钮点击:轻"叮"
// - 卡片 hover:微"咔哒"
//
// 用 localStorage 记忆静音偏好
// ────────────────────────────────────────────────────────────
(function () {
  var doc, win;
  try {
    doc = window.parent.document;
    win = window.parent;
  } catch (e) { return; }

  // 防止重复注入
  if (win.__bhiSoundInit) return;
  win.__bhiSoundInit = true;

  var AC = win.AudioContext || win.webkitAudioContext;
  if (!AC) return;  // 不支持 Audio API,放弃

  var ctx = null;
  function getCtx() {
    if (!ctx) {
      try { ctx = new AC(); } catch (e) { return null; }
    }
    return ctx;
  }

  var muted = false;
  try { muted = win.localStorage.getItem('bhi_sound_muted') === '1'; } catch (e) {}

  // 合成"叮"声(衰减正弦波)
  function ding(freq, duration, vol) {
    if (muted) return;
    var c = getCtx();
    if (!c) return;
    if (c.state === 'suspended') c.resume();

    var t0 = c.currentTime;
    var osc = c.createOscillator();
    var gain = c.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, t0);
    gain.gain.setValueAtTime(0, t0);
    gain.gain.linearRampToValueAtTime(vol, t0 + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
    osc.connect(gain).connect(c.destination);
    osc.start(t0);
    osc.stop(t0 + duration);
  }

  function softClick() { ding(2200, 0.08, 0.04); }
  function tabSwitch() { ding(1320, 0.18, 0.06); setTimeout(function(){ ding(1760, 0.12, 0.04); }, 60); }
  function softHover() { ding(3520, 0.05, 0.015); }

  // ── 绑定监听 ──
  doc.addEventListener('click', function (e) {
    var tgt = e.target;
    // tab 切换
    if (tgt && (tgt.matches('[role="tab"]') || tgt.closest('[role="tab"]'))) {
      tabSwitch();
      return;
    }
    // 普通按钮
    if (tgt && (tgt.matches('button') || tgt.closest('button'))) {
      softClick();
    }
  }, true);

  // 仅 glass 卡 hover(不要太多)
  var lastHoverTime = 0;
  doc.addEventListener('mouseenter', function (e) {
    var tgt = e.target;
    if (tgt && tgt.classList && tgt.classList.contains('glass')) {
      var now = Date.now();
      if (now - lastHoverTime < 250) return;  // 节流
      lastHoverTime = now;
      softHover();
    }
  }, true);

  // ── 浮动音量按钮 ──
  function buildToggle() {
    var btn = doc.createElement('div');
    btn.id = 'bhi-sound-toggle';
    btn.innerHTML = muted ? '🔇' : '🔊';
    btn.title = muted ? '已静音 · 点击开启' : '声音开启 · 点击静音';
    btn.style.cssText = [
      'position:fixed',
      'bottom:18px', 'right:18px',
      'width:38px', 'height:38px',
      'border-radius:50%',
      'background:rgba(0,212,255,.08)',
      'border:1px solid rgba(0,212,255,.32)',
      'display:flex', 'align-items:center', 'justify-content:center',
      'cursor:pointer',
      'z-index:9998',
      'font-size:16px',
      'transition:all .2s',
      'backdrop-filter:blur(10px)',
      '-webkit-backdrop-filter:blur(10px)',
      'opacity:.7'
    ].join(';');

    btn.addEventListener('mouseenter', function () {
      btn.style.opacity = '1';
      btn.style.background = 'rgba(0,212,255,.18)';
      btn.style.transform = 'scale(1.08)';
    });
    btn.addEventListener('mouseleave', function () {
      btn.style.opacity = '.7';
      btn.style.background = 'rgba(0,212,255,.08)';
      btn.style.transform = 'scale(1)';
    });

    btn.addEventListener('click', function () {
      muted = !muted;
      btn.innerHTML = muted ? '🔇' : '🔊';
      btn.title = muted ? '已静音 · 点击开启' : '声音开启 · 点击静音';
      try { win.localStorage.setItem('bhi_sound_muted', muted ? '1' : '0'); } catch (e) {}
      if (!muted) softClick();
    });

    doc.body.appendChild(btn);
  }

  // 等 body 准备好
  if (doc.readyState === 'loading') {
    doc.addEventListener('DOMContentLoaded', buildToggle);
  } else {
    setTimeout(buildToggle, 100);
  }
})();
