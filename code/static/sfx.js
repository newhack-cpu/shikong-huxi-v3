// ════════════════════════════════════════════════════════════════
// 时空呼吸 v3 · 音效系统(Web Audio API · 默认关闭)
// 用代码合成声音,不依赖外部音频文件
// ════════════════════════════════════════════════════════════════
(function () {
  // 用 parent 上下文,使 toggle 全局可控
  var win = window.parent || window;
  var doc = win.document;

  // 防重复注入
  if (win.__sfxInit) return;
  win.__sfxInit = true;

  // 默认关闭(避免页面加载就响)
  win.__sfxEnabled = false;

  var ctx = null;

  function getCtx() {
    if (!ctx) {
      try {
        var AC = win.AudioContext || win.webkitAudioContext;
        ctx = new AC();
      } catch (e) { return null; }
    }
    return ctx;
  }

  // 合成"叮"声(短促高频)
  function ding() {
    if (!win.__sfxEnabled) return;
    var c = getCtx();
    if (!c) return;
    var t = c.currentTime;
    var osc = c.createOscillator();
    var gain = c.createGain();
    osc.frequency.setValueAtTime(880, t);
    osc.frequency.exponentialRampToValueAtTime(1760, t + 0.05);
    gain.gain.setValueAtTime(0.08, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    osc.connect(gain).connect(c.destination);
    osc.start(t);
    osc.stop(t + 0.25);
  }

  // 合成"嗡"声(低频长音,像呼吸)
  function hum() {
    if (!win.__sfxEnabled) return;
    var c = getCtx();
    if (!c) return;
    var t = c.currentTime;
    var osc = c.createOscillator();
    var osc2 = c.createOscillator();
    var gain = c.createGain();
    osc.frequency.setValueAtTime(220, t);
    osc2.frequency.setValueAtTime(330, t);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.06, t + 0.1);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.8);
    osc.connect(gain);
    osc2.connect(gain);
    gain.connect(c.destination);
    osc.start(t); osc2.start(t);
    osc.stop(t + 0.85); osc2.stop(t + 0.85);
  }

  // 暴露给开关使用
  win.__sfxDing = ding;
  win.__sfxHum = hum;

  // 监听全局事件(在 parent document 上)
  // KPI 卡 hover → ding
  doc.addEventListener('mouseover', function (e) {
    if (!win.__sfxEnabled) return;
    var t = e.target;
    if (t && t.closest && t.closest('.kpi-wrap')) {
      // 防抖:每 300ms 最多响一次
      var now = Date.now();
      if (now - (win.__sfxLastDing || 0) > 300) {
        win.__sfxLastDing = now;
        ding();
      }
    }
  });

  // 呼吸球点击 → hum
  doc.addEventListener('click', function (e) {
    if (!win.__sfxEnabled) return;
    var t = e.target;
    if (t && t.closest && t.closest('.breath-orb')) {
      hum();
    }
  });

  console.log('[sfx] ready, enabled:', win.__sfxEnabled);
})();
