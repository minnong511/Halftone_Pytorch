'use strict';

const paramMeta = {
    cell: {
        slider: document.getElementById('cell'),
        display: document.getElementById('cell-value'),
        toSlider: (v) => v,
        fromSlider: (v) => parseInt(v, 10),
        format: (v) => v.toString(),
    },
    r_max: {
        slider: document.getElementById('r_max'),
        display: document.getElementById('r_max-value'),
        toSlider: (v) => v,
        fromSlider: (v) => parseInt(v, 10),
        format: (v) => v.toString(),
    },
    r_min: {
        slider: document.getElementById('r_min'),
        display: document.getElementById('r_min-value'),
        toSlider: (v) => v,
        fromSlider: (v) => parseInt(v, 10),
        format: (v) => v.toString(),
    },
    gamma: {
        slider: document.getElementById('gamma'),
        display: document.getElementById('gamma-value'),
        toSlider: (v) => Math.round(v * 10),
        fromSlider: (v) => parseInt(v, 10) / 10,
        format: (v) => v.toFixed(2),
    },
    blur_level: {
        slider: document.getElementById('blur_level'),
        display: document.getElementById('blur_level-value'),
        toSlider: (v) => v,
        fromSlider: (v) => parseInt(v, 10),
        format: (v) => v.toString(),
    },
};

const statusEl = document.getElementById('status');
const hardwareInfoEl = document.getElementById('hardware-info');
const modeSelect = document.getElementById('mode');
const modeDisplay = document.getElementById('mode-value');
const colorInput = document.getElementById('dot_color');
const colorDisplay = document.getElementById('color-value');

function applyParams(params) {
    Object.entries(paramMeta).forEach(([key, config]) => {
        if (!(key in params)) {
            return;
        }
        const sliderVal = config.toSlider(params[key]);
        if (config.slider.value !== String(sliderVal)) {
            config.slider.value = sliderVal;
        }
        config.display.textContent = config.format(params[key]);
    });
    if ('cell' in params) {
        const maxR = Math.max(1, Math.min(32, Math.floor(params.cell / 2)));
        paramMeta.r_max.slider.max = maxR;
        if (params.r_max > maxR) {
            paramMeta.r_max.slider.value = maxR;
        }
    }
    if ('r_max' in params) {
        paramMeta.r_min.slider.max = params.r_max;
        if (params.r_min > params.r_max) {
            paramMeta.r_min.slider.value = params.r_max;
        }
    }
    if ('mode' in params && modeSelect) {
        if (modeSelect.value !== params.mode) {
            modeSelect.value = params.mode;
        }
        if (modeDisplay) {
            modeDisplay.textContent = params.mode.toUpperCase();
        }
    }
    if ('color_hex' in params && colorInput) {
        const hex = params.color_hex;
        if (colorInput.value !== hex) {
            colorInput.value = hex;
        }
        if (colorDisplay) {
            colorDisplay.textContent = hex.toUpperCase();
        }
    }
    statusEl.textContent = '';
}

function applyHardwareInfo(items) {
    hardwareInfoEl.innerHTML = '';
    items.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'hardware-row';
        const label = document.createElement('span');
        label.textContent = item.label;
        const value = document.createElement('span');
        value.textContent = item.status;
        row.appendChild(label);
        row.appendChild(value);
        hardwareInfoEl.appendChild(row);
    });
}

function fetchSettings() {
    fetch('/api/settings')
        .then((res) => res.json())
        .then((data) => {
            applyParams(data.params);
            applyHardwareInfo(data.hardware);
        })
        .catch((err) => {
            statusEl.textContent = '설정을 불러올 수 없습니다.';
            console.error(err);
        });
}

function sendPartialUpdate(key, value) {
    fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
    })
        .then((res) => res.json())
        .then((data) => {
            applyParams(data.params);
        })
        .catch((err) => console.error(err));
}

Object.entries(paramMeta).forEach(([key, config]) => {
    config.slider.addEventListener('input', (ev) => {
        const raw = ev.target.value;
        const val = config.fromSlider(raw);
        config.display.textContent = config.format(val);
    });
    config.slider.addEventListener('change', (ev) => {
        const raw = ev.target.value;
        const val = config.fromSlider(raw);
        sendPartialUpdate(key, val);
    });
});

document.getElementById('snapshot').addEventListener('click', () => {
    fetch('/api/snapshot', { method: 'POST' })
        .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            if (ok && data.path) {
                statusEl.textContent = `저장됨: ${data.path}`;
            } else {
                statusEl.textContent = '스냅샷 저장 실패';
            }
        })
        .catch((err) => {
            statusEl.textContent = '스냅샷 저장 실패';
            console.error(err);
        });
});

if (modeSelect) {
    modeSelect.addEventListener('change', (ev) => {
        const value = ev.target.value;
        if (modeDisplay) {
            modeDisplay.textContent = value.toUpperCase();
        }
        sendPartialUpdate('mode', value);
    });
    if (modeDisplay) {
        modeDisplay.textContent = modeSelect.value.toUpperCase();
    }
}

if (colorInput) {
    colorInput.addEventListener('input', (ev) => {
        const value = ev.target.value;
        if (colorDisplay) {
            colorDisplay.textContent = value.toUpperCase();
        }
    });
    colorInput.addEventListener('change', (ev) => {
        const value = ev.target.value;
        if (colorDisplay) {
            colorDisplay.textContent = value.toUpperCase();
        }
        sendPartialUpdate('color_hex', value);
    });
    if (colorDisplay) {
        colorDisplay.textContent = colorInput.value.toUpperCase();
    }
}

let shutdownRequested = false;

function requestShutdown() {
    if (shutdownRequested) {
        return;
    }
    shutdownRequested = true;
    if (navigator.sendBeacon) {
        const blob = new Blob(['bye'], { type: 'text/plain' });
        navigator.sendBeacon('/api/shutdown', blob);
    } else {
        fetch('/api/shutdown', { method: 'POST', keepalive: true }).catch(() => {});
    }
}

window.addEventListener('pagehide', (event) => {
    if (!event.persisted) {
        requestShutdown();
    }
});

window.addEventListener('beforeunload', () => {
    requestShutdown();
});

fetchSettings();
