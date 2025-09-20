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
const programInfoEl = document.getElementById('program-info');
const hardwareInfoEl = document.getElementById('hardware-info');

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
    statusEl.textContent = '';
}

function applyProgramInfo(items) {
    programInfoEl.innerHTML = '';
    items.forEach((txt) => {
        const li = document.createElement('li');
        li.textContent = txt;
        programInfoEl.appendChild(li);
    });
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
            applyProgramInfo(data.program_settings);
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

fetchSettings();
