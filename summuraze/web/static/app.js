// summuraze — страница записей (#23). Без сборщика и фреймворка: список, конспект, транскрипт
// и плеер. Данные — /api/*, разметка — здесь. Текст из записи вставляется только через
// textContent: транскрипт и ответ модели — данные, а не HTML.
"use strict";

const app = document.getElementById("app");
const crumb = document.getElementById("crumb");
let timer = null; // опрос сервера, пока что-то идёт
let ticker = null; // секундомер стадии

const STATUS = {
  running: "идёт",
  done: "готово",
  failed: "ошибка",
  interrupted: "прервано",
};
const PROFILE = { lecture: "конспект", raw: "транскрипт по блокам" };
const MARK = {
  unsupported: ["❌", "нет в записи"],
  partial: ["◐", "частично"],
  unchecked: ["❔", "не проверено"],
};

// ── Мелочи ────────────────────────────────────────────────────────────

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function clock(seconds) {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

function localDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("ru-RU", { dateStyle: "medium", timeStyle: "short" });
}

function money(usd) {
  if (usd === null || usd === undefined) return "—";
  return usd === 0 ? "$0" : `$${usd.toFixed(4)}`;
}

async function api(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function stopTimers() {
  clearTimeout(timer);
  clearInterval(ticker);
  timer = ticker = null;
}

// Секундомер стадии от часов сервера: страница может быть открыта на другом устройстве.
function startTicker(serverNow) {
  const offset = Date.parse(serverNow) - Date.now();
  const tick = () => {
    for (const node of document.querySelectorAll("[data-since]")) {
      const since = Date.parse(node.dataset.since);
      node.textContent = clock(Math.max(0, (Date.now() + offset - since) / 1000));
    }
  };
  tick();
  ticker = setInterval(tick, 1000);
}

function statusNode(card) {
  if (!card.status) return el("span", { class: "status muted" }, "не обрабатывалась");
  const text = STATUS[card.status] || card.status;
  if (card.status === "running") {
    return el(
      "span",
      { class: "status running" },
      `${text} · ${card.stage_title || "старт"} `,
      card.stage_started_at ? el("span", { "data-since": card.stage_started_at }) : null,
    );
  }
  if (card.status === "failed" || card.status === "interrupted") {
    const where = card.stage_title ? ` на стадии «${card.stage_title}»` : "";
    return el("span", { class: `status ${card.status}`, title: card.error || "" }, text + where);
  }
  return el("span", { class: `status ${card.status}` }, text);
}

// ── Список записей ────────────────────────────────────────────────────

function showList() {
  crumb.textContent = "записи";
  document.title = "summuraze";
  // Форма запуска — вне перерисовки: опрос списка не должен стирать то, что человек выбирает.
  const list = el("div", {});
  app.replaceChildren(launchPanel(null), list);
  refreshList(list);
}

async function refreshList(list) {
  let data;
  try {
    data = await api("/api/recordings");
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, `Не загрузить список: ${e.message}`));
    return;
  }
  if (!list.isConnected) return; // пока ждали ответа, ушли на другую страницу
  if (!data.items.length) {
    list.replaceChildren(
      el("p", { class: "muted" }, "Записей пока нет. Загрузи файл выше, запиши лекцию: summuraze record, или прогони файл: summuraze run <файл>."),
    );
    return;
  }
  const rows = data.items.map((c) =>
    el(
      "tr",
      { "data-id": c.id, onclick: () => (location.hash = `#/r/${c.id}`) },
      el(
        "td",
        {},
        el("div", { class: "name" }, c.name),
        el("div", { class: "sub" }, localDate(c.started_at || c.first_seen)),
      ),
      el("td", { class: "num" }, c.duration ? clock(c.duration) : "—"),
      el("td", {}, statusNode(c)),
      el("td", { class: "opt" }, PROFILE[c.profile] || c.profile || "—"),
      el(
        "td",
        { class: "opt" },
        c.where || "—",
        c.llm_model ? el("div", { class: "sub" }, c.llm_model) : null,
      ),
      el("td", { class: "num opt" }, money(c.cost_usd)),
    ),
  );
  clearInterval(ticker);
  list.replaceChildren(
    el(
      "table",
      { class: "list" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "Запись"),
          el("th", { class: "num" }, "Длина"),
          el("th", {}, "Статус"),
          el("th", { class: "opt" }, "Профиль"),
          el("th", { class: "opt" }, "Куда ушёл текст"),
          el("th", { class: "num opt" }, "Цена"),
        ),
      ),
      el("tbody", {}, rows),
    ),
  );
  startTicker(data.now);
  const busy = data.items.some((c) => c.status === "running");
  timer = setTimeout(() => refreshList(list), busy ? 2000 : 15000);
}

// ── Запуск прогона ────────────────────────────────────────────────────

// Форма «сделать конспект»: на списке — с выбором файла, на странице записи — для неё же.
function launchPanel(fixedPath) {
  const msg = el("p", { class: "error" });
  const opts = el("div", {});
  const pick = async (path) => {
    msg.textContent = "";
    opts.replaceChildren(el("p", { class: "muted" }, "Смотрю запись…"));
    try {
      const o = await api(`/api/options?path=${encodeURIComponent(path)}`);
      opts.replaceChildren(renderOptions(o, msg));
    } catch (e) {
      opts.replaceChildren();
      msg.textContent = e.message;
    }
  };
  if (fixedPath) {
    pick(fixedPath);
    return el("section", { class: "launch" }, el("h2", {}, "Сделать заново"), opts, msg);
  }

  const select = el("select", { onchange: () => select.value && pick(select.value) });
  select.append(el("option", { value: "" }, "— выбери запись —"));
  api("/api/files")
    .then((d) => {
      for (const f of d.items) {
        const fresh = f.recording_id ? "" : " · новая";
        const size = `${(f.size / 2 ** 20).toFixed(0)} МБ`;
        select.append(el("option", { value: f.path }, `${f.name} · ${localDate(f.modified * 1000)} · ${size}${fresh}`));
      }
      if (!d.items.length) select.append(el("option", { disabled: true }, `в ${d.folder} записей нет`));
    })
    .catch((e) => (msg.textContent = `Не прочитать папку записей: ${e.message}`));

  const file = el("input", { type: "file", accept: "audio/*,video/*", hidden: true });
  const uploadButton = el("button", { type: "button", class: "secondary", onclick: () => file.click() }, "Загрузить файл…");
  file.addEventListener("change", async () => {
    const f = file.files[0];
    if (!f) return;
    msg.textContent = "";
    uploadButton.disabled = true;
    uploadButton.textContent = `Загружаю ${f.name}…`;
    try {
      const r = await fetch(`/api/upload?name=${encodeURIComponent(f.name)}`, { method: "PUT", body: f });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail);
      select.append(el("option", { value: d.path }, `${d.name} · загружена`));
      select.value = d.path;
      pick(d.path);
    } catch (e) {
      msg.textContent = `Не загрузить: ${e.message}`;
    } finally {
      uploadButton.disabled = false;
      uploadButton.textContent = "Загрузить файл…";
      file.value = "";
    }
  });
  return el(
    "section",
    { class: "launch" },
    el("h2", {}, "Новый конспект"),
    el("div", { class: "row" }, select, uploadButton, file),
    opts,
    msg,
  );
}

function renderOptions(o, msg) {
  const profile = el("select", {}, o.profiles.map((p) => el("option", { value: p.id }, p.title)));
  let chosen = o.providers.find((p) => p.ready) || null;
  const consentBox = el("input", { type: "checkbox" });
  const consentText = el("span", {});
  const consentRow = el("label", { class: "consent" }, consentBox, consentText);
  const go = el("button", { type: "button", class: "primary" }, "Сделать конспект");

  const providers = o.providers.map((p) => {
    const price = p.usd === null ? "цена неизвестна" : `≈ ${money(p.usd)}`;
    return el(
      "label",
      { class: `provider${p.ready ? "" : " off"}` },
      el("input", {
        type: "radio",
        name: "provider",
        value: p.id,
        checked: p === chosen,
        disabled: !p.ready,
        onchange: () => {
          chosen = p;
          sync();
        },
      }),
      el(
        "span",
        {},
        el("b", {}, p.title),
        ` — ${p.model} · ${p.host}`,
        el("div", { class: "sub" }, p.ready ? `${price} · ${p.basis}` : p.problem),
      ),
    );
  });

  function sync() {
    const need = Boolean(chosen && chosen.consent_needed);
    consentRow.hidden = !need;
    if (need) consentText.textContent = ` разрешаю отправлять текст транскрипта на ${chosen.host} — ответ запомню; аудио не уходит никогда`;
    go.disabled = !chosen || (need && !consentBox.checked);
  }
  consentBox.addEventListener("change", sync);

  go.addEventListener("click", async () => {
    msg.textContent = "";
    go.disabled = true;
    go.textContent = "Запускаю…";
    try {
      const r = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: o.path, profile: profile.value, provider: chosen.id, consent: consentBox.checked }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail);
      const target = `#/r/${d.recording_id}`;
      if (location.hash === target) route();
      else location.hash = target;
    } catch (e) {
      msg.textContent = e.message;
      go.textContent = "Сделать конспект";
      sync();
    }
  });
  sync();
  return el(
    "div",
    { class: "options" },
    o.duration ? el("div", { class: "meta" }, `Длительность ${clock(o.duration)}`) : null,
    el("label", { class: "row" }, "Что сделать: ", profile),
    el("div", { class: "providers" }, providers),
    consentRow,
    go,
  );
}

function progressContent(card) {
  const stop = el("button", { type: "button", class: "secondary" }, "Остановить");
  stop.addEventListener("click", async () => {
    if (!confirm("Остановить прогон? Готовые стадии останутся в кэше.")) return;
    stop.disabled = true;
    const r = await fetch(`/api/recordings/${card.id}/cancel`, { method: "POST" });
    stop.textContent = r.ok ? "останавливаю…" : "не остановить";
  });
  return ["Идёт обработка: ", statusNode(card), " ", stop];
}

function logDetails(launch, title) {
  return el("details", { class: "log" }, el("summary", {}, title), el("pre", {}, launch.log || "лог пуст"));
}

// ── Страница записи ───────────────────────────────────────────────────

async function showRecording(id) {
  let data;
  try {
    data = await api(`/api/recordings/${id}`);
  } catch (e) {
    app.replaceChildren(el("p", { class: "error" }, `Не загрузить запись: ${e.message}`));
    return;
  }
  const card = data.card;
  crumb.textContent = card.name;
  document.title = `${data.document ? data.document.title : card.name} — summuraze`;

  const audio = data.audio
    ? el("audio", { controls: true, preload: "metadata", src: `/api/recordings/${id}/audio` })
    : null;
  const seek = (t) => {
    if (!audio || t === null || t === undefined) return;
    audio.currentTime = t;
    audio.play().catch(() => {});
  };
  const tc = (t) =>
    t === null || t === undefined
      ? el("span", { class: "notc" }, "⚠️ без таймкода")
      : audio
        ? el("button", { class: "tc", title: "Слушать с этого места", onclick: () => seek(t) }, `▶ ${clock(t)}`)
        : el("span", { class: "tc" }, clock(t));
  const mark = (m) => {
    if (!m || m.status === "ok") return null;
    if (m.status === "moved") return el("span", { class: "mark moved" }, "↪ сказано в", tc(m.evidence));
    if (m.status === "no_timecode") return null; // tc уже показал «без таймкода»
    const [icon, label] = MARK[m.status] || ["", m.status];
    return el("span", { class: `mark ${m.status}` }, `${icon} ${label}${m.note ? `: ${m.note}` : ""}`);
  };

  const meta = [
    card.duration ? clock(card.duration) : null,
    data.asr_model ? `распознано ${data.asr_model}` : null,
    data.document ? `${PROFILE[data.document.profile] || data.document.profile}: ${data.document.llm_model}` : null,
    card.where ? `текст → ${card.where}` : null,
  ].filter(Boolean);

  const banners = [];
  const launch = data.launch;
  // Процесс запущен, но прогона в БД ещё нет: поднимается Python, считается хеш файла.
  const starting = launch && card.run_id === launch.previous_run;
  const progress = el("div", { class: "banner" }, ...progressContent(card));
  if (starting && launch.running) {
    banners.push(el("div", { class: "banner" }, "Запускается…"));
  } else if (starting) {
    banners.push(el("div", { class: "banner bad" }, `Прогон не запустился (код ${launch.code}).`, logDetails(launch, "лог")));
  } else if (card.status === "running") {
    banners.push(progress);
  } else if (card.status === "failed" || card.status === "interrupted") {
    const log = launch && launch.code ? logDetails(launch, "лог прогона") : null;
    banners.push(el("div", { class: "banner bad" }, statusNode(card), card.error ? ` — ${card.error}` : "", log));
  }
  const again = el("div", {});
  const againButton =
    card.status === "running" || (starting && launch.running)
      ? null
      : el("button", { type: "button", class: "link", onclick: () => again.replaceChildren(launchPanel(card.path)) }, "Сделать заново…");
  if (!data.audio) {
    banners.push(el("div", { class: "banner warn" }, "Звука нет: удалён по сроку хранения или из кэша — таймкоды без перехода."));
  }
  if (data.verification) {
    const all = data.document && data.document.parts.every((p) => p.items.every((i) => !i.mark || i.mark.status === "ok"));
    banners.push(el("div", { class: `banner ${all ? "" : "warn"}` }, `Проверено по записи: ${data.verification}`));
  }
  if (data.missing) banners.push(el("div", { class: "banner warn" }, data.missing));

  const doc = el("article", { class: "doc" });
  if (data.document) {
    doc.append(el("h1", {}, data.document.title));
    for (const part of data.document.parts) doc.append(renderPart(part, tc, mark));
    if (!data.document.parts.length && !data.document.blocks.length) {
      doc.append(el("p", { class: "muted" }, "Документ пустой."));
    }
    if (data.document.blocks.length) {
      doc.append(el("p", { class: "muted" }, "Профиль «транскрипт по блокам»: заголовки блоков — в транскрипте."));
    }
  } else if (!data.missing) {
    doc.append(el("p", { class: "muted" }, "Конспекта нет — запись только расшифрована."));
  }
  if (data.notes.length) {
    doc.append(el("p", { class: "meta" }, "Заметка Obsidian: ", data.notes.join(", ")));
  }

  const transcript = renderTranscript(data, seek);
  app.replaceChildren(
    el("div", { class: "player" }, audio, el("div", { class: "meta" }, meta.join(" · "), againButton ? " · " : "", againButton)),
    ...banners,
    again,
    el("div", { class: "columns" }, doc, transcript.node),
  );
  if (audio) follow(audio, transcript);
  startTicker(data.now);
  if (starting && launch.running) timer = setTimeout(route, 2000);
  else if (card.status === "running") timer = setTimeout(() => watchRun(id, progress), 3000);
}

// Пока прогон идёт — обновлять только строку статуса: перерисовка страницы сбросила бы плеер.
async function watchRun(id, progress) {
  let data;
  try {
    data = await api(`/api/recordings/${id}`);
  } catch {
    timer = setTimeout(() => watchRun(id, progress), 5000);
    return;
  }
  if (data.card.status !== "running") {
    route();
    return;
  }
  clearInterval(ticker);
  progress.replaceChildren(...progressContent(data.card));
  startTicker(data.now);
  timer = setTimeout(() => watchRun(id, progress), 3000);
}

function renderPart(part, tc, mark) {
  const items = part.items.map((i) => {
    if (part.kind === "glossary") {
      return el("li", {}, el("span", { class: "term" }, i.term), ` — ${i.text}`, tc(i.at), mark(i.mark));
    }
    if (part.kind === "qa") {
      return el("li", {}, i.text, tc(i.at), mark(i.mark), i.answer ? el("div", { class: "answer" }, i.answer) : null);
    }
    return el("li", {}, i.text, tc(i.at), mark(i.mark));
  });
  return el(
    "section",
    { class: `part ${part.kind}` },
    el("h2", {}, part.title, part.kind === "section" ? tc(part.at) : null),
    items.length ? el("ul", {}, items) : null,
  );
}

function renderTranscript(data, seek) {
  const lines = [];
  const body = el("div", {});
  const blocks = data.document ? [...data.document.blocks] : [];
  for (const line of data.lines) {
    while (blocks.length && blocks[0].at <= line.start + 0.001) {
      body.append(el("div", { class: "block" }, blocks.shift().title));
    }
    const node = el(
      "div",
      { class: `line${line.suspect ? " suspect" : ""}`, onclick: () => seek(line.start), title: line.reason || null },
      el("span", { class: "t" }, clock(line.start)),
      el("span", { class: "x" }, line.suspect ? `⚠️ ${line.text}` : line.text),
    );
    lines.push({ start: line.start, node });
    body.append(node);
  }
  const followBox = el("input", { type: "checkbox", checked: true });
  const node = el(
    "aside",
    { class: "transcript" },
    el("div", { class: "bar" }, `Транскрипт · ${data.lines.length} строк`, el("label", {}, followBox, " следить")),
    data.lines.length ? body : el("p", { class: "muted", style: "padding: 0 12px" }, "Транскрипта нет."),
  );
  return { node, lines, followBox };
}

// Подсветка строки, которая сейчас звучит; прокрутка к ней — если включено «следить».
function follow(audio, transcript) {
  const { lines, followBox, node } = transcript;
  let current = -1;
  audio.addEventListener("timeupdate", () => {
    const t = audio.currentTime;
    let lo = 0;
    let hi = lines.length - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (lines[mid].start <= t + 0.05) {
        found = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    if (found === current) return;
    if (current >= 0) lines[current].node.classList.remove("now");
    current = found;
    if (current < 0) return;
    const line = lines[current].node;
    line.classList.add("now");
    if (followBox.checked && !audio.paused) {
      const box = node.getBoundingClientRect();
      const r = line.getBoundingClientRect();
      if (r.top < box.top + 40 || r.bottom > box.bottom) line.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  });
}

// ── Маршрут ───────────────────────────────────────────────────────────

function route() {
  stopTimers();
  const m = location.hash.match(/^#\/r\/(\d+)/);
  if (m) showRecording(m[1]);
  else showList();
}

window.addEventListener("hashchange", () => {
  window.scrollTo(0, 0);
  route();
});
route();
