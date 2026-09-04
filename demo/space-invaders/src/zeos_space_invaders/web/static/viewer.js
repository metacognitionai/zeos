// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/* Draws what the payload says. No fold, no arithmetic on the log -- that lives
 * in web/payload.py, where it is tested. Scrubbing is an array index, which is
 * possible because a run is a recording and the frames are all here.
 */
(function () {
  "use strict";

  const DATA = window.__RUN__;           // set when the page was exported
  // The board comes from the run's own `meta.json`; these are the fallback for
  // runs recorded before it was there, and the defaults in game/rules.py.
  const W = 9, H = 8, CELL = 40;
  // The board this run was played on, read from `meta` once when the run is
  // rendered: the grid and the ship have to be placed from one source, or the
  // ship lands outside the viewBox on any board that is not the default.
  let played = {w: W, h: H};
  const PLAY_MS = 120;

  const $ = (id) => document.getElementById(id);
  const screen = $("screen");

  let run = null;                        // the episode payload being shown
  let at = 0;                            // the tick under the cursor
  let playing = false, timer = null;
  let pieces = {};                       // board elements, reused across ticks
  let folds = {};                        // which decision folds the reader opened

  // --- small builders -------------------------------------------------------

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function svg(tag, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    return node;
  }

  function table(headers, rows) {
    const t = el("table");
    const head = el("thead"), tr = el("tr");
    headers.forEach((h) => {
      const th = el("th", h.num ? "num" : null, h.label !== undefined ? h.label : h);
      tr.appendChild(th);
    });
    head.appendChild(tr);
    t.appendChild(head);
    const body = el("tbody");
    rows.forEach((r) => body.appendChild(r));
    t.appendChild(body);
    return t;
  }

  function cells(values, cls) {
    const tr = el("tr", cls);
    values.forEach((v) => {
      const td = el("td", v && v.cls ? v.cls : null);
      const text = v && typeof v === "object" && "text" in v ? v.text : v;
      td.textContent = text === null || text === undefined ? "—" : String(text);
      tr.appendChild(td);
    });
    return tr;
  }

  const seconds = (v) => (v === null || v === undefined ? null : v.toFixed(2) + " s");

  function tokensOf(usage) {
    if (!usage) return null;
    const total = Object.keys(usage).reduce(
      (sum, k) => (typeof usage[k] === "number" ? sum + usage[k] : sum), 0);
    return total ? total.toLocaleString() : null;
  }

  function describe(meta) {
    return [meta.player, meta.view, meta.effort, meta.clock,
            meta.seed === null || meta.seed === undefined ? null : "seed " + meta.seed]
      .filter(Boolean).join(" · ");
  }

  // --- the columns of the index ---------------------------------------------

  /* Declared once, because Tabulator wants column definitions and the toolbar
   * wants the groups. `group` is ours: a column with no group is always shown,
   * the rest come and go a group at a time. A missing number draws as an empty
   * cell rather than a dash, because a dash per absence would be most of the
   * table. */
  const GROUPS = ["game", "agent", "kernel", "setup"];
  const OPEN = ["game"];              // shown before anyone has chosen

  /* Bumped whenever the list below changes: Tabulator stores a layout under
   * this id, and a stored layout beats the code's defaults forever. */
  const LAYOUT = "invader-index-v11";

  /* Every persistence id this page owns; `reset` clears all of them, since the
     comparison's two tables have no button of their own. Tabulator keys them
     `tabulator-<id>-<sort|filter|columns>`. */
  const STORED = "tabulator-invader-";

  /* Folded comparisons are table state like the sort, stored under `STORED`
     where `reset` clears them; Tabulator's own suffixes are `sort`, `filter`
     and `columns`, so this one cannot collide. */
  const FOLDED = STORED + "index-folded";

  const COLUMNS = [
    // Wider than its title needs: the tree control and its indent live here.
    {field: "started_at", title: "when", kind: "when", min: 128},
    {field: "player", title: "player", kind: "text"},
    {field: "view", title: "view", kind: "text"},
    {field: "effort", title: "effort", kind: "text"},
    {field: "clock", title: "clock", kind: "text"},
    {field: "seed", title: "seed", kind: "number"},
    {field: "outcome", title: "outcome", kind: "outcome"},
    {field: "score", title: "score", kind: "number"},
    {field: "ticks", title: "ticks", kind: "number"},

    {field: "lives", title: "lives", kind: "number", group: "game"},
    {field: "monsters_left", title: "left", kind: "number", group: "game"},
    {field: "reward", title: "reward", kind: "number", dp: 1, group: "game"},
    {field: "tick_seconds", title: "tick s", kind: "number", dp: 2,
     group: "game"},
    // The rules the run was played under, not a result: a run with monster fire
    // turned up must not read like one without.
    {field: "fire_chance", title: "fire", kind: "number", dp: 2, group: "game"},
    // Two runs at different board sizes have to be distinguishable in the
    // table, not only inside the file.
    {field: "width", title: "w", kind: "number", group: "game"},
    {field: "height", title: "h", kind: "number", group: "game"},
    {field: "lives", title: "lives", kind: "number", group: "game"},
    {field: "danger_rows", title: "fall", kind: "number", group: "game"},

    {field: "decisions", title: "decisions", kind: "number", group: "agent"},
    {field: "actions_per_tick", title: "act/tick", kind: "number", dp: 3,
     group: "agent"},
    {field: "mean_latency", title: "s/dec", kind: "number", dp: 2, group: "agent"},
    {field: "mean_ticks_waited", title: "waited", kind: "number", dp: 2,
     group: "agent"},
    {field: "unparseable", title: "unparsed", kind: "number", group: "agent"},
    {field: "tokens", title: "tokens", kind: "count", group: "agent"},
    {field: "output_tokens", title: "out tok", kind: "count", group: "agent"},
    {field: "seconds", title: "wall", kind: "number", dp: 1, group: "agent"},
    {field: "dropped_after_game_over", title: "dropped", kind: "number",
     group: "agent"},

    {field: "pilot_moves", title: "pilot", kind: "number", group: "kernel"},
    {field: "reflexes", title: "reflexes", kind: "number", group: "kernel"},
    // `criteria` is zeos's verdict off the journal; `preempts` is the kernel's
    // own count, and the two after it are our bookkeeping about the I/O that
    // followed.
    {field: "criteria", title: "criteria", kind: "text", group: "kernel"},
    {field: "preemptions", title: "preempts", kind: "number", group: "kernel"},
    // A completion a new board made *stale* is cancelled, and `voided` is the
    // part of it that amounted to no syscall and is not sent again.
    {field: "generations", title: "completions", kind: "number", group: "kernel"},
    {field: "cancellations", title: "stale", kind: "number", group: "kernel"},
    {field: "voided", title: "voided", kind: "number", group: "kernel"},
    // Both are in the transcript and charged to the descriptor's budget; only
    // `decoded` is ever sent back.
    {field: "decoded_words", title: "decoded", kind: "number", group: "kernel"},
    {field: "native_words", title: "reflex", kind: "number", group: "kernel"},
    {field: "thinking_words", title: "thinking", kind: "number", group: "kernel"},

    // `min` and `grow` where the *data* is the wide thing: a model name or an
    // endpoint runs to twenty-odd characters, and an even share of the width
    // truncates exactly the columns worth reading.
    {field: "model", title: "model", kind: "mono", group: "setup",
     min: 160, grow: 2},
    {field: "base_url", title: "endpoint", kind: "mono", group: "setup",
     min: 190, grow: 2},
    {field: "history", title: "history", kind: "number", group: "setup"},
    // "on"/"off" rather than a boolean, so the column gets the list filter the
    // other enumerations have.
    {field: "stream", title: "stream", kind: "text", group: "setup"},
    // The machine's policy for the tail of a cancelled completion: under a
    // syscall schema a half-answer is a fragment of a call that never happened.
    {field: "partial", title: "partial", kind: "text", group: "setup"},
    {field: "max_steps", title: "max steps", kind: "number", group: "setup"},
    {field: "commit", title: "commit", kind: "mono", group: "setup", min: 115},
    {field: "zeos", title: "zeos", kind: "mono", group: "setup", min: 105},
  ];

  /* What the search box looks in. Text fields only, and every one of them is
     something you would plausibly remember a run by. */
  const SEARCH = ["run", "player", "model", "view", "effort", "clock",
                  "outcome", "commit"];

  const shortWhen = (v) => String(v || "").replace("T", " ").slice(5, 16);

  /* What a header needs, measured rather than estimated: the header is the
   * widest thing in most columns and by how much depends on its letters, so the
   * text is measured in a probe sharing the header's type (one declaration in
   * viewer.css) and the sort arrow and menu handle are this constant. The
   * floors are what let `fitColumns` share spare width, and overflow into a
   * scroller rather than squeeze titles to ellipses. */
  const HEADER_CHROME = 56;

  const floorFor = (function () {
    const known = {};
    return (title) => {
      if (known[title] === undefined) {
        const probe = el("div", "probe");
        const span = el("span", null, title);
        probe.appendChild(span);
        document.body.appendChild(probe);
        known[title] =
          Math.ceil(span.getBoundingClientRect().width) + HEADER_CHROME;
        probe.remove();
      }
      return known[title];
    };
  })();

  /* `minWidth` for a column list written inline, as the comparison's two are.
     Spread after, so a column that names its own floor keeps it. */
  function withFloors(columns) {
    return columns.map((col) => ({minWidth: floorFor(col.title), ...col}));
  }

  function empty(value) {
    return value === null || value === undefined || value === "";
  }

  /* What a cell says, before anything is built out of it. `diff` compares these
     rather than the values behind them: two latencies that both draw as `2.00`
     are the same cell to the reader. */
  function textOf(col, value) {
    if (col.kind === "when") return shortWhen(value);
    if (empty(value)) return "";
    if (col.kind === "count") return Number(value).toLocaleString();
    if (col.kind === "number") {
      return col.dp === undefined ? String(value) : Number(value).toFixed(col.dp);
    }
    return String(value);
  }

  /* Tabulator sanitises its own `plaintext` formatter's string and inserts a
     custom formatter's string as HTML, so every formatter here returns a DOM
     node or a number, never a string made out of data. */
  function formatterFor(col) {
    if (col.kind === "when") {
      return (cell) => document.createTextNode(textOf(col, cell.getValue()));
    }
    if (col.kind === "outcome") {
      return (cell) => outcomeCell(cell.getRow().getData());
    }
    if (col.kind === "mono") {
      return (cell) => (empty(cell.getValue())
        ? "" : el("span", "mono", cell.getValue()));
    }
    if (col.kind === "count" || col.kind === "number") {
      return (cell) => textOf(col, cell.getValue());
    }
    return undefined;                 // plaintext: sanitised by Tabulator
  }

  /* A comparison carries real numbers -- the means over its episodes -- because
     a spanning cell has nothing to sort by, and is still marked as one. */
  function outcomeCell(row) {
    const text = outcomeText(row);
    if (row.unreadable) return el("span", "flag bad", text);
    if (row.is_compare) return el("span", row.finished ? null : "dim", text);
    if (!row.finished) return el("span", "dim", text);
    if (!text) return "";
    return el("span", row.outcome === "won" ? "won"
                : row.outcome === "lost" ? "lost" : null, text);
  }

  /* The same ladder as words, for the diff and for the cell above to draw. */
  function outcomeText(row) {
    if (row.unreadable) return "unreadable";
    if (row.is_compare) return row.wins + "/" + row.episode_count + " won";
    if (!row.finished) return "unfinished";
    return empty(row.outcome) ? "" : String(row.outcome);
  }

  function definition(col, withFilters) {
    const def = {
      field: col.field,
      title: col.title,
      visible: !col.group || OPEN.indexOf(col.group) >= 0,
      sorter: col.kind === "text" || col.kind === "mono"
        || col.kind === "when" || col.kind === "outcome" ? "string" : "number",
      formatter: formatterFor(col),
      headerMenu: columnMenu,
      headerTooltip: col.group ? col.title + " — " + col.group : col.title,
    };
    if (def.sorter === "number") def.hozAlign = "right";
    def.minWidth = Math.max(col.min || 0, floorFor(col.title));
    if (col.grow) def.widthGrow = col.grow;
    if (!withFilters) return def;
    // A list filter for the enumerations and an "at least this much" box for
    // the numbers, both Tabulator's own.
    if (def.sorter === "number") {
      def.headerFilter = "number";
      def.headerFilterFunc = ">=";
      def.headerFilterPlaceholder = "≥";
    } else {
      def.headerFilter = "list";
      def.headerFilterParams = {valuesLookup: true, clearable: true};
    }
    return def;
  }

  /* Every column, with a tick against the shown ones. Rebuilt each time the
     menu opens, so the ticks are the table's state rather than a copy of it. */
  function columnMenu() {
    const items = [];
    let group;
    COLUMNS.forEach((col) => {
      if (col.group !== group) {
        if (items.length) items.push({separator: true});
        group = col.group;
      }
      items.push({
        label: menuLabel(col),
        action: (event) => {
          // Kept open so several columns can be chosen in one visit, which means
          // the tick has to be redrawn here: Tabulator builds the menu once.
          event.stopPropagation();
          indexTable.toggleColumn(col.field);
          refit();
          drawGroups();
          const item = event.target.closest(".tabulator-menu-item");
          if (item) {
            item.textContent = "";
            item.appendChild(menuLabel(col));
          }
        },
      });
    });
    return items;
  }

  function menuLabel(col) {
    const column = indexTable && indexTable.getColumn(col.field);
    const on = column && column.isVisible();
    const node = el("span", "menu-col" + (on ? " on" : ""));
    node.appendChild(el("b", null, on ? "✓" : " "));
    node.appendChild(document.createTextNode(col.title));
    return node;
  }

  // --- leaving a screen -----------------------------------------------------

  /* Every Tabulator on the page. Destroyed rather than dropped: a column menu
     lives on `document.body`, and one left behind matches `refreshNow`'s
     open-menu guard forever. */
  let live = [];

  function dropTables() {
    live.forEach((table) => {
      try {
        table.destroy();
      } catch (err) {
        // A table already torn down. Nothing to do, and nothing to say.
      }
    });
    live = [];
  }

  function keep(table) {
    live.push(table);
    return table;
  }

  // --- reloading on a timer -------------------------------------------------

  /* One screen at a time knows how to reload itself; `reloader` is that
     function, null on a screen that must not, which is any run. A reload is
     `replaceData` rather than a rebuild, because the sort, filters, chosen
     columns and scroll position are not in the payload. */
  const INTERVALS = [2, 5, 15, 60];

  /* Deliberately not under `STORED`: `reset` in the toolbar is the table's
     reset, and how often the page reloads is not the table's business. */
  const REFRESH = "invader-refresh";

  let reloader = null, refreshTimer = null, refreshing = false;
  let refreshOn = true, refreshEvery = 5;

  function wireRefresh() {
    try {
      const saved = JSON.parse(localStorage.getItem(REFRESH) || "{}");
      if (typeof saved.on === "boolean") refreshOn = saved.on;
      if (INTERVALS.indexOf(saved.every) >= 0) refreshEvery = saved.every;
    } catch (err) {
      // No storage: reload on the defaults.
    }
    $("every").value = String(refreshEvery);
    $("auto").onclick = () => { refreshOn = !refreshOn; keepRefresh(); };
    $("every").onchange = () => {
      refreshEvery = +$("every").value;
      keepRefresh();
    };
    // Coming back to the tab reloads at once, because background-tab timers are
    // throttled to about one a minute. The timer is deliberately not stopped on
    // `document.hidden`: an embedded browser pane can report hidden while on
    // screen.
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) refreshNow();
    });
    armRefresh();
  }

  function keepRefresh() {
    try {
      localStorage.setItem(
        REFRESH, JSON.stringify({on: refreshOn, every: refreshEvery}));
    } catch (err) {
      // Then the choice lasts the session.
    }
    armRefresh();
  }

  function armRefresh() {
    clearInterval(refreshTimer);
    refreshTimer = null;
    $("refresh").hidden = !reloader;
    $("auto").setAttribute("aria-pressed", refreshOn ? "true" : "false");
    $("every").disabled = !refreshOn;
    if (!reloader || !refreshOn) return;
    refreshTimer = setInterval(refreshNow, refreshEvery * 1000);
  }

  function refreshNow() {
    // Sit one out if the screen changed under us, the last reload has not come
    // back, or a menu is open -- reloading under an open menu shuts it in the
    // reader's hand.
    if (!reloader || refreshing) return;
    if (document.querySelector(".tabulator-menu, .tabulator-edit-list")) return;
    refreshing = true;
    // `new Promise` plus `finally` so the flag comes back down on every path,
    // including a synchronous throw out of `reloader()`.
    new Promise((resolve) => resolve(reloader()))
      .then(pulse, () => {
        // A server that went away, or a run directory read halfway through
        // being written; the next tick tries again.
      })
      .finally(() => { refreshing = false; });
  }

  /* Something has to say the timer is alive, or "auto" is a button that appears
     to do nothing on a directory where nothing is happening. */
  function pulse() {
    const button = $("auto");
    button.classList.remove("pulse");
    void button.offsetWidth;      // restarts the animation rather than ignoring
    button.classList.add("pulse");
  }

  // --- the index ------------------------------------------------------------

  let indexTable = null;             // the Tabulator over the runs, or null
  let indexHost = null;              // the element it was built in
  let indexRows = [];                // what it was built from
  let filtersOn = false;             // a filter box per column
  const folded = readFolded();       // comparisons the reader has shut, by path

  /* --- diff mode ---
     Tick some rows and the table keeps only the columns whose cells are not all
     the same. The picks are keyed by path rather than held on the rows, because
     a reload builds every row again; `restoring` is up while they are put back,
     so `replaceData`'s deselection is not read as the reader unticking
     everything. Not persisted: a comparison is something you are doing now, not
     a layout you painted. */
  let diffOn = false;
  let restoring = false;
  let pickSeq = 0;
  const picked = new Map();          // path -> the order it was ticked in

  /* Tabulator walks a sort list from the *end*, so the last entry is the
     primary one and this reads backwards on purpose. */
  const PICKED_FIRST = [{column: "started_at", dir: "desc"},
                        {column: "_picked", dir: "asc"}];

  /* What the mode replaces on the index's table. `persistence: false` is
     load-bearing: a persisting table would store the hidden columns as the
     reader's own choice. `dataTree: false` because Tabulator only ever sorts a
     child inside its own parent, so "picked rows first" cannot mean anything
     under a tree. */
  const DIFF = {
    dataTree: false,
    persistence: false,
    selectableRows: true,
    rowHeader: {formatter: "rowSelection", titleFormatter: "rowSelection",
                headerSort: false, resizable: false, width: 34, cssClass: "pick"},
    initialSort: PICKED_FIRST,
  };

  /* A comparison and its episodes as one flat list. */
  function flatten(rows) {
    return rows.reduce((all, row) => all.concat(
      row.is_compare ? [row, ...(row.episodes || [])] : [row]), []);
  }

  function rankOf(row) {
    const at = picked.get(row.getData().path);
    // A finite sentinel, not Infinity: two unpicked rows have to subtract to 0,
    // and Infinity - Infinity is NaN.
    return at === undefined ? Number.MAX_SAFE_INTEGER : at;
  }

  function diffColumns() {
    const columns = COLUMNS.map((col) => {
      const def = definition(col, false);
      // Which columns are up is the diff's answer and the order is "picked
      // first"; a header click would quietly replace both.
      delete def.headerMenu;
      def.headerSort = false;
      return def;
    });
    // Never drawn: the rank lives in `picked` and the sorter reads it there, so
    // a reload cannot lose it.
    columns.push({field: "_picked", title: "", visible: false, headerSort: false,
                  sorter: (a, b, rowA, rowB) => rankOf(rowA) - rankOf(rowB)});
    return columns;
  }

  /* What a cell says, for the diff to compare. The outcome column is a ladder
     over the whole row rather than one field, so it answers for itself. */
  function cellText(col, row) {
    return col.kind === "outcome" ? outcomeText(row) : textOf(col, row[col.field]);
  }

  /* Which columns to keep: the ones whose text differs across the basis. `when`
     is always kept, since it is what names a row. */
  function differing(rows) {
    const shown = new Set(["started_at"]);
    COLUMNS.forEach((col) => {
      const seen = new Set(rows.map((row) => cellText(col, row)));
      if (seen.size > 1) shown.add(col.field);
    });
    return shown;
  }

  /* The mode, applied: which columns are up and which rows are at the top. Run
     on every change to the picks and after every reload. */
  function applyDiff() {
    if (!diffOn || !indexTable) return;
    const rows = indexTable.getData();
    // Two or more picks is a comparison and the columns answer to it; otherwise
    // the whole directory is the basis.
    const basis = picked.size > 1 ? rows.filter((row) => picked.has(row.path)) : rows;
    const shown = differing(basis);
    COLUMNS.forEach((col) => {
      const column = indexTable.getColumn(col.field);
      const want = shown.has(col.field);
      if (!column || column.isVisible() === want) return;
      if (want) indexTable.showColumn(col.field);
      else indexTable.hideColumn(col.field);
    });
    indexTable.setSort(PICKED_FIRST);
    if (indexHost) indexHost.classList.toggle("picking", picked.size > 0);
    refit();
    showCount();
  }

  /* The reader ticked or unticked something. The event carries every selected
     row, so the Map is brought into step with it rather than patched. */
  function repicked(rows) {
    if (restoring) return;
    const now = rows.map((row) => row.getData().path);
    [...picked.keys()].forEach((path) => {
      if (now.indexOf(path) < 0) picked.delete(path);
    });
    now.forEach((path) => { if (!picked.has(path)) picked.set(path, ++pickSeq); });
    applyDiff();
  }

  /* Put the picks back on rows built after a reload, and forget the ones whose
     run is gone: a rank pointing at nothing would sort forever at the top. */
  function repick() {
    const rows = indexTable.getRows();
    const here = rows.map((row) => row.getData().path);
    [...picked.keys()].forEach((path) => {
      if (here.indexOf(path) < 0) picked.delete(path);
    });
    rows.forEach((row) => {
      if (picked.has(row.getData().path)) row.select();
    });
  }

  function readFolded() {
    try {
      const saved = JSON.parse(localStorage.getItem(FOLDED) || "[]");
      return new Set(Array.isArray(saved) ? saved : []);
    } catch (err) {
      // No storage, or something else under the key: everything starts open.
      return new Set();
    }
  }

  function keepFolded() {
    // Only the paths still in the directory, or the list would carry every
    // comparison anyone had ever shut.
    const here = new Set(indexRows.map((row) => row.path));
    try {
      localStorage.setItem(
        FOLDED, JSON.stringify([...folded].filter((path) => here.has(path))));
    } catch (err) {
      // Then the fold lasts the session.
    }
  }

  function indexHeader(rows) {
    const comparisons = rows.filter((r) => r.is_compare).length;
    const runs = rows.length - comparisons;
    setHeader("Space Invader runs",
      [count(runs, "run"), comparisons ? count(comparisons, "comparison") : null]
        .filter(Boolean).join(" · "));
  }

  function renderIndex(rows) {
    indexRows = rows;
    indexHeader(rows);
    $("scrubber").hidden = true;
    $("back").hidden = true;
    dropTables();
    indexTable = null;
    screen.textContent = "";

    if (!rows.length) {
      $("toolbar").hidden = true;
      const box = el("div", "empty");
      box.appendChild(el("p", null, "No runs here yet."));
      const how = el("p");
      how.appendChild(document.createTextNode("Play one: "));
      how.appendChild(el("code", null, "uv run agent --player random"));
      box.appendChild(how);
      screen.appendChild(box);
      // Nothing to reload in place, so the first run to land builds the table.
      reloader = () => get("/api/index").then((fresh) => {
        if (fresh.length) renderIndex(fresh);
      });
      armRefresh();
      return;
    }

    const lane = el("section", "lane wide");
    const host = el("div");
    indexHost = host;
    lane.appendChild(host);
    screen.appendChild(lane);         // in the document before Tabulator measures it

    $("toolbar").hidden = false;
    indexTable = keep(new Tabulator(host, {
      data: diffOn ? flatten(rows) : rows,
      columns: diffOn ? diffColumns() : COLUMNS.map((col) => definition(col, filtersOn)),
      columnDefaults: {
        // An absence is never the interesting end of a column.
        sorterParams: {alignEmptyValues: "bottom"},
        headerSortTristate: true,
      },
      // `fitColumns` shares spare width and, once the `minWidth` floors add up
      // to more than the pane, overflows into a scroller instead of squeezing.
      layout: "fitColumns",
      movableColumns: true,
      // A comparison is the head of its episodes, and sorting keeps it that way.
      dataTree: true,
      dataTreeChildField: "episodes",
      dataTreeSort: true,
      // A comparison the reader shut stays shut: `replaceData` builds every row
      // again and asks this afresh, so a plain `true` would be the fold
      // springing open on the next reload.
      dataTreeStartExpanded: (row) => !folded.has(row.getData().path),
      dataTreeElementColumn: "started_at",
      initialSort: [{column: "started_at", dir: "desc"}],
      // Not widths: a stored width outlives `minWidth`, so a header clipped once
      // would stay clipped forever. Not `filter`: Tabulator stores only the
      // programmatic filter, which `tableBuilt` clears on every load.
      persistence: {sort: true, columns: ["visible"]},
      persistenceID: LAYOUT,
      // A filter that matches nothing, or a runs directory emptied while it was
      // being watched.
      placeholder: "Nothing to show.",
      // Last, so it replaces the tree, the stored layout and the sort.
      ...(diffOn ? DIFF : {}),
    }));
    indexTable.on("rowClick", openRow);
    indexTable.on("rowSelectionChanged", (data, rows) => repicked(rows));
    indexTable.on("dataTreeRowCollapsed", (row) => {
      folded.add(row.getData().path);
      keepFolded();
    });
    indexTable.on("dataTreeRowExpanded", (row) => {
      folded.delete(row.getData().path);
      keepFolded();
    });
    indexTable.on("renderComplete", showCount);
    indexTable.on("tableBuilt", () => {
      drawGroups();
      // The mode opens on what varies across the whole directory.
      if (diffOn) applyDiff();
      else applySearch($("search").value.trim());
    });
    wireToolbar();

    reloader = () => get("/api/index").then((fresh) => {
      indexRows = fresh;
      indexHeader(fresh);
      // The picks are put back by path after the rebuild; `restoring` covers it
      // because `replaceData` wipes the selection along with the rows
      // (`rows-wipe`, whatever `selectableRowsPersistence` says).
      restoring = diffOn;
      // `replaceData` and not a rebuild: the sort, the header filters and the
      // chosen columns live on the table.
      return keepingPlace(
        host, () => indexTable.replaceData(diffOn ? flatten(fresh) : fresh))
        .then(() => {
          if (!diffOn) return;
          repick();
          restoring = false;
          applyDiff();
        })
        .finally(() => { restoring = false; });
    });
    armRefresh();
  }

  /* `replaceData` scrolls the table back to the left, and on a wide table
     reloaded every few seconds that drags the reader home every tick. The
     table's own scroller and not the window's: a reload adds rows, so the page
     scroll is not disturbed. */
  function keepingPlace(host, replace) {
    const holder = host.querySelector(".tabulator-tableholder");
    const left = holder ? holder.scrollLeft : 0;
    return Promise.resolve(replace()).then(() => {
      if (holder) holder.scrollLeft = left;
    });
  }

  function openRow(event, row) {
    // Clicking the expand arrow means "show me the episodes", not "open the
    // comparison".
    if (event.target.closest(".tabulator-data-tree-control")) return;
    // In the mode the whole row is the tick box; opening a run means leaving
    // the mode, and one stray click should not throw the comparison away.
    if (diffOn) return row.toggleSelect();
    const data = row.getData();
    if (data.unreadable) return;
    go((data.is_compare ? "compare/" : "run/") + data.path);
  }

  /* Counted on `renderComplete` rather than `dataFiltered`, which fires before
     `getDataCount` has caught up and, under `dataTree`, again per branch. */
  function showCount() {
    if (!indexTable) return;
    if (diffOn) {
      // Rows are never hidden here, so the number worth printing is how many
      // columns the diff put away.
      const up = COLUMNS.filter((col) => {
        const column = indexTable.getColumn(col.field);
        return column && column.isVisible();
      }).length;
      $("shown").textContent =
        (picked.size ? count(picked.size, "row") + " picked · " : "")
        + up + " of " + COLUMNS.length + " columns";
      return;
    }
    const shown = indexTable.getDataCount("active");
    $("shown").textContent =
      shown === indexRows.length ? "" : shown + " of " + indexRows.length;
  }

  // --- the toolbar ----------------------------------------------------------

  function wireToolbar() {
    $("search").oninput = () => applySearch($("search").value.trim());
    $("diff").onclick = () => {
      diffOn = !diffOn;
      // Entering, search and filters are put away -- they do not compose with
      // a fixed comparison; leaving, the picks go with the mode.
      if (diffOn) {
        $("search").value = "";
        filtersOn = false;
      } else {
        picked.clear();
      }
      // The rebuild `filters` takes, for the reason given there.
      renderIndex(indexRows);
    };
    $("filters").onclick = () => {
      filtersOn = !filtersOn;
      // Rebuilt rather than patched: turning a filter box off has to drop the
      // filter it held, and one path known to leave a consistent table beats
      // two that nearly do.
      renderIndex(indexRows);
    };
    $("reset").onclick = () => {
      // Persistence is per browser and beats the defaults, so a painted-in
      // layout needs a way out that is not devtools.
      Object.keys(localStorage)
        .filter((key) => key.indexOf(STORED) === 0)
        .forEach((key) => localStorage.removeItem(key));
      // The key is gone; the set it was read from is not.
      folded.clear();
      filtersOn = false;
      diffOn = false;
      picked.clear();
      $("search").value = "";
      renderIndex(indexRows);
    };
    $("filters").setAttribute("aria-pressed", filtersOn ? "true" : "false");
    $("diff").setAttribute("aria-pressed", diffOn ? "true" : "false");
    // Both choose which rows are on screen, which is the mode's business while
    // it is on.
    $("filters").disabled = diffOn;
    $("search").disabled = diffOn;
  }

  function applySearch(term) {
    if (!indexTable) return;
    if (!term) return indexTable.clearFilter();     // header filters survive
    // One nested array is Tabulator's OR, so this is "any of these fields".
    indexTable.setFilter([
      SEARCH.map((field) => ({field: field, type: "like", value: term})),
    ]);
  }

  /* One chip per group, pressed when any of its columns is showing; a chip sets
     the whole group, so a half-hidden group still has one click that shows all
     of it. */
  function drawGroups() {
    const box = $("groups");
    box.textContent = "";
    GROUPS.forEach((group) => {
      const fields = COLUMNS.filter((c) => c.group === group).map((c) => c.field);
      const on = fields.some((field) => {
        const column = indexTable && indexTable.getColumn(field);
        return column && column.isVisible();
      });
      const chip = el("button", "chip", group);
      chip.type = "button";
      chip.setAttribute("aria-pressed", on ? "true" : "false");
      // The diff has already answered which columns are up; shown rather than
      // hidden so the reader can see which groups they belong to.
      chip.disabled = diffOn;
      chip.onclick = () => {
        fields.forEach((field) =>
          on ? indexTable.hideColumn(field) : indexTable.showColumn(field));
        refit();
        drawGroups();
      };
      box.appendChild(chip);
    });
  }

  /* Showing or hiding a column does not re-run the layout; a redraw asks
     `fitColumns` to share the width out again over what is showing. */
  function refit() {
    if (indexTable) indexTable.redraw();
  }

  function count(n, noun) {
    return n + " " + noun + (n === 1 ? "" : "s");
  }

  // --- a comparison ---------------------------------------------------------

  /* The two tables here sort but do not hide columns: each is a handful of
     columns chosen for one question, and there is nothing to unclutter. */
  function sortable(host, rows, columns, id, sort) {
    return keep(new Tabulator(host, {
      data: rows,
      columns: withFloors(columns),
      columnDefaults: {sorterParams: {alignEmptyValues: "bottom"},
                       headerSortTristate: true},
      layout: "fitColumns",
      initialSort: sort,
      persistence: {sort: true},
      persistenceID: id,
    }));
  }

  const numeric = (field, title, render) => ({
    field: field, title: title, sorter: "number", hozAlign: "right",
    formatter: (cell) => (empty(cell.getValue()) ? ""
      : render ? render(cell.getValue(), cell.getRow().getData())
      : String(cell.getValue())),
  });

  function renderCompare(data, path) {
    const meta = data.meta || {};
    setHeader("compare · " + (meta.players || []).join(" "),
              meta.seeds + " seeds · " + meta.max_steps + " max steps · " + (meta.model || ""));
    $("scrubber").hidden = true;
    $("toolbar").hidden = true;
    $("back").hidden = false;
    dropTables();
    screen.textContent = "";

    const players = el("section", "lane wide");
    players.appendChild(el("h2", null, "Players"));
    const playersHost = el("div");
    players.appendChild(playersHost);
    screen.appendChild(players);
    const playersTable = sortable(playersHost, data.players || [], [
      {field: "player", title: "player"},
      numeric("score", "score", (v, row) => v + " ± " + row.score_sd),
      numeric("decisions", "steps"),
      numeric("wins", "wins", (v, row) => v + "/" + row.episodes),
      numeric("aimed", "aimed", (v) => v + "%"),
      numeric("unparseable", "unparsed"),
      numeric("per_decision", "s/dec", (v) => Number(v).toFixed(2)),
      numeric("tokens", "tokens", (v) => Number(v).toLocaleString()),
    ], "invader-compare-players-v1", [{column: "score", dir: "desc"}]);

    const episodes = el("section", "lane wide");
    episodes.appendChild(el("h2", null, "Episodes"));
    const episodesHost = el("div");
    episodes.appendChild(episodesHost);
    screen.appendChild(episodes);
    const episodesTable = sortable(episodesHost, data.episodes || [], [
      // The column that names the episode, so it does not lose the end of every
      // name to an ellipsis.
      {field: "run", title: "episode", minWidth: 165, widthGrow: 2,
       formatter: (cell) => el("span", "mono", cell.getValue())},
      {field: "outcome", title: "outcome",
       formatter: (cell) => outcomeCell(cell.getRow().getData())},
      numeric("score", "score"),
      numeric("ticks", "ticks"),
      numeric("aimed", "aimed", (v) => v + "%"),
      numeric("unparseable", "unparsed"),
      numeric("decisions", "decisions"),
      numeric("tokens", "tokens", (v) => Number(v).toLocaleString()),
    ], "invader-compare-episodes-v1", [{column: "run", dir: "asc"}]);
    episodesTable.on(
      "rowClick", (event, row) => go("run/" + row.getData().path));

    reloader = () => get("/api/compare/" + path).then((fresh) => Promise.all([
      keepingPlace(playersHost, () => playersTable.replaceData(fresh.players || [])),
      keepingPlace(episodesHost, () => episodesTable.replaceData(fresh.episodes || [])),
    ]));
    armRefresh();
  }

  // --- one run --------------------------------------------------------------

  function renderRun(data) {
    run = data;
    at = 0;
    pieces = {};
    folds = {};
    const meta = data.meta || {}, summary = data.summary;
    setHeader(describe(meta) || meta.run,
              summary
                ? [summary.outcome, "score " + summary.score, summary.lives + " lives",
                   summary.decisions + " decisions", summary.unparseable + " unparsed"]
                  .join(" · ")
                : "unfinished — no summary was written");
    $("back").hidden = !!DATA;
    $("toolbar").hidden = true;
    // A run is a recording and the scrubber is a place in it; a reload would
    // have to decide where the cursor goes when twenty frames arrive.
    reloader = null;
    armRefresh();
    dropTables();
    screen.textContent = "";

    const lanes = el("div", "lanes");
    played = {w: meta.width || W, h: meta.height || H};
    lanes.appendChild(boardLane(played.w, played.h));
    lanes.appendChild(decisionLane());
    screen.appendChild(lanes);
    if (data.kernel) screen.appendChild(kernelLane());

    wireScrubber();
    show(0);
  }

  function boardLane(w, h) {
    const lane = el("section", "lane");
    lane.appendChild(el("h2", null, "Board"));
    const board = svg("svg", {
      id: "board", viewBox: "0 0 " + w * CELL + " " + h * CELL,
      role: "img", "aria-label": "the board at the tick under the cursor"});
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        board.appendChild(svg("rect", {
          class: "cell", x: col * CELL, y: row * CELL, width: CELL, height: CELL}));
      }
    }
    lane.appendChild(board);
    const readout = el("div", "readout");
    readout.id = "readout";
    lane.appendChild(readout);
    return lane;
  }

  function decisionLane() {
    const lane = el("section", "lane warm");
    lane.appendChild(el("h2", null, "Decision"));
    const body = el("div");
    body.id = "decision";
    lane.appendChild(body);
    return lane;
  }

  function kernelLane() {
    const lane = el("section", "lane cool");
    lane.appendChild(el("h2", null, "Kernel"));
    const body = el("div");
    body.id = "kernel";
    lane.appendChild(body);
    lane.appendChild(deeper());
    return lane;
  }

  // One snapshot per world tick, so this pane can say the kernel was busy but
  // never in what order; that and the wiring are `zeos debug`'s, and an
  // exported page has to say so.
  function deeper() {
    const note = el("p", "deeper");
    const where = (run.meta || {}).case;
    note.appendChild(document.createTextNode(
      "One snapshot per world tick. For the order of events inside a tick, or "
      + "the case's wiring, "));
    if (!where) {
      note.appendChild(document.createTextNode(
        "run zeos debug against this run's case — recorded in meta.json as "
        + "\u201ccase\u201d on runs made after 2026-09-01."));
      return note;
    }
    note.appendChild(document.createTextNode("run:"));
    note.appendChild(el("code", null,
      "zeos debug " + where + " --journal <run>/kernel.jsonl"));
    return note;
  }

  // --- drawing one tick -----------------------------------------------------

  function show(tick) {
    const span = run.frames.length;
    at = Math.max(0, Math.min(span - 1, tick));
    const frame = run.frames[at];
    $("slider").value = at;
    $("at").textContent = at + " / " + (span - 1);
    drawBoard(frame);
    drawReadout(frame);
    drawDecision(at);
    if (run.kernel) drawKernel(at);
    moveCursor();
  }

  function place(key, node, col, row) {
    if (!pieces[key]) {
      $("board").appendChild(node);
      pieces[key] = node;
    }
    pieces[key].setAttribute(
      "transform", "translate(" + col * CELL + "," + row * CELL + ")");
    pieces[key].dataset.seen = "1";
    return pieces[key];
  }

  function drawBoard(frame) {
    Object.keys(pieces).forEach((k) => { delete pieces[k].dataset.seen; });

    Object.keys(frame.monsters).forEach((id) => {
      const [row, col] = frame.monsters[id];
      let node = pieces["m" + id];
      if (!node) {
        node = svg("g", {class: "monster"});
        node.appendChild(svg("rect", {
          x: 5, y: 10, width: CELL - 10, height: CELL - 20, rx: 3}));
        const label = svg("text", {
          x: CELL / 2, y: CELL / 2 + 3.5, "text-anchor": "middle",
          "font-size": 11, "font-family": "ui-monospace, monospace"});
        label.textContent = "m" + id;
        node.appendChild(label);
      }
      place("m" + id, node, col, row);
    });

    let player = pieces.p;
    if (!player) {
      player = svg("path", {
        class: "player",
        d: "M" + (CELL / 2) + " 8 L" + (CELL - 6) + " " + (CELL - 8) +
           " L6 " + (CELL - 8) + " Z"});
    }
    place("p", player, frame.player, played.h - 1);

    if (frame.missile) {
      let shot = pieces.shot;
      if (!shot) {
        shot = svg("rect", {
          class: "shot", x: CELL / 2 - 2, y: 8, width: 4, height: CELL - 16, rx: 2});
      }
      place("shot", shot, frame.missile[1], frame.missile[0]);
    }

    (frame.dangers || []).forEach(([row, col], i) => {
      let danger = pieces["d" + i];
      if (!danger) {
        danger = svg("circle", {class: "danger", cx: CELL / 2, cy: CELL / 2, r: 6});
      }
      place("d" + i, danger, col, row);
    });

    Object.keys(pieces).forEach((key) => {
      if (!pieces[key].dataset.seen) {
        pieces[key].remove();
        delete pieces[key];
      }
    });
  }

  function drawReadout(frame) {
    const out = $("readout");
    out.textContent = "";
    const add = (label, value, cls) => {
      const span = el("span", cls);
      span.appendChild(document.createTextNode(label + " "));
      span.appendChild(el("b", null, value));
      out.appendChild(span);
    };
    add("score", frame.score);
    add("lives", frame.lives);
    add("left", Object.keys(frame.monsters).length);
    if (frame.over) {
      out.appendChild(el("span", "over", frame.won ? "YOU WIN" : "GAME OVER"));
    }
  }

  function drawDecision(tick) {
    const box = $("decision");
    box.textContent = "";
    const landed = run.by_tick[tick] || [];
    const waiting = run.in_flight[tick];

    landed.forEach((index) => box.appendChild(decisionCard(run.decisions[index], tick)));

    if (waiting !== null && waiting !== undefined) {
      const d = run.decisions[waiting];
      const waited = tick - d.tick;
      // A labelled block of its own: a card ends in an open fold whose `pre` has
      // no visible end, so a trailing line reads as the last line of what we
      // sent.
      const note = el("div", "waiting");
      note.appendChild(el("div", "pending", "still thinking"));
      const line = el("p");
      line.appendChild(el("b", null, d.by));
      line.appendChild(document.createTextNode(
        waited === 0
          ? " was asked on this tick and has not answered. It lands on tick " +
            d.tick_applied + ", " + count(d.tick_applied - tick, "tick") + " from here."
          : " was asked on tick " + d.tick + ", " + count(waited, "tick") +
            " ago. It lands on tick " + d.tick_applied + "."));
      note.appendChild(line);
      box.appendChild(note);
    } else if (!landed.length) {
      box.appendChild(el("p", "idle", "Nothing was chosen on this tick."));
    }
  }

  function decisionCard(d, tick) {
    const card = el("div");
    const head = el("div", "headline");
    head.appendChild(el("span", "who " + d.by, d.by));
    head.appendChild(el("span", "action", d.action));
    if (d.parsed === false) head.appendChild(el("span", "flag bad", "unreadable reply"));
    if (d.applied === false) head.appendChild(el("span", "flag bad", "too late"));
    if (d.preempted) head.appendChild(el("span", "flag cool", "pass abandoned"));
    if (d.retried) head.appendChild(el("span", "flag", "retried"));
    card.appendChild(head);

    const facts = el("dl", "facts");
    const late = d.tick_applied - d.tick;
    const fact = (label, value, cls) => {
      if (value === null || value === undefined) return;
      facts.appendChild(el("dt", null, label));
      facts.appendChild(el("dd", cls, value));
    };
    fact("chose on", "tick " + d.tick + (late > 0 ? " — the board " + late +
         " tick" + (late === 1 ? "" : "s") + " before this one" : ""),
         late > 0 ? "late" : null);
    fact("latency", seconds(d.latency));
    fact("tokens", tokensOf(d.usage));
    fact("stopped", d.stop_reason);
    if (d.kernel_ticks !== undefined) fact("kernel", d.kernel_ticks + " boundaries");
    card.appendChild(facts);

    if (d.reasoning) card.appendChild(fold("Reasoning", d.reasoning));
    if (d.reply) card.appendChild(fold("Reply", d.reply));
    if (d.prompt) {
      card.appendChild(fold("What the model was sent", d.prompt, tick === 0));
    }
    return card;
  }

  function fold(label, text, open) {
    const box = el("details");
    // The cards are rebuilt on every tick, so a fold the reader opened has to
    // be reopened here; `folds` is what they said, `open` only the default.
    box.open = label in folds ? folds[label] : !!open;
    box.addEventListener("toggle", () => { folds[label] = box.open; });
    box.appendChild(el("summary", null, label));
    box.appendChild(el("pre", null, text));
    return box;
  }

  function drawKernel(tick) {
    const box = $("kernel");
    box.textContent = "";
    const view = run.kernel[tick];
    if (!view) return;

    box.appendChild(table(
      ["job", "state", {label: "prio", num: true}, "blocked on", {label: "tok", num: true}],
      (view.jobs || []).map((job) => cells([
        {text: job.name, cls: "who " + job.name},
        job.state,
        {text: job.priority, cls: "num"},
        {text: job.blocked_on, cls: "mono"},
        {text: job.tokens, cls: "num"}], job.alive ? null : "dim"))));

    const counts = el("div", "readout");
    const add = (label, value, title) => {
      const span = el("span");
      if (title) span.title = title;
      span.appendChild(document.createTextNode(label + " "));
      span.appendChild(el("b", null, value));
      counts.appendChild(span);
    };
    // Token counts kept by the kernel about its own machine, not model calls:
    // zeos calls a decoded token a "forward pass", which in this project means
    // one call to the served model.
    add("kernel decoded", view.passes.toLocaleString(),
        "tokens the scheduled tree decoded -- not calls to the model");
    add("kernel context", view.tokens.toLocaleString(),
        "tokens decoded plus tokens injected into contexts");
    add("preemptions", view.preemptions,
        "kernel preemptions: the reflex took the machine off the pilot "
        + "mid-reply. Only a running job can be preempted, which is why the "
        + "pilot reads its answer a piece at a time");
    box.appendChild(counts);

    const ticker = el("ol", "ticker");
    (run.kernel_events[tick] || []).forEach((e) => ticker.appendChild(el("li", null, e.text)));
    if (!ticker.children.length) ticker.appendChild(el("li", null, "the kernel did nothing this tick"));
    box.appendChild(ticker);
  }

  // --- the scrubber ---------------------------------------------------------

  function wireScrubber() {
    const span = run.frames.length;
    $("scrubber").hidden = false;
    $("slider").max = Math.max(0, span - 1);
    $("slider").value = 0;

    const marks = $("marks");
    marks.textContent = "";
    marks.appendChild(new Option("jump to…", ""));
    (run.marks || []).forEach((m) => marks.appendChild(
      new Option(m.label + " @ " + m.tick, m.tick)));
    marks.onchange = () => {
      if (marks.value !== "") { play(false); show(+marks.value); }
      marks.value = "";
    };

    $("slider").oninput = (e) => { play(false); show(+e.target.value); };
    $("back-one").onclick = () => { play(false); show(at - 1); };
    $("fwd-one").onclick = () => { play(false); show(at + 1); };
    $("play").onclick = () => play(!playing);
    drawSignals();
  }

  function play(on) {
    playing = on;
    $("play").setAttribute("aria-pressed", on ? "true" : "false");
    $("play").innerHTML = on ? "&#10073;&#10073;" : "&#9654;";
    clearInterval(timer);
    if (!on) return;
    timer = setInterval(() => {
      if (at >= run.frames.length - 1) return play(false);
      show(at + 1);
    }, PLAY_MS);
  }

  /* Three channels over the whole episode: how long the chooser was waiting,
   * who moved the stick, and where it mattered -- lateness is a shape, not a
   * number. */
  function drawSignals() {
    const span = run.frames.length;
    const strip = $("signals");
    strip.textContent = "";
    strip.setAttribute("viewBox", "0 0 " + span + " 30");

    for (let tick = 10; tick < span; tick += 10) {
      strip.appendChild(svg("line", {
        x1: tick, x2: tick, y1: 0, y2: 30, stroke: "var(--rule)",
        "stroke-width": 1, "vector-effect": "non-scaling-stroke"}));
    }

    run.decisions.forEach((d) => {
      const late = d.tick_applied - d.tick;
      if (late > 0) {
        // Inset, so back-to-back waits read as two spans rather than one bar.
        strip.appendChild(svg("rect", {
          x: d.tick + 0.12, y: 1, width: Math.max(0.2, late - 0.24), height: 7,
          fill: "var(--deliberate)", opacity: 0.55}));
      }
      // `held` is the absence of a write in recordings made before the stick
      // was edge-triggered; marking every one would draw a solid bar saying
      // nothing.
      if (d.by === "held") return;
      const colour = d.by === "evade" ? "var(--reflex)"
        : (d.by === "pilot" || d.by === "model") ? "var(--deliberate)"
        : "var(--ink-dim)";
      strip.appendChild(svg("rect", {
        x: d.tick_applied, y: 11, width: 1, height: 7,
        fill: colour, opacity: d.applied === false ? 0.3 : 1}));
    });

    (run.marks || []).forEach((m) => {
      const colour = m.label === "life lost" ? "var(--hurt)"
        : m.label === "kill" ? "var(--kill)"
        : m.label === "reflex" ? "var(--reflex)"
        : "var(--ink-dim)";
      strip.appendChild(svg("rect", {
        x: m.tick, y: 21, width: 1, height: 7, fill: colour}));
    });

    const cursor = svg("line", {
      id: "cursor", x1: 0, x2: 0, y1: 0, y2: 30,
      stroke: "var(--cursor)", "stroke-width": 1,
      "vector-effect": "non-scaling-stroke"});
    strip.appendChild(cursor);

    let dragging = false;
    const seek = (event) => {
      const box = strip.getBoundingClientRect();
      const fraction = (event.clientX - box.left) / box.width;
      play(false);
      show(Math.round(fraction * (span - 1)));
    };
    strip.addEventListener("pointerdown", (e) => {
      dragging = true; strip.setPointerCapture(e.pointerId); seek(e);
    });
    strip.addEventListener("pointermove", (e) => { if (dragging) seek(e); });
    strip.addEventListener("pointerup", () => { dragging = false; });
  }

  function moveCursor() {
    const cursor = $("cursor");
    if (!cursor) return;
    cursor.setAttribute("x1", at + 0.5);
    cursor.setAttribute("x2", at + 0.5);
  }

  // --- routing --------------------------------------------------------------

  function setHeader(title, subtitle) {
    $("title").textContent = title;
    $("subtitle").textContent = subtitle || "";
    document.title = title === "Space Invader runs" ? title
      : title + " — Space Invader runs";
  }

  function go(hash) { location.hash = "#/" + hash; }

  function fail(message) {
    $("scrubber").hidden = true;
    $("toolbar").hidden = true;
    reloader = null;
    armRefresh();
    dropTables();
    screen.textContent = "";
    const box = el("div", "empty");
    box.appendChild(el("p", null, message));
    screen.appendChild(box);
  }

  function load() {
    play(false);
    // The screen is changing, so whatever it knew how to reload is gone until
    // the next one says otherwise.
    reloader = null;
    armRefresh();
    const hash = location.hash.replace(/^#\/?/, "");
    const cut = hash.indexOf("/");
    const kind = cut < 0 ? hash : hash.slice(0, cut);
    const path = cut < 0 ? "" : hash.slice(cut + 1);

    if (!kind) return get("/api/index").then(renderIndex).catch(offline);
    if (kind === "run") return get("/api/run/" + path).then(renderRun).catch(offline);
    if (kind === "compare") {
      return get("/api/compare/" + path)
        .then((data) => renderCompare(data, path)).catch(offline);
    }
    fail("No such screen: " + kind);
  }

  function get(url) {
    return fetch(url).then((r) => {
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.json();
    });
  }

  function offline(err) {
    fail("Could not read that run: " + err.message +
         ". The viewer serves runs/ — start it with `uv run viewer`.");
  }

  function theme() {
    const root = document.documentElement;
    const dark = root.getAttribute("data-theme") === "dark" ||
      (!root.hasAttribute("data-theme") &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    root.setAttribute("data-theme", dark ? "light" : "dark");
  }

  document.addEventListener("keydown", (e) => {
    if (!run || $("scrubber").hidden) return;
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft") { play(false); show(at - 1); }
    else if (e.key === "ArrowRight") { play(false); show(at + 1); }
    else if (e.key === " ") { e.preventDefault(); play(!playing); }
    else if (e.key === "Home") { play(false); show(0); }
    else if (e.key === "End") { play(false); show(run.frames.length - 1); }
  });

  $("theme").onclick = theme;

  wireRefresh();

  if (DATA && DATA.route === "run") {
    renderRun(DATA.run);
  } else {
    window.addEventListener("hashchange", load);
    load();
  }
})();
