// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/* Draws what the payload says.
 *
 * No fold and no arithmetic on the journal -- that lives in payload.py, where pytest
 * can reach it. The one computation here is the delta merge, and it is a mechanical
 * mirror of `payload.apply_delta`, whose losslessness is proved in Python rather
 * than hoped for in a browser. Everything else is layout, which is presentation and
 * belongs on this side of the line.
 *
 * Scrubbing is an index. A recorded run is a recording, so stepping backwards costs
 * the same as stepping forwards.
 */
(function () {
  "use strict";

  /* Sections of a frame that are keyed lists, so a delta can carry one row rather
   * than the whole list. Must agree with payload.KEYED. */
  var KEYED = { jobs: "job", pipes: "name", vectors: "name", resources: "name" };

  /* Reconstructing every frame from the base costs one merge per step, so a full
   * keyframe is kept every STRIDE frames: seeking anywhere is then at most STRIDE
   * merges, and holding them costs a fraction of holding every frame. */
  var STRIDE = 64;

  /* The token stream is capped at what a person can actually read. Everything
   * before it is still in the payload -- scrub back to it, or filter to one job. */
  var STREAM_ROWS = 200;

  /* Movement -> the glyph and class the row is drawn with. Must agree with
   * payload.MOVEMENTS. */
  var MOVES = {
    decode: { glyph: "gen", cls: "gen" },
    inject: { glyph: "in", cls: "in" },
    write: { glyph: "\u2192", cls: "write" },
    read: { glyph: "\u2190", cls: "read" }
  };

  var STATES = [
    "running", "ready", "blocked", "suspended", "pinned_idle", "faulted", "done"
  ];

  /* The gaps are wide because every edge carries its pipe's name, and a label that
   * lands on a box is worse than no label. GUTTER and GAP_X are what the midpoint of
   * a horizontal edge has to sit in. */
  var NODE_W = 210, NODE_H = 78, GAP_X = 132, GAP_Y = 52;
  var CHIP_W = 156, CHIP_H = 36, GUTTER = 124, PAD = 26;
  /* Priority is the diagram's vertical axis, so it gets an axis column of its own
   * rather than a caption floating over the left gutter's chips. */
  var AXIS = 84;

  var $ = function (id) { return document.getElementById(id); };

  var payload = null, S = null, F = null;
  var at = 0, frame = null, previous = null;
  /* Which slice of the token log the stream pane shows: "" for everything,
   * "job:N" or "pipe:NAME". Not remembered across reloads -- unlike the edge
   * toggles it is a question about one run, not a way of reading the diagram. */
  var tokenFilter = "";
  var keyframes = [], nodes = {}, chips = {}, objects = {}, playing = false, timer = null;

  /* Which edge kinds are drawn. Layout ignores this entirely -- see buildLayout --
   * so hiding a kind subtracts lines from a fixed picture rather than redrawing a
   * different one. */
  var show = { pipes: true, vectors: true, maps: true };
  var STORE_KEY = "zeos.debugger.show";
  var WIDTH_KEY = "zeos.debugger.side";

  /* How narrow each side of the seam may get. The panes have a fixed two-column
   * table and a progress bar in them, so below MIN_SIDE they stop being readable
   * rather than merely getting tight; MIN_GRAPH keeps at least one node column and
   * its gutter of chips visible, which is the least that is still a diagram. */
  var MIN_SIDE = 240, MIN_GRAPH = 280;

  /* The width the reader actually asked for, before clamping. Kept separate from the
   * width in force so that a window too narrow to honour it borrows from the panes
   * temporarily and gives it back on the way out, rather than quietly overwriting the
   * choice with whatever fitted at the worst moment. */
  var desiredWidth = null;

  /* Remembered per browser, because a served page is reloaded constantly while the
   * case is being worked on and losing the filter every time is a papercut. Storage
   * can throw outright (a private window, site data blocked), so every touch of it
   * is guarded and the default survives. */
  function loadToggles() {
    try {
      var saved = JSON.parse(window.localStorage.getItem(STORE_KEY) || "null");
      if (saved) {
        Object.keys(show).forEach(function (k) {
          if (typeof saved[k] === "boolean") show[k] = saved[k];
        });
      }
    } catch (err) { /* no storage; the defaults stand */ }
  }

  function saveToggles() {
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(show));
    } catch (err) { /* nothing to do; the toggles still work for this session */ }
  }

  /* --- the divider --------------------------------------------------------- */

  /* One clamp, used by the drag, the keyboard and the restore, so a width can never
   * arrive by a route that skips the bounds. The upper bound is derived rather than
   * fixed: on a narrow window MIN_GRAPH is what decides how wide the panes may be. */
  function clampWidth(px) {
    var total = document.querySelector("main").getBoundingClientRect().width;
    var divider = $("divider").getBoundingClientRect().width;
    var widest = Math.max(MIN_SIDE, total - divider - MIN_GRAPH);
    return Math.round(Math.min(Math.max(px, MIN_SIDE), widest));
  }

  function setWidth(px, remember) {
    desiredWidth = px;
    return applyWidth(remember);
  }

  /* Re-clamps whatever was last asked for against the window as it is now. */
  function applyWidth(remember) {
    var width = clampWidth(desiredWidth);
    document.documentElement.style.setProperty("--side-w", width + "px");
    var divider = $("divider");
    divider.setAttribute("aria-valuenow", String(width));
    divider.setAttribute("aria-valuemax", String(clampWidth(Infinity)));
    if (remember) {
      try {
        window.localStorage.setItem(WIDTH_KEY, String(desiredWidth));
      } catch (err) { /* no storage; the width still holds for this session */ }
    }
    return width;
  }

  function sideWidth() {
    return $("side").getBoundingClientRect().width;
  }

  function installDivider() {
    var divider = $("divider");
    divider.setAttribute("aria-valuemin", String(MIN_SIDE));

    desiredWidth = sideWidth();  // the stylesheet's default, until told otherwise
    try {
      var saved = parseInt(window.localStorage.getItem(WIDTH_KEY) || "", 10);
      if (!isNaN(saved)) setWidth(saved, false);
    } catch (err) { /* no storage; the stylesheet default stands */ }

    /* Pointer events rather than mouse events: one code path covers a trackpad, a
     * touchscreen and a stylus, and capture means a fast drag that outruns the
     * cursor still lands on the divider rather than being lost to the pane. */
    divider.addEventListener("pointerdown", function (event) {
      event.preventDefault();
      divider.setPointerCapture(event.pointerId);
      divider.classList.add("dragging");
      document.body.classList.add("resizing");
    });

    divider.addEventListener("pointermove", function (event) {
      if (!divider.hasPointerCapture(event.pointerId)) return;
      /* Measured from the window's right edge, so the width tracks the pointer
       * exactly however the rest of the page is laid out. */
      setWidth(document.documentElement.clientWidth - event.clientX, false);
    });

    function release(event) {
      if (!divider.hasPointerCapture(event.pointerId)) return;
      divider.releasePointerCapture(event.pointerId);
      divider.classList.remove("dragging");
      document.body.classList.remove("resizing");
      /* Pin the asked-for width to the width actually on screen before persisting:
       * a drag that overshot the edge asked for a number nobody saw, and storing it
       * would have the panel unfold to it on a wider monitor. */
      setWidth(sideWidth(), true);
    }
    divider.addEventListener("pointerup", release);
    divider.addEventListener("pointercancel", release);

    divider.addEventListener("keydown", function (event) {
      var step = event.shiftKey ? 64 : 16;
      if (event.key === "ArrowLeft") setWidth(sideWidth() + step, true);
      else if (event.key === "ArrowRight") setWidth(sideWidth() - step, true);
      else if (event.key === "Home") setWidth(340, true);   // the designed default
      else return;
      event.preventDefault();
    });

    /* A window narrow enough to violate MIN_GRAPH borrows from the panes, and a
     * window that grows again hands it straight back. */
    window.addEventListener("resize", function () { applyWidth(false); });
  }

  /* --- small builders ------------------------------------------------------ */

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function svg(tag, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var key in attrs) {
      if (attrs[key] !== null && attrs[key] !== undefined) {
        node.setAttribute(key, String(attrs[key]));
      }
    }
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function byKey(a, b) {
    if (typeof a === "number" && typeof b === "number") return a - b;
    return String(a) < String(b) ? -1 : (String(a) > String(b) ? 1 : 0);
  }

  function nsToText(ns) {
    if (ns === null || ns === undefined) return "";
    if (ns >= 1e9) return (ns / 1e9).toFixed(2) + "s";
    if (ns >= 1e6) return (ns / 1e6).toFixed(1) + "ms";
    if (ns >= 1e3) return (ns / 1e3).toFixed(1) + "us";
    return ns + "ns";
  }

  /* --- frames -------------------------------------------------------------- */

  /* The mirror of payload.apply_delta. Rows are re-sorted by key because the fold
   * emits every section sorted, so merging in key order reproduces the frame the
   * kernel's monitor actually produced. */
  function merge(base, delta) {
    var out = {}, key;
    for (key in base) out[key] = base[key];
    for (key in delta) {
      var field = KEYED[key];
      if (!field) { out[key] = delta[key]; continue; }
      var rows = new Map();
      (out[key] || []).forEach(function (row) { rows.set(row[field], row); });
      delta[key].forEach(function (row) { rows.set(row[field], row); });
      out[key] = Array.from(rows.keys()).sort(byKey).map(function (k) { return rows.get(k); });
    }
    return out;
  }

  function frameAt(index) {
    if (!F || !F.count) return null;
    index = Math.max(0, Math.min(index, F.count - 1));
    var block = Math.floor(index / STRIDE);
    if (!keyframes[block]) {
      var from = block;
      while (from > 0 && !keyframes[from]) from--;
      var built = keyframes[from] || F.base;
      for (var n = from * STRIDE; n < block * STRIDE; n++) {
        built = merge(built, F.deltas[n]);
        if ((n + 1) % STRIDE === 0) keyframes[(n + 1) / STRIDE] = built;
      }
      keyframes[block] = built;
    }
    var current = keyframes[block];
    for (var i = block * STRIDE; i < index; i++) current = merge(current, F.deltas[i]);
    return current;
  }

  function jobsOf(name) {
    if (!frame) return [];
    return (frame.jobs || []).filter(function (j) { return j.descriptor === name; });
  }

  /* The job whose state the diagram shows for a descriptor. A reentrant vector can
   * have several instances alive at once; the most active one is the one worth
   * seeing, and the node carries a count when there is more than one. */
  function leadJob(name) {
    var candidates = jobsOf(name);
    if (!candidates.length) return null;
    var best = candidates[0];
    candidates.forEach(function (j) {
      if (STATES.indexOf(j.state) < STATES.indexOf(best.state)) best = j;
    });
    return best;
  }

  function pipeState(name) {
    if (!frame) return null;
    return (frame.pipes || []).find(function (p) { return p.name === name; }) || null;
  }

  /* The world as it stands, which is not the same as the frame's world.
   *
   * A case seeds its initial values straight into the store before the kernel starts,
   * so no WorldWritten event carries them and the fold has never seen them. The
   * declared values are the floor; the journal supplies every change on top. Reading
   * only the frame would show a case's opening state as unset, which is wrong in the
   * one direction that matters -- a resume diff is against these values. */
  function worldNow(view) {
    var out = {}, key;
    for (key in S.world) out[key] = S.world[key];
    if (view) for (key in (view.world || {})) out[key] = view.world[key];
    return out;
  }

  function wroteSince(name) {
    if (!frame || !previous) return false;
    var now = pipeState(name);
    var was = (previous.pipes || []).find(function (p) { return p.name === name; });
    return !!(now && was && now.written > was.written);
  }

  /* --- the wiring graph ---------------------------------------------------- */

  /* Which descriptors write a pipe and which read it, from the declared bindings.
   * Direction is the alias convention -- see payload.ALIAS_KIND, which documents
   * why an extra binding is drawn as both. */
  function roles(pipe) {
    var writers = [], readers = [], vectors = [];
    S.edges.forEach(function (e) {
      if (e.pipe !== pipe) return;
      if (e.kind === "write" || e.kind === "duplex") writers.push(e.descriptor);
      if (e.kind === "read" || e.kind === "duplex") readers.push(e.descriptor);
      if (e.kind === "interrupt") vectors.push(e);
    });
    return { writers: writers, readers: readers, vectors: vectors };
  }

  function pipeSpec(name) {
    return S.pipes.find(function (p) { return p.name === name; }) ||
      { name: name, ring: 2, capacity_tokens: 0, device: false, world_object: null };
  }

  function chipName(chip) {
    return chip.pipe ? chip.pipe.name : chip.object;
  }

  function buildLayout() {
    nodes = {};
    chips = {};
    objects = {};

    var bands = [];
    S.descriptors.forEach(function (d) {
      if (bands.indexOf(d.priority) < 0) bands.push(d.priority);
    });
    bands.sort(function (a, b) { return a - b; });

    var widest = 1;
    bands.forEach(function (p) {
      var n = S.descriptors.filter(function (d) { return d.priority === p; }).length;
      if (n > widest) widest = n;
    });

    var gutterX = PAD + AXIS;
    var left = gutterX + CHIP_W + GUTTER;
    var y = PAD + 18;
    var rows = [];
    bands.forEach(function (priority) {
      var members = S.descriptors.filter(function (d) { return d.priority === priority; });
      rows.push({ priority: priority, y: y, height: NODE_H });
      members.forEach(function (d, i) {
        nodes[d.name] = {
          descriptor: d,
          x: left + i * (NODE_W + GAP_X),
          y: y,
          priority: priority
        };
      });
      y += NODE_H + GAP_Y;
    });

    var bandWidth = widest * NODE_W + (widest - 1) * GAP_X;
    var rightX = left + bandWidth + GUTTER;

    /* Gutter chips. A pipe nothing declares a writer for is a source; one nothing
     * reads is a sink; one backed by a world object is an actuator and always sits
     * on the right, next to the object it changes. */
    var lefts = [], rights = [];
    S.pipes.forEach(function (p) {
      var role = roles(p.name);
      var side = null;
      if (p.world_object) side = "right";
      else if (!role.writers.length) side = "left";
      else if (!role.readers.length) side = "right";
      if (!side) return;
      var anchors = (side === "left" ? role.readers.concat(role.vectors.map(function (v) {
        return v.descriptor;
      })) : role.writers);
      var ys = anchors.map(function (n) { return nodes[n] ? nodes[n].y : PAD; });
      var mid = ys.length
        ? ys.reduce(function (a, b) { return a + b; }, 0) / ys.length
        : PAD;
      var chip = {
        pipe: p,
        x: side === "left" ? gutterX : rightX,
        y: mid + (NODE_H - CHIP_H) / 2,
        side: side
      };
      chips[p.name] = chip;
      if (p.world_object) objects[p.world_object] = chip;
      (side === "left" ? lefts : rights).push(chip);
    });

    /* An object a job maps but nothing actuates has no chip yet, and still has to be
     * somewhere for its status-region edge to come from. It is the world, not a pipe,
     * so it gets a chip carrying only the object. */
    S.edges.forEach(function (edge) {
      if (edge.kind !== "maps" || objects[edge.object]) return;
      var node = nodes[edge.descriptor];
      var chip = {
        pipe: null,
        object: edge.object,
        x: rightX,
        y: (node ? node.y : PAD) + (NODE_H - CHIP_H) / 2,
        side: "right"
      };
      objects[edge.object] = chip;
      rights.push(chip);
    });

    /* Pack each gutter downward so chips never overlap. Deterministic: sorted by the
     * y they wanted, then pushed apart in that order. */
    [lefts, rights].forEach(function (column) {
      column.sort(function (a, b) {
        return a.y - b.y || byKey(chipName(a), chipName(b));
      });
      var floorY = PAD;
      column.forEach(function (chip) {
        chip.y = Math.max(chip.y, floorY);
        floorY = chip.y + CHIP_H + 14;
      });
    });

    var bottom = Math.max(
      y,
      lefts.length ? lefts[lefts.length - 1].y + CHIP_H : 0,
      rights.length ? rights[rights.length - 1].y + CHIP_H : 0
    );
    /* Status regions run in a lane of their own beneath everything. Routed straight,
     * they cross whatever nodes stand between the world column and the job that maps
     * it -- and in a case where two jobs map each other's counter, that is always the
     * other job. Their own lane also says something true: a status region is not
     * pipe traffic, and it should not share the lanes that are. */
    var mapsLane = bottom + 72;
    var mapped = S.edges.some(function (e) { return e.kind === "maps"; });
    return {
      rows: rows,
      width: rightX + CHIP_W + PAD,
      height: (mapped ? mapsLane + 16 : bottom) + PAD,
      left: left,
      gutterX: gutterX,
      mapsLane: mapsLane
    };
  }

  function anchors(node) {
    return {
      right: [node.x + NODE_W, node.y + NODE_H / 2],
      left: [node.x, node.y + NODE_H / 2],
      top: [node.x + NODE_W / 2, node.y],
      bottom: [node.x + NODE_W / 2, node.y + NODE_H]
    };
  }

  /* Control points lean the way the edge travels, so a right-to-left edge -- a status
   * region feeding a job from the world column -- reads as a reverse S rather than
   * looping out past both ends. */
  function curve(from, to, bow) {
    var dx = (Math.abs(to[0] - from[0]) * 0.5 + 20) * (to[0] >= from[0] ? 1 : -1);
    return "M " + from[0] + " " + from[1] +
      " C " + (from[0] + dx) + " " + (from[1] + (bow || 0)) +
      " " + (to[0] - dx) + " " + (to[1] + (bow || 0)) +
      " " + to[0] + " " + to[1];
  }

  function vcurve(from, to) {
    var dy = Math.abs(to[1] - from[1]) * 0.5 + 16;
    return "M " + from[0] + " " + from[1] +
      " C " + from[0] + " " + (from[1] + dy) +
      " " + to[0] + " " + (to[1] - dy) +
      " " + to[0] + " " + to[1];
  }

  /* Route between two nodes. Forward edges run at mid-height; a return edge dips
   * below the band so a pair handing a baton back and forth reads as two lanes
   * rather than one overdrawn line. */
  function nodePath(from, to, back) {
    var a = anchors(from), b = anchors(to);
    if (from.y === to.y) {
      if (from.x < to.x && !back) return curve(a.right, b.left, 0);
      return "M " + a.bottom[0] + " " + a.bottom[1] +
        " C " + a.bottom[0] + " " + (a.bottom[1] + 44) +
        " " + b.bottom[0] + " " + (b.bottom[1] + 44) +
        " " + b.bottom[0] + " " + b.bottom[1];
    }
    return from.y < to.y ? vcurve(a.bottom, b.top) : vcurve(a.top, b.bottom);
  }

  function midpoint(path, host) {
    var probe = svg("path", { d: path });
    probe.setAttribute("visibility", "hidden");
    host.appendChild(probe);
    var point;
    try {
      point = probe.getPointAtLength(probe.getTotalLength() / 2);
    } catch (err) {
      point = { x: 0, y: 0 };  /* jsdom and friends have no path geometry */
    }
    host.removeChild(probe);
    return point;
  }

  function drawGraph() {
    var box = buildLayout();
    var root = $("svg");
    clear(root);
    root.setAttribute("viewBox", "0 0 " + box.width + " " + box.height);
    root.setAttribute("width", box.width);
    root.setAttribute("height", box.height);

    /* One arrowhead per ring, because a marker does not inherit the referencing
     * path's stroke. Without them a right-to-left edge is ambiguous, and a status
     * region runs right to left -- so the diagram would show the connection while
     * leaving which way the information moves to guesswork. */
    var defs = svg("defs", {});
    [0, 1, 2, 3].forEach(function (ring) {
      var marker = svg("marker", {
        id: "arrow-ring" + ring,
        class: "arrow ring" + ring,
        viewBox: "0 0 10 10",
        refX: 9, refY: 5,
        markerWidth: 6, markerHeight: 6,
        orient: "auto-start-reverse"
      });
      marker.appendChild(svg("path", { d: "M 0 1 L 10 5 L 0 9 z" }));
      defs.appendChild(marker);
    });
    root.appendChild(defs);

    /* Three layers, and the order is the readability rule: boxes cover the edges
     * that pass behind them, and labels cover everything. An edge disappearing under
     * a box still reads as an edge; a label half-hidden by one reads as neither. */
    var edgeLayer = svg("g", {});
    var nodeLayer = svg("g", {});
    var labelLayer = svg("g", {});
    root.appendChild(edgeLayer);
    root.appendChild(nodeLayer);
    root.appendChild(labelLayer);

    box.rows.forEach(function (row) {
      var ruleY = row.y - GAP_Y / 2;
      edgeLayer.appendChild(svg("line", {
        class: "band-rule", x1: box.gutterX, y1: ruleY, x2: box.width - PAD, y2: ruleY
      }));
      var label = svg("text", {
        class: "band-label", x: PAD, y: row.y + row.height / 2, "dominant-baseline": "middle"
      });
      label.textContent = "prio " + row.priority;
      labelLayer.appendChild(label);
    });

    S.pipes.forEach(function (spec) {
      var role = roles(spec.name);
      var chip = chips[spec.name];
      var live = pipeState(spec.name);
      var depth = live ? live.depth : 0;
      var blocker = frame && (frame.jobs || []).some(function (j) {
        return j.blocked_on === spec.name;
      });

      function paint(path, extra) {
        var classes = ["edge", "ring" + spec.ring];
        if (extra) classes.push(extra);
        if (depth > 0) classes.push("busy"); else classes.push("idle");
        if (blocker) classes.push("blocking");
        if (wroteSince(spec.name)) classes.push("flash");
        var node = svg("path", {
          class: classes.join(" "),
          d: path,
          "marker-end": "url(#arrow-ring" + spec.ring + ")"
        });
        node.appendChild(svg("title", {})).textContent =
          spec.name + " -- ring " + spec.ring + ", " + depth + "/" +
          spec.capacity_tokens + " tokens";
        edgeLayer.appendChild(node);
        var point = midpoint(path, edgeLayer);
        var text = svg("text", {
          class: "edge-label" + (depth > 0 ? " busy" : ""),
          x: point.x, y: point.y - 4, "text-anchor": "middle"
        });
        text.textContent = spec.name + (depth > 0 ? " (" + depth + ")" : "");
        labelLayer.appendChild(text);
      }

      if (chip && chip.side === "left") {
        if (show.pipes) {
          role.readers.forEach(function (name) {
            if (!nodes[name]) return;
            paint(curve([chip.x + CHIP_W, chip.y + CHIP_H / 2], anchors(nodes[name]).left, 0));
          });
        }
        // A vector binding is not a pipe binding, so it toggles on its own: the same
        // device pipe can be a handler's interrupt source and another job's stdin.
        if (show.vectors) {
          role.vectors.forEach(function (edge) {
            if (!nodes[edge.descriptor]) return;
            paint(
              curve(
                [chip.x + CHIP_W, chip.y + CHIP_H / 2],
                anchors(nodes[edge.descriptor]).left,
                0
              ),
              "interrupt"
            );
          });
        }
      } else if (chip) {
        if (!show.pipes) return;
        role.writers.forEach(function (name) {
          if (!nodes[name]) return;
          paint(curve(anchors(nodes[name]).right, [chip.x, chip.y + CHIP_H / 2], 0));
        });
      } else {
        if (!show.pipes) return;
        role.writers.forEach(function (w) {
          role.readers.forEach(function (r) {
            if (w === r || !nodes[w] || !nodes[r]) return;
            paint(nodePath(nodes[w], nodes[r], nodes[w].x > nodes[r].x));
          });
        });
      }
    });

    /* Status regions: the world reaching back into a job without the job reading.
     * The kernel rewrites the region in place when the object changes, so the edge
     * is ring 0 -- kernel-injected content -- and not a pipe at all. It is the only
     * information path here that no pipe binding describes, and for a case whose
     * behaviours are built on "trust the status line" it is the important one. */
    S.edges.forEach(function (edge) {
      if (edge.kind !== "maps" || !show.maps) return;
      var node = nodes[edge.descriptor], chip = objects[edge.object];
      if (!node || !chip) return;
      var world = worldNow(frame), before = worldNow(previous);
      var moved = previous && world[edge.object] !== before[edge.object];
      var from = [chip.x + CHIP_W / 2, chip.y + CHIP_H];
      var to = [node.x + NODE_W * 0.72, node.y + NODE_H];
      var path = "M " + from[0] + " " + from[1] +
        " C " + from[0] + " " + box.mapsLane +
        " " + to[0] + " " + box.mapsLane +
        " " + to[0] + " " + to[1];
      var drawn = svg("path", {
        class: "edge ring0 maps" + (moved ? " flash" : ""),
        d: path,
        "marker-end": "url(#arrow-ring0)"
      });
      drawn.appendChild(svg("title", {})).textContent =
        edge.object + " -- " + (edge.region === "status" ? "status region" : "mapped") +
        ", " + edge.mode + ", rewritten in place by the kernel";
      edgeLayer.appendChild(drawn);
      var point = midpoint(path, edgeLayer);
      var text = svg("text", {
        class: "edge-label maps" + (moved ? " busy" : ""),
        x: point.x, y: point.y - 4, "text-anchor": "middle"
      });
      text.textContent = (edge.region === "status" ? "STATUS " : "map ") + edge.object;
      labelLayer.appendChild(text);
    });

    var drawnChips = {};
    Object.keys(chips).forEach(function (name) { drawnChips[chipName(chips[name])] = chips[name]; });
    Object.keys(objects).forEach(function (obj) { drawnChips[chipName(objects[obj])] = objects[obj]; });
    Object.keys(drawnChips).sort().forEach(function (key) {
      drawChip(nodeLayer, drawnChips[key]);
    });
    S.descriptors.forEach(function (d) { drawNode(nodeLayer, nodes[d.name]); });
  }

  function drawChip(host, chip) {
    var spec = chip.pipe;
    // An object nothing actuates still has a chip, so that a status region has
    // somewhere to come from. It is the world rather than a pipe, hence ring 0.
    var object = spec ? spec.world_object : chip.object;
    var value = object ? worldNow(frame)[object] : null;
    var changed = object && previous && value !== worldNow(previous)[object];
    var group = svg("g", {
      class: "chip ring" + (spec ? spec.ring : 0) + (changed ? " changed" : ""),
      transform: "translate(" + chip.x + "," + chip.y + ")"
    });
    group.appendChild(svg("rect", { width: CHIP_W, height: CHIP_H }));
    var title = svg("text", { x: 8, y: 14 });
    title.textContent = spec ? spec.name : object;
    group.appendChild(title);
    var sub = svg("text", { class: "sub", x: 8, y: 26 });
    if (object) {
      sub.textContent = (spec ? object + " = " : "= ") +
        (value === undefined || value === null ? "?" : value);
    } else {
      sub.textContent = spec.device ? "device" : "ring " + spec.ring;
    }
    group.appendChild(sub);
    host.appendChild(group);
  }

  function drawNode(host, node) {
    if (!node) return;
    var d = node.descriptor;
    var job = leadJob(d.name);
    var instances = jobsOf(d.name).length;
    var group = svg("g", {
      class: "node " + (job ? job.state : "absent"),
      transform: "translate(" + node.x + "," + node.y + ")"
    });
    group.appendChild(svg("rect", { class: "body", width: NODE_W, height: NODE_H }));
    group.appendChild(svg("circle", { class: "dot", cx: 15, cy: 18 }));

    var name = svg("text", { class: "name", x: 27, y: 22 });
    name.textContent = d.name + (instances > 1 ? " x" + instances : "");
    group.appendChild(name);

    var meta = svg("text", { class: "meta", x: 15, y: 39 });
    var priority = job && job.priority !== job.base_priority
      ? d.priority + " -> " + job.priority
      : String(d.priority);
    meta.textContent = "prio " + priority + (job ? "  " + job.state : "  not spawned");
    group.appendChild(meta);

    var detail = svg("text", { class: "meta", x: 15, y: 53 });
    detail.textContent = job
      ? ("integ " + job.integrity + "  " + job.tokens + " tok" +
         (job.blocked_on ? "  on " + job.blocked_on : ""))
      : (d.budget.tokens ? "budget " + d.budget.tokens + " tok" : "");
    group.appendChild(detail);

    /* Flags get their own line rather than sharing the name's. A descriptor name is
     * as long as its author made it, so anything beside it eventually collides. */
    var flags = [];
    if (d.pinned) flags.push("PINNED");
    if (!d.preemptible) flags.push("MASKED");
    if (d.utterances.length) flags.push("SPOKEN-TO");
    if (d.requires.tooling.length || d.requires.locomotion) flags.push("EMBODIED");
    if (flags.length) {
      var flag = svg("text", { class: "flag", x: 15, y: 68 });
      flag.textContent = flags.join(" · ");
      group.appendChild(flag);
    }

    var tip = svg("title", {});
    tip.textContent = [
      d.name,
      "priority " + d.priority + (d.pinned ? ", pinned" : ""),
      "reads: " + (d.reads.join(", ") || "-"),
      "writes: " + (d.writes.join(", ") || "-"),
      "source: " + d.source,
      "",
      "click to read " + d.source.split("/").pop()
    ].join("\n");
    group.appendChild(tip);

    /* Keyboard-reachable as well as clickable: the diagram is the page's primary
     * content, and a box that only a mouse can open is a box half the readers cannot. */
    group.setAttribute("tabindex", "0");
    group.setAttribute("role", "button");
    group.onclick = function () { openSource(d); };
    group.onkeydown = function (event) {
      if (event.key === "Enter" || event.key === " ") { openSource(d); event.preventDefault(); }
    };
    host.appendChild(group);
  }

  /* --- the descriptor sheet -------------------------------------------------- */

  /* What a box in the diagram cannot show: the prompt the behaviour actually runs.
   * The frontmatter arrives already parsed, so the declared contract is listed from
   * the same fields the diagram lays out with, and the body is printed verbatim. */
  function openSource(d) {
    $("modal-name").textContent = d.name;
    $("modal-source").textContent = d.source;

    var declared = $("modal-declared");
    clear(declared);
    function fact(key, value) {
      if (!value) return;
      var row = el("div", "row");
      row.appendChild(el("span", "k", key));
      row.appendChild(el("span", "v", value));
      declared.appendChild(row);
    }
    var flags = [];
    if (d.pinned) flags.push("pinned");
    if (!d.preemptible) flags.push("masked");
    fact("priority", String(d.priority) + (flags.length ? "  (" + flags.join(", ") + ")" : ""));
    fact("ring", String(d.ring));
    fact("budget", d.budget.tokens ? d.budget.tokens + " tokens" : "");
    fact("context", d.context.window ? d.context.window + " tokens, " + d.context.eviction : "");
    Object.keys(d.pipes).forEach(function (alias) { fact(alias, d.pipes[alias]); });
    fact("reads", d.reads.join(", "));
    fact("writes", d.writes.join(", "));

    $("modal-body").textContent = d.body || "(this descriptor has no body)";
    $("modal").hidden = false;
    $("modal-close").focus();
  }

  function closeSource() { $("modal").hidden = true; }

  /* --- side panes ---------------------------------------------------------- */

  function drawJobs() {
    var host = $("jobs");
    clear(host);
    if (!frame || !(frame.jobs || []).length) {
      host.appendChild(el("p", "empty", "No jobs yet."));
      return;
    }
    var table = el("table");
    var head = el("tr");
    /* The narrow columns are marked by class rather than by position, so reordering
     * the table cannot silently narrow whichever column happens to be third. */
    [["job"], ["state"], ["prio", "tight"], ["int", "tight"], ["tok"], ["on"]]
      .forEach(function (column) {
        head.appendChild(el("th", column[1] || null, column[0]));
      });
    table.appendChild(el("thead")).appendChild(head);
    var body = el("tbody");
    frame.jobs.forEach(function (j) {
      var tr = el("tr", j.state === "running" ? "running" : (j.state === "faulted" ? "faulted" : ""));
      tr.appendChild(el("td", null, j.job + " " + j.descriptor));
      var state = el("td");
      state.appendChild(el("span", "state " + j.state, j.state));
      tr.appendChild(state);
      tr.appendChild(el("td", "num tight",
        j.priority === j.base_priority ? j.priority : j.base_priority + "→" + j.priority));
      tr.appendChild(el("td", "num tight", j.integrity));
      tr.appendChild(el("td", "num", j.tokens));
      tr.appendChild(el("td", null, j.blocked_on || (j.waiting_for || "—")));
      body.appendChild(tr);
    });
    table.appendChild(body);
    host.appendChild(table);
  }

  function drawStack() {
    var host = $("stack");
    clear(host);
    var stack = frame ? (frame.stack || []) : [];
    if (!stack.length) {
      host.appendChild(el("p", "empty", "Nothing suspended."));
      return;
    }
    stack.slice().reverse().forEach(function (id, i) {
      var job = (frame.jobs || []).find(function (j) { return j.job === id; });
      host.appendChild(el("div", "stack-frame",
        (i === 0 ? "top  " : "     ") + id + "  " + (job ? job.descriptor : "?")));
    });
  }

  function drawPipes() {
    var host = $("pipes");
    clear(host);
    var rows = el("div", "rows");
    S.pipes.forEach(function (spec) {
      var live = pipeState(spec.name);
      var depth = live ? live.depth : 0;
      var row = el("div", "row" + (wroteSince(spec.name) ? " changed" : ""));
      row.appendChild(el("span", "k", spec.name));
      row.appendChild(el("span", "v", depth + "/" + spec.capacity_tokens));
      rows.appendChild(row);
      var bar = el("div", "bar" + (depth >= spec.capacity_tokens ? " full" : ""));
      var fill = el("span");
      fill.style.width = spec.capacity_tokens
        ? Math.min(100, (100 * depth) / spec.capacity_tokens) + "%"
        : "0%";
      bar.appendChild(fill);
      rows.appendChild(bar);
      /* What is actually sitting in the buffer. `contents` is bounded by
       * PIPE_PREVIEW in the fold, so a deeper pipe says how much it is not
       * showing rather than pretending the preview is the whole buffer. */
      var contents = live ? live.contents || [] : [];
      if (contents.length) {
        var toks = el("div", "toks");
        contents.forEach(function (text) { toks.appendChild(el("code", "tok", text)); });
        if (depth > contents.length) {
          toks.appendChild(el("span", "more", "+" + (depth - contents.length) + " more"));
        }
        rows.appendChild(toks);
      }
    });
    host.appendChild(rows);
  }

  /* Every token that has moved at or before this frame, in journal order.
   *
   * `F.tokens` is a flat log sorted by frame, so "up to here" is a prefix -- found
   * by walking from the end, which is cheap because only the tail is drawn. No
   * arithmetic on the journal happens here; the log arrives already tagged with the
   * frame each entry lands on. */
  function drawTokens() {
    var host = $("tokens");
    clear(host);
    var log = F ? (F.tokens || []) : [];
    if (!log.length) {
      host.appendChild(el("p", "empty",
        F && F.count ? "No tokens moved in this run." : "No journal loaded."));
      return;
    }

    var kind = tokenFilter ? tokenFilter.slice(0, tokenFilter.indexOf(":")) : "";
    var want = tokenFilter ? tokenFilter.slice(tokenFilter.indexOf(":") + 1) : "";
    /* The prefix boundary: the first entry past this frame. Binary search because
     * the log is sorted by frame and playing a long run re-renders this constantly. */
    var lo = 0, hi = log.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (log[mid][0] <= at) lo = mid + 1; else hi = mid;
    }
    var end = lo;
    var rows = [], total = 0;
    for (var i = end - 1; i >= 0; i--) {
      var entry = log[i];
      if (kind === "job" && String(entry[2]) !== want) continue;
      if (kind === "pipe" && entry[3] !== want) continue;
      total++;
      if (rows.length < STREAM_ROWS) rows.push(entry);
    }
    rows.reverse();
    if (!rows.length) {
      host.appendChild(el("p", "empty", "Nothing here yet at this point in the run."));
      return;
    }

    if (total > rows.length) {
      host.appendChild(el("p", "empty",
        "showing the last " + rows.length + " of " + total + " movements"));
    }
    var list = el("div", "stream");
    rows.forEach(function (entry) {
      var move = MOVES[entry[1]];
      var line = el("div", "move " + move.cls + (entry[0] === at ? " now" : ""));
      line.appendChild(el("span", "glyph", move.glyph));
      /* A device adapter's write has no job, which is a fact worth showing rather
       * than a blank: it is how the world reaches the kernel. */
      var who = entry[2] === null ? "world" : "job " + entry[2];
      line.appendChild(el("span", "who", entry[3] ? who + " " + entry[3] : who));
      var toks = el("span", "toks");
      entry[4].forEach(function (text) { toks.appendChild(el("code", "tok", text)); });
      line.appendChild(toks);
      list.appendChild(line);
    });
    host.appendChild(list);
    /* Pinned to the newest, because a stream is read at its head. */
    host.scrollTop = host.scrollHeight;
  }

  /* The filter's options are the run's jobs and the case's pipes, rebuilt when the
   * set of jobs changes -- jobs appear as the run goes on. */
  function tokenFilterOptions() {
    var select = $("token-filter");
    var options = [{ value: "", label: "everything" }];
    (frame ? frame.jobs || [] : []).forEach(function (j) {
      options.push({ value: "job:" + j.job, label: "job " + j.job + " " + j.descriptor });
    });
    S.pipes.forEach(function (p) {
      options.push({ value: "pipe:" + p.name, label: p.name });
    });

    /* Rebuilt only when the choices actually change. A select rebuilt on every frame
     * closes itself the moment it is opened, which makes it unusable during playback. */
    var signature = options.map(function (o) { return o.value; }).join("|");
    if (select.dataset.built === signature) return;
    select.dataset.built = signature;
    clear(select);
    options.forEach(function (o) {
      var option = el("option", null, o.label);
      option.value = o.value;
      select.appendChild(option);
    });

    select.value = tokenFilter;
    /* Scrubbing back past a job's spawn removes its option. Falling back to
     * everything beats leaving the pane filtered to something the box no longer says. */
    if (select.value !== tokenFilter) { tokenFilter = ""; select.value = ""; }
  }

  function drawWorld() {
    var host = $("world");
    clear(host);
    var world = worldNow(frame);
    var before = worldNow(previous);
    var keys = Object.keys(world).sort();
    if (!keys.length) {
      host.appendChild(el("p", "empty", "No world state declared."));
      return;
    }
    var rows = el("div", "rows");
    keys.forEach(function (key) {
      var was = previous ? before[key] : world[key];
      var row = el("div", "row" + (was !== world[key] ? " changed" : ""));
      row.appendChild(el("span", "k", key));
      row.appendChild(el("span", "v", world[key]));
      rows.appendChild(row);
    });
    host.appendChild(rows);
  }

  function drawFaults() {
    var host = $("faults");
    clear(host);
    var faults = frame ? (frame.faults || []) : [];
    if (!faults.length) {
      host.appendChild(el("p", "empty", "No faults."));
      return;
    }
    faults.forEach(function (f) {
      var node = el("div", "fault");
      node.appendChild(el("span", "kind", f.kind + (f.job ? "  job " + f.job : "")));
      node.appendChild(el("span", "detail", f.detail));
      host.appendChild(node);
    });
  }

  function drawLint() {
    var host = $("lint");
    clear(host);
    if (!S.lint.length) {
      host.appendChild(el("p", "empty", "Clean."));
      return;
    }
    S.lint.forEach(function (f) {
      var node = el("div", "finding " + f.severity);
      node.appendChild(el("span", "rule", f.rule + (f.descriptor ? " [" + f.descriptor + "]" : "")));
      node.appendChild(el("span", "detail", f.detail));
      host.appendChild(node);
    });
  }

  /* --- the timeline -------------------------------------------------------- */

  var LANE_H = 15, LANE_GAP = 3, LABEL_W = 118, MARK_H = 15;

  function drawTimeline() {
    var root = $("timeline");
    clear(root);
    if (!F || !F.count) return;
    var lanes = F.lanes;
    var height = MARK_H + lanes.length * (LANE_H + LANE_GAP) + 4;
    var width = Math.max(360, root.clientWidth || 900);
    var plot = width - LABEL_W - 8;
    root.setAttribute("viewBox", "0 0 " + width + " " + height);
    root.setAttribute("height", height);

    var x = function (index) { return LABEL_W + (plot * index) / F.count; };

    F.marks.forEach(function (mark) {
      root.appendChild(svg("line", {
        class: "mark-tick", x1: x(mark[0]), y1: 2, x2: x(mark[0]), y2: height
      }));
      var label = svg("text", { class: "mark-label", x: x(mark[0]) + 3, y: 10 });
      label.textContent = mark[1];
      root.appendChild(label);
    });

    lanes.forEach(function (lane, row) {
      var y = MARK_H + row * (LANE_H + LANE_GAP);
      var label = svg("text", { class: "lane-label", x: 0, y: y + 11 });
      label.textContent = lane.job + " " + lane.descriptor;
      root.appendChild(label);
      root.appendChild(svg("rect", {
        class: "lane-bg", x: LABEL_W, y: y, width: plot, height: LANE_H, rx: 2
      }));
      lane.runs.forEach(function (run) {
        var rect = svg("rect", {
          class: "seg " + run[2],
          x: x(run[0]),
          y: y,
          width: Math.max(1, x(run[1]) - x(run[0])),
          height: LANE_H,
          rx: 2
        });
        rect.appendChild(svg("title", {})).textContent =
          lane.descriptor + " " + run[2] + " for " + (run[1] - run[0]) + " events";
        root.appendChild(rect);
      });
    });

    root.appendChild(svg("line", {
      class: "cursor", x1: x(at), y1: 0, x2: x(at), y2: height
    }));

    root.onclick = function (event) {
      var bounds = root.getBoundingClientRect();
      var scale = width / bounds.width;
      var px = (event.clientX - bounds.left) * scale;
      seek(Math.round(((px - LABEL_W) / plot) * F.count));
    };
  }

  /* --- transport ----------------------------------------------------------- */

  function seek(index) {
    if (!F || !F.count) return;
    at = Math.max(0, Math.min(index, F.count - 1));
    previous = at > 0 ? frameAt(at - 1) : null;
    frame = frameAt(at);
    render();
  }

  function step(by) { seek(at + by); }

  function nearestMark(direction) {
    if (!F || !F.marks.length) return at;
    var best = at;
    if (direction > 0) {
      best = F.count - 1;
      F.marks.forEach(function (m) { if (m[0] > at && m[0] < best) best = m[0]; });
    } else {
      best = 0;
      F.marks.forEach(function (m) { if (m[0] < at && m[0] > best) best = m[0]; });
    }
    return best;
  }

  /* Frames within one tick share a virtual instant; F.ticks holds each tick's first
   * frame. Backward from mid-tick lands on the start of the current tick first,
   * which is how a media player treats "previous" and feels right for a boundary. */
  function nearestTick(direction) {
    if (!F || !F.ticks || !F.ticks.length) return at;
    var i;
    if (direction > 0) {
      for (i = 0; i < F.ticks.length; i++) {
        if (F.ticks[i] > at) return F.ticks[i];
      }
      return F.count - 1;
    }
    for (i = F.ticks.length - 1; i >= 0; i--) {
      if (F.ticks[i] < at) return F.ticks[i];
    }
    return 0;
  }

  function play(on) {
    playing = on;
    $("play").setAttribute("aria-pressed", String(on));
    $("play").innerHTML = on ? "&#9646;&#9646;" : "&#9654;";
    if (timer) { clearInterval(timer); timer = null; }
    if (on) {
      timer = setInterval(function () {
        if (at >= F.count - 1) { play(false); return; }
        step(1);
      }, 90);
    }
  }

  /* --- rendering ----------------------------------------------------------- */

  function render() {
    drawGraph();
    drawJobs();
    drawStack();
    drawPipes();
    tokenFilterOptions();
    drawTokens();
    drawWorld();
    drawFaults();

    if (!F || !F.count) return;
    $("headline").textContent = frame.headline + "  ·  " +
      frame.counters.forward_passes + " forward passes  ·  " +
      "tick " + frame.tick + "  ·  " +
      nsToText(frame.clock.virtual_ns);
    $("slider").value = String(at);
    $("at").textContent = at + " / " + (F.count - 1);
    var ticker = $("ticker");
    clear(ticker);
    ticker.appendChild(el("b", null, "#" + frame.seq + "  "));
    ticker.appendChild(document.createTextNode(frame.last_event || "—"));
    drawTimeline();
  }

  /* What the case declares, counted. The three counts that name an edge kind are
   * also that kind's switch, because they are the same fact asked two ways -- how
   * many pipes are there, and am I looking at them. A kind the case has none of stays
   * in the row, disabled: "vectors 0" says this tree has no interrupts, where a
   * missing badge would leave that to be inferred from an absence. */
  function badges() {
    var host = $("badges");
    clear(host);

    function add(label, value, cls) {
      var node = el("span", "badge " + (cls || ""));
      node.appendChild(document.createTextNode(label + " "));
      node.appendChild(el("b", null, value));
      host.appendChild(node);
    }

    function addToggle(kind, key, value) {
      var node = el("button", "badge");
      node.type = "button";
      node.id = "toggle-" + kind;
      node.disabled = value === 0;
      node.title = "show or hide " + kind + " in the diagram (" + key + ")";
      node.appendChild(document.createTextNode(kind + " "));
      node.appendChild(el("b", null, value));
      node.appendChild(el("kbd", null, key));
      node.onclick = function () { toggle(kind); };
      node.setAttribute("aria-pressed", String(show[kind] && !node.disabled));
      host.appendChild(node);
    }

    add("descriptors", S.descriptors.length);
    addToggle("pipes", "p", S.pipes.length);
    addToggle("vectors", "v", S.vectors.length);
    addToggle("maps", "m", S.edges.filter(function (e) { return e.kind === "maps"; }).length);

    var errors = S.lint.filter(function (f) { return f.severity === "error"; }).length;
    var warnings = S.lint.filter(function (f) { return f.severity === "warning"; }).length;
    if (errors) add("lint errors", errors, "error");
    if (warnings) add("lint warnings", warnings, "warning");
    if (F && F.count) add("events", F.count);
  }

  function toggle(kind) {
    var button = $("toggle-" + kind);
    if (button.disabled) return;
    show[kind] = !show[kind];
    button.setAttribute("aria-pressed", String(show[kind]));
    saveToggles();
    drawGraph();  // the only pane the toggles touch
  }

  /* The lines, and what distinguishes them.
   *
   * Two axes share one stroke, which is exactly why this needs saying: the *kind* of
   * relationship is the dash pattern, and the pipe's ring is the colour. The samples
   * carry the same classes the diagram paints with -- a legend that redrew the styles
   * itself would be a second source of truth about how an edge looks, and would go on
   * looking right after the diagram changed. */
  var EDGE_KEY = [
    ["pipe", "edge ring2",
     "a declared binding: a job reads or writes this pipe"],
    ["interrupt", "edge ring2 interrupt",
     "a vector binding: a write dispatches the handler, which preempts what is running"],
    ["status map", "edge ring0 maps",
     "the kernel rewrites a mapped object into the job's context -- the job learns the " +
     "new value without performing a read"],
    ["carrying tokens", "edge ring2 busy",
     "tokens are sitting in this pipe at this point in the run"],
    ["blocking a reader", "edge ring2 blocking",
     "a job is parked on this pipe with nothing to read"]
  ];

  function edgeKey() {
    var host = $("edge-key");
    clear(host);
    host.appendChild(el("b", "key-label", "edges:"));

    EDGE_KEY.forEach(function (entry) {
      var item = el("i", null);
      item.title = entry[2];
      var sample = svg("svg", { class: "sample", width: 30, height: 10 });
      /* A straight sample rather than a curve: the shape carries no meaning in the
       * diagram, only the stroke does. */
      sample.appendChild(svg("path", { class: entry[1], d: "M 1 5 L 29 5" }));
      item.appendChild(sample);
      item.appendChild(document.createTextNode(entry[0]));
      host.appendChild(item);
    });

    /* Colour is the pipe's ring, and only the rings this case actually declares are
     * worth a swatch: a key listing four rings for a tree that uses one teaches the
     * reader to look for a distinction that is not there. Ring 0 is included whenever
     * anything is mapped, because a status region is kernel-injected and carries it. */
    var rings = {};
    S.pipes.forEach(function (p) { rings[p.ring] = true; });
    if (S.edges.some(function (e) { return e.kind === "maps"; })) rings[0] = true;
    Object.keys(rings).sort().forEach(function (ring) {
      var item = el("i", null);
      item.title = "colour marks the ring content on this edge carries";
      var swatch = el("span", "swatch");
      swatch.style.background = "var(--ring" + ring + ")";
      item.appendChild(swatch);
      item.appendChild(document.createTextNode("ring " + ring));
      host.appendChild(item);
    });
  }

  function legend() {
    var host = $("legend");
    clear(host);
    host.appendChild(el("b", "key-label", "jobs:"));
    STATES.forEach(function (state) {
      var item = el("i");
      var swatch = el("span", "swatch");
      swatch.style.background = "var(--" + state.replace("_", "-") + ")";
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(state));
      host.appendChild(item);
    });
  }

  /* --- boot ---------------------------------------------------------------- */

  function start(data) {
    payload = data;
    S = payload.structure;
    F = payload.frames || null;
    document.title = "ZEOS debugger — " + S.case;
    $("case").textContent = S.case;
    loadToggles();  // before badges(), which renders the switches in their state
    installDivider();
    badges();
    legend();
    edgeKey();
    drawLint();

    if (F && F.count) {
      $("foot").hidden = false;
      $("nojournal").hidden = true;
      $("slider").max = String(F.count - 1);
      var marks = $("marks");
      F.marks.forEach(function (mark) {
        var option = el("option", null, "#" + mark[0] + "  " + mark[1]);
        option.value = String(mark[0]);
        marks.appendChild(option);
      });
      marks.onchange = function () {
        if (marks.value !== "") { seek(parseInt(marks.value, 10)); marks.value = ""; }
      };
      $("slider").oninput = function () { seek(parseInt($("slider").value, 10)); };
      $("token-filter").onchange = function () {
        tokenFilter = $("token-filter").value;
        drawTokens();
      };
      $("first").onclick = function () { seek(0); };
      $("last").onclick = function () { seek(F.count - 1); };
      $("prev").onclick = function () { step(-1); };
      $("next").onclick = function () { step(1); };
      $("prev-mark").onclick = function () { seek(nearestMark(-1)); };
      $("next-mark").onclick = function () { seek(nearestMark(1)); };
      $("prev-tick").onclick = function () { seek(nearestTick(-1)); };
      $("next-tick").onclick = function () { seek(nearestTick(1)); };
      $("play").onclick = function () { play(!playing); };
      seek(0);
    } else {
      $("nojournal").hidden = false;
      $("headline").textContent = S.descriptors.length + " behaviours, " +
        S.pipes.length + " pipes, " + S.vectors.length + " vectors";
      render();
    }
  }

  $("modal-close").onclick = closeSource;
  /* The backdrop closes; the sheet itself must not, or every click inside it shuts
   * the thing the click was aimed at. */
  $("modal").onclick = function (event) { if (event.target === $("modal")) closeSource(); };

  document.addEventListener("keydown", function (event) {
    /* While the sheet is open it owns the keyboard. Otherwise Escape would close it
     * *and* space would toggle playback behind it, which is two surprises at once. */
    if (!$("modal").hidden) {
      if (event.key === "Escape") { closeSource(); event.preventDefault(); }
      return;
    }
    /* A focused node or divider handles its own keys; the transport must not also
     * see them, or one arrow press both resizes the panel and steps the journal. */
    if (event.target && event.target.closest
        && event.target.closest("g.node, #divider")) return;

    // The toggles work with or without a journal: a tree you have not run yet is
    // exactly when you want to read one relationship at a time.
    if (event.key === "p" || event.key === "v" || event.key === "m") {
      toggle({ p: "pipes", v: "vectors", m: "maps" }[event.key]);
      event.preventDefault();
      return;
    }
    if (!F || !F.count) return;
    var jump = event.shiftKey ? 10 : 1;
    switch (event.key) {
      case "ArrowLeft": step(-jump); break;
      case "ArrowRight": step(jump); break;
      case "Home": seek(0); break;
      case "End": seek(F.count - 1); break;
      case ",": seek(nearestMark(-1)); break;
      case ".": seek(nearestMark(1)); break;
      case "[": seek(nearestTick(-1)); break;
      case "]": seek(nearestTick(1)); break;
      case " ": play(!playing); break;
      default: return;
    }
    event.preventDefault();
  });

  $("theme").onclick = function () {
    var now = document.documentElement.getAttribute("data-theme");
    var next = now === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    if (F && F.count) render();
  };

  window.addEventListener("resize", function () { if (F && F.count) drawTimeline(); });

  if (window.__ZEOS__) {
    start(window.__ZEOS__);
  } else {
    /* Served rather than exported: the shell arrives with no data and asks for it,
     * so editing static/ and reloading shows the edit. */
    fetch("api/payload")
      .then(function (r) { return r.json(); })
      .then(start)
      .catch(function (err) {
        $("headline").textContent = "could not load payload: " + err;
      });
  }
})();
