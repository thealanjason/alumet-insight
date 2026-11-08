/** Restyle existing Plotly figures when the dashboard theme toggles.

Keep in sync with frontend/style.py (figure_color_theme, palettes, pair colors).

SVG colors flip with a stylesheet so a toggle does not redraw every trace.
WebGL canvases cannot be recolored that way; only the visible one is redrawn,
and only after the page has painted.
*/

var FIGURE_THEME = {
    light: {
        paper: "#ffffff",
        plot: "#f7f8fa",
        font: "#1f2937",
        grid: "rgba(31, 41, 55, 0.12)",
        legend: "rgba(255, 255, 255, 0.92)",
        legend_font: "#000000",
        process_fill: "rgba(62, 107, 143, 0.16)",
        derived_title: "#7E5693",
    },
    dark: {
        paper: "#252c3e",
        plot: "#1e2433",
        font: "#ECEFF4",
        grid: "rgba(216, 222, 233, 0.15)",
        legend: "rgba(37, 44, 62, 0.88)",
        legend_font: "#ffffff",
        process_fill: "rgba(136, 192, 208, 0.12)",
        derived_title: "#B48EAD",
    },
};

var PLOT_COLORS_DARK = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#88C0D0", "#EBCB8B", "#D08770", "#B48EAD",
];
var PLOT_COLORS_LIGHT = [
    "#3D4ED8", "#C73E2A", "#0A8F6C", "#7B3FD4", "#D97706",
    "#0E8AAA", "#C43D6E", "#4F7D3B", "#A21CAF", "#B45309",
    "#3E6B8F", "#A36A00", "#B85C38", "#7E5693",
];
var PAIR_DARK = ["#88C0D0", "#FF6B6B", "#FF8C42", "#A3BE8C", "#FFFFFF"];
var PAIR_LIGHT = ["#3E6B8F", "#C73E2A", "#D97706", "#4F7D3B", "#1F2937"];
var DEVICE_DARK = ["#88C0D0", "#EBCB8B", "#A3BE8C", "#A4AEBC"];
var DEVICE_LIGHT = ["#3E6B8F", "#B45309", "#4F7D3B", "#6B7280"];

function _normHex(color) {
    if (typeof color !== "string") {
        return "";
    }
    return color.trim().toUpperCase();
}

function _colorMap(toLight) {
    var from = toLight ? PLOT_COLORS_DARK.concat(PAIR_DARK) : PLOT_COLORS_LIGHT.concat(PAIR_LIGHT);
    var to = toLight ? PLOT_COLORS_LIGHT.concat(PAIR_LIGHT) : PLOT_COLORS_DARK.concat(PAIR_DARK);
    var map = {};
    for (var i = 0; i < from.length; i++) {
        map[_normHex(from[i])] = to[i];
    }
    var srcTheme = toLight ? FIGURE_THEME.dark : FIGURE_THEME.light;
    var dstTheme = toLight ? FIGURE_THEME.light : FIGURE_THEME.dark;
    map[_normHex(srcTheme.derived_title)] = dstTheme.derived_title;
    map[_normHex(srcTheme.font)] = dstTheme.font;
    return map;
}

function _hexToRgb(hex) {
    var h = hex.replace("#", "");
    if (h.length !== 6) {
        return null;
    }
    return [
        parseInt(h.slice(0, 2), 16),
        parseInt(h.slice(2, 4), 16),
        parseInt(h.slice(4, 6), 16),
    ];
}

function _mapColor(color, map) {
    if (typeof color !== "string") {
        return color;
    }
    var hex = _normHex(color);
    if (map[hex]) {
        return map[hex];
    }
    var rgba = color.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([0-9.]+))?\s*\)$/i);
    if (!rgba) {
        return color;
    }
    for (var key in map) {
        if (!Object.prototype.hasOwnProperty.call(map, key) || key.charAt(0) !== "#") {
            continue;
        }
        var rgb = _hexToRgb(key);
        if (rgb && rgb[0] === Number(rgba[1]) && rgb[1] === Number(rgba[2]) && rgb[2] === Number(rgba[3])) {
            var mapped = _hexToRgb(map[key]);
            if (!mapped) {
                return color;
            }
            if (rgba[4] === undefined) {
                return "rgb(" + mapped[0] + ", " + mapped[1] + ", " + mapped[2] + ")";
            }
            return "rgba(" + mapped[0] + ", " + mapped[1] + ", " + mapped[2] + ", " + rgba[4] + ")";
        }
    }
    return color;
}

function _replaceDeviceColors(text, toLight) {
    if (typeof text !== "string" || !text) {
        return text;
    }
    var from = toLight ? DEVICE_DARK : DEVICE_LIGHT;
    var to = toLight ? DEVICE_LIGHT : DEVICE_DARK;
    var out = text;
    for (var i = 0; i < from.length; i++) {
        out = out.replace(new RegExp(from[i], "ig"), to[i]);
    }
    return out;
}

function _plotlyAxisId(axisKey) {
    if (axisKey === "yaxis") {
        return "y";
    }
    var numbered = /^yaxis(\d+)$/.exec(axisKey);
    if (!numbered || numbered[1] === "1") {
        return "y";
    }
    return "y" + numbered[1];
}

function _traceOnAxis(trace, axisKey) {
    var want = _plotlyAxisId(axisKey);
    var got = (trace && trace.yaxis) || "y";
    if (got === "y1") {
        got = "y";
    }
    return got === want;
}

function _traceColorForAxis(data, axisKey) {
    for (var i = 0; i < data.length; i++) {
        var trace = data[i] || {};
        if (!_traceOnAxis(trace, axisKey)) {
            continue;
        }
        if (trace.line && typeof trace.line.color === "string") {
            return trace.line.color;
        }
        if (trace.marker && typeof trace.marker.color === "string") {
            return trace.marker.color;
        }
    }
    return "";
}

function _ensureFont(container, key) {
    if (!container[key] || typeof container[key] !== "object") {
        container[key] = {};
    }
    return container[key];
}

function _applyThemeToLayout(layout, light, map) {
    if (!layout) {
        return;
    }
    var theme = light ? FIGURE_THEME.light : FIGURE_THEME.dark;
    layout.paper_bgcolor = theme.paper;
    layout.plot_bgcolor = theme.plot;
    _ensureFont(layout, "font").color = theme.font;
    if (layout.title && typeof layout.title === "object") {
        _ensureFont(layout.title, "font").color = theme.font;
    }
    if (layout.legend) {
        layout.legend.bgcolor = theme.legend;
        _ensureFont(layout.legend, "font").color = theme.legend_font;
    }
    Object.keys(layout).forEach(function (key) {
        if (key.indexOf("xaxis") !== 0 && key.indexOf("yaxis") !== 0) {
            return;
        }
        var axis = layout[key];
        if (!axis || typeof axis !== "object") {
            return;
        }
        axis.gridcolor = theme.grid;
        axis.zerolinecolor = theme.grid;
        _ensureFont(axis, "tickfont").color = theme.font;
        if (axis.title && typeof axis.title === "object") {
            _ensureFont(axis.title, "font").color = theme.font;
        }
    });
    var shapes = layout.shapes || [];
    for (var i = 0; i < shapes.length; i++) {
        if (shapes[i]) {
            shapes[i].fillcolor = theme.process_fill;
        }
    }
    var meta = layout.meta || {};
    if (meta.dual_y_line_ticks) {
        var titleColors = meta.axis_title_colors || {};
        ["yaxis", "yaxis2"].forEach(function (axisKey) {
            var axis = layout[axisKey];
            if (!axis) {
                return;
            }
            var lineColor = _traceColorForAxis(layout._traceData || [], axisKey);
            if (lineColor) {
                _ensureFont(axis, "tickfont").color = _mapColor(lineColor, map);
            }
            if (titleColors[axisKey] && axis.title && typeof axis.title === "object") {
                _ensureFont(axis.title, "font").color = _replaceDeviceColors(titleColors[axisKey], light);
            }
        });
    }
    var annotations = layout.annotations || [];
    for (var a = 0; a < annotations.length; a++) {
        var ann = annotations[a];
        if (!ann) {
            continue;
        }
        var annColor = ann.font && ann.font.color;
        _ensureFont(ann, "font").color = annColor ? _mapColor(annColor, map) : theme.font;
        if (typeof ann.text === "string") {
            ann.text = _replaceDeviceColors(ann.text, light);
        }
    }
}

function _applyThemeToTraces(data, map) {
    if (!data) {
        return;
    }
    for (var i = 0; i < data.length; i++) {
        var trace = data[i];
        if (!trace) {
            continue;
        }
        if (trace.line && typeof trace.line.color === "string") {
            trace.line.color = _mapColor(trace.line.color, map);
        }
        if (trace.marker && typeof trace.marker.color === "string") {
            trace.marker.color = _mapColor(trace.marker.color, map);
        }
        if (trace.marker && trace.marker.line && typeof trace.marker.line.color === "string") {
            trace.marker.line.color = _mapColor(trace.marker.line.color, map);
        }
        if (typeof trace.fillcolor === "string") {
            trace.fillcolor = _mapColor(trace.fillcolor, map);
        }
    }
}

function _syncFigureModel(gd, light, map) {
    var data = gd.data || [];
    if (gd.layout) {
        gd.layout._traceData = data;
        _applyThemeToLayout(gd.layout, light, map);
        delete gd.layout._traceData;
    }
    if (gd._fullLayout) {
        gd._fullLayout._traceData = data;
        _applyThemeToLayout(gd._fullLayout, light, map);
        delete gd._fullLayout._traceData;
    }
    _applyThemeToTraces(data, map);
    _applyThemeToTraces(gd._fullData, map);
}

function _hasGlTraces(gd) {
    var data = gd.data || [];
    for (var i = 0; i < data.length; i++) {
        var kind = data[i] && data[i].type;
        if (kind === "scattergl" || kind === "heatmapgl" || kind === "pointcloud") {
            return true;
        }
    }
    return false;
}

function _plotIsHidden(gd) {
    var panel = gd.closest ? gd.closest(".tab-panel-idle") : null;
    return !!panel;
}

function _setDomTheme(gd, themeName) {
    gd.classList.remove("plotly-dom-light", "plotly-dom-dark");
    gd.classList.add(themeName === "light" ? "plotly-dom-light" : "plotly-dom-dark");
}

function _tagUntaggedPlots(themeName) {
    var nodes = document.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < nodes.length; i++) {
        var gd = nodes[i];
        if (!gd.classList.contains("plotly-dom-light") && !gd.classList.contains("plotly-dom-dark")) {
            _setDomTheme(gd, themeName);
        }
    }
}

function _domMatchesPage(gd) {
    var page = window._plotlyPageTheme === "light" ? "plotly-dom-light" : "plotly-dom-dark";
    return gd.classList.contains(page);
}

function _cssColorValue(color) {
    return String(color).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function _structuralPairs(toLight) {
    var src = toLight ? FIGURE_THEME.dark : FIGURE_THEME.light;
    var dst = toLight ? FIGURE_THEME.light : FIGURE_THEME.dark;
    return [
        {src: src.paper, dst: dst.paper, prop: "fill"},
        {src: src.plot, dst: dst.plot, prop: "fill"},
        {src: src.legend, dst: dst.legend, prop: "fill"},
        {src: src.grid, dst: dst.grid, prop: "stroke"},
        {src: src.process_fill, dst: dst.process_fill, prop: "fill"},
    ];
}

function _tracePairs(toLight) {
    var pairs = [];
    var seen = {};
    function add(src, dst) {
        if (!src || !dst || String(src).toLowerCase() === String(dst).toLowerCase()) {
            return;
        }
        var key = String(src).replace(/\s+/g, "").toLowerCase();
        if (seen[key]) {
            return;
        }
        seen[key] = true;
        pairs.push([src, dst]);
    }
    var map = _colorMap(toLight);
    Object.keys(map).forEach(function (key) {
        add(key, map[key]);
    });
    var fromDev = toLight ? DEVICE_DARK : DEVICE_LIGHT;
    var toDev = toLight ? DEVICE_LIGHT : DEVICE_DARK;
    for (var i = 0; i < fromDev.length; i++) {
        add(fromDev[i], toDev[i]);
    }
    return pairs;
}

function _colorParts(color) {
    if (typeof color !== "string") {
        return null;
    }
    if (color.charAt(0) === "#") {
        var hex = _hexToRgb(color);
        if (!hex) {
            return null;
        }
        return {r: hex[0], g: hex[1], b: hex[2]};
    }
    var match = color.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (!match) {
        return null;
    }
    return {r: Number(match[1]), g: Number(match[2]), b: Number(match[3])};
}

function _rgbTuple(parts) {
    return "rgb(" + parts.r + ", " + parts.g + ", " + parts.b + ")";
}

function _rulesForPairs(scope, pairs) {
    var css = "";
    for (var i = 0; i < pairs.length; i++) {
        var item = pairs[i];
        var src = item.src || item[0];
        var dst = item.dst || item[1];
        var prop = item.prop || "both";
        var parts = _colorParts(src);
        if (!parts) {
            continue;
        }
        var srcRgb = _rgbTuple(parts);
        var dstParts = _colorParts(dst);
        var dstCss = dst.charAt(0) === "#" || !dstParts ? dst : _rgbTuple(dstParts);
        var quoted = _cssColorValue(src);
        if (prop === "fill" || prop === "both") {
            css += scope + ' [style*="fill: ' + srcRgb + '"]{fill:' + dstCss + " !important}";
            css += scope + ' [style*="fill:' + srcRgb + '"]{fill:' + dstCss + " !important}";
            css += scope + ' [style*="background: ' + srcRgb + '"]{background-color:' + dstCss + " !important}";
            css += scope + ' [style*="background:' + srcRgb + '"]{background-color:' + dstCss + " !important}";
            css += scope + ' [fill="' + quoted + '" i]{fill:' + dstCss + " !important}";
        }
        if (prop === "stroke" || prop === "both") {
            css += scope + ' [style*="stroke: ' + srcRgb + '"]{stroke:' + dstCss + " !important}";
            css += scope + ' [style*="stroke:' + srcRgb + '"]{stroke:' + dstCss + " !important}";
            css += scope + ' [stroke="' + quoted + '" i]{stroke:' + dstCss + " !important}";
        }
        if (src.charAt(0) === "#") {
            css += scope + ' [style*="color:' + quoted + '" i]{color:' + dstCss + " !important}";
            css += scope + ' [style*="color: ' + quoted + '" i]{color:' + dstCss + " !important}";
        }
    }
    return css;
}

function _ensureThemeStyles() {
    if (document.getElementById("plotly-theme-css")) {
        return;
    }
    var lightScope = 'html[data-bs-theme="light"] .js-plotly-plot.plotly-dom-dark';
    var darkScope = 'html[data-bs-theme="dark"] .js-plotly-plot.plotly-dom-light';
    var css = "";
    css += _rulesForPairs(lightScope, _tracePairs(true).concat(_structuralPairs(true)));
    css += _rulesForPairs(darkScope, _tracePairs(false).concat(_structuralPairs(false)));
    css += lightScope + " .legend text{fill:" + FIGURE_THEME.light.legend_font + " !important}";
    css += darkScope + " .legend text{fill:" + FIGURE_THEME.dark.legend_font + " !important}";
    var style = document.createElement("style");
    style.id = "plotly-theme-css";
    style.textContent = css;
    document.head.appendChild(style);
}

function _recolorLabels(light) {
    var labeled = document.querySelectorAll(
        ".device-class-chip, .process-grid-selection-caption, .comparative-plot-title, .comparative-legend-label, .device-class-key, .comparative-plot-title-text"
    );
    for (var j = 0; j < labeled.length; j++) {
        var root = labeled[j];
        var styled = [root].concat(Array.prototype.slice.call(root.querySelectorAll("[style]")));
        for (var k = 0; k < styled.length; k++) {
            var node = styled[k];
            if (!node.style || !node.style.color) {
                continue;
            }
            node.style.color = _replaceDeviceColors(node.style.color, light);
        }
    }
}

var _glGeneration = 0;

function _scheduleVisibleGlRedraw() {
    if (!window.Plotly || !window.Plotly.redraw) {
        return;
    }
    var generation = ++_glGeneration;
    var nodes = document.querySelectorAll(".js-plotly-plot");
    var pending = [];
    for (var i = 0; i < nodes.length; i++) {
        var gd = nodes[i];
        if (_hasGlTraces(gd) && !_plotIsHidden(gd) && !_domMatchesPage(gd)) {
            pending.push(gd);
        }
    }
    function step() {
        if (generation !== _glGeneration || !pending.length) {
            return;
        }
        var gd = pending.shift();
        try {
            window.Plotly.redraw(gd);
        } catch (err) {
            /* An empty WebGL view can reject a redraw. */
        }
        if (pending.length) {
            window.requestAnimationFrame(step);
        }
    }
    if (pending.length) {
        window.requestAnimationFrame(step);
    }
}

function _normalizedColor(color) {
    return String(color || "").replace(/\s+/g, "").toLowerCase();
}

function _rgbOf(hex) {
    var rgb = _hexToRgb(hex);
    if (!rgb) {
        return "";
    }
    return "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
}

function _detectDomTheme(gd) {
    var bg = gd.querySelector ? gd.querySelector(".main-svg > .bg") : null;
    var fill = bg ? _normalizedColor(bg.getAttribute("fill")) : "";
    if (fill === _normalizedColor(FIGURE_THEME.dark.paper) || fill === _rgbOf(FIGURE_THEME.dark.paper)) {
        return "dark";
    }
    if (
        fill === _normalizedColor(FIGURE_THEME.light.paper) ||
        fill === _rgbOf(FIGURE_THEME.light.paper) ||
        fill === "white" ||
        fill === "#fff"
    ) {
        return "light";
    }
    return window._plotlyPageTheme || "dark";
}

function _bindThemePlotEvents() {
    if (window._plotlyThemeEventsBound) {
        return;
    }
    window._plotlyThemeEventsBound = true;
    document.addEventListener("plotly_afterplot", function (evt) {
        var gd = evt.target && evt.target.closest
            ? evt.target.closest(".js-plotly-plot")
            : null;
        if (!gd) {
            return;
        }
        _setDomTheme(gd, _detectDomTheme(gd));
    }, true);
}

window.flushPlotlyThemeGl = _scheduleVisibleGlRedraw;

window.restylePlotlyTheme = function (useLightMode) {
    var light = !!useLightMode;
    var next = light ? "light" : "dark";
    var prev = window._plotlyPageTheme;
    window._plotlyPageTheme = next;
    _ensureThemeStyles();
    _bindThemePlotEvents();
    if (prev === undefined) {
        _tagUntaggedPlots(next);
        return;
    }
    if (prev === next) {
        return;
    }
    _tagUntaggedPlots(prev);
    var map = _colorMap(light);
    var nodes = document.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < nodes.length; i++) {
        try {
            _syncFigureModel(nodes[i], light, map);
        } catch (err) {
            /* Skip this graph and keep going. The stylesheet still recolors it. */
        }
    }
    _recolorLabels(light);
    _scheduleVisibleGlRedraw();
};
