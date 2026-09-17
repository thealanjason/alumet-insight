/** Restyle existing Plotly figures when the dashboard theme toggles.

Keep in sync with frontend/style.py (figure_color_theme, palettes, pair colors).
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

function _relayoutGraph(gd, light) {
    if (!gd || !window.Plotly || !gd.offsetWidth || !gd.offsetHeight) {
        return;
    }
    var theme = light ? FIGURE_THEME.light : FIGURE_THEME.dark;
    var layout = gd.layout || {};
    var patch = {
        paper_bgcolor: theme.paper,
        plot_bgcolor: theme.plot,
        "font.color": theme.font,
        "title.font.color": theme.font,
        "legend.bgcolor": theme.legend,
        "legend.font.color": theme.legend_font,
    };
    Object.keys(layout).forEach(function (key) {
        if (key.indexOf("xaxis") === 0 || key.indexOf("yaxis") === 0) {
            patch[key + ".gridcolor"] = theme.grid;
            patch[key + ".zerolinecolor"] = theme.grid;
            patch[key + ".tickfont.color"] = theme.font;
            patch[key + ".title.font.color"] = theme.font;
        }
    });
    var shapes = layout.shapes || [];
    for (var i = 0; i < shapes.length; i++) {
        patch["shapes[" + i + "].fillcolor"] = theme.process_fill;
    }
    var map = _colorMap(light);
    var annotations = layout.annotations || [];
    for (var a = 0; a < annotations.length; a++) {
        var ann = annotations[a] || {};
        var annColor = ann.font && ann.font.color;
        patch["annotations[" + a + "].font.color"] = annColor ? _mapColor(annColor, map) : theme.font;
    }
    try {
        Plotly.relayout(gd, patch);
    } catch (err) {
        return;
    }

    var data = gd.data || [];
    data.forEach(function (trace, index) {
        var restyle = {};
        if (trace.line && typeof trace.line.color === "string") {
            restyle["line.color"] = _mapColor(trace.line.color, map);
        }
        if (trace.marker && typeof trace.marker.color === "string") {
            restyle["marker.color"] = _mapColor(trace.marker.color, map);
        }
        if (trace.marker && trace.marker.line && typeof trace.marker.line.color === "string") {
            restyle["marker.line.color"] = _mapColor(trace.marker.line.color, map);
        }
        if (typeof trace.fillcolor === "string") {
            restyle.fillcolor = _mapColor(trace.fillcolor, map);
        }
        if (Object.keys(restyle).length) {
            try {
                Plotly.restyle(gd, restyle, index);
            } catch (err) {
                /* Prefetched or empty traces can reject a restyle. */
            }
        }
    });
}

window.restylePlotlyTheme = function (useLightMode) {
    var light = !!useLightMode;
    var nodes = document.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < nodes.length; i++) {
        _relayoutGraph(nodes[i], light);
    }
};
