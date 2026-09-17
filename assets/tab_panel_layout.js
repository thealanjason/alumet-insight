/** Viewport sizing and tab-panel clientside helpers for the main analysis tabs. */

var TAB_PANEL_IDS = {
    "time-series-tab": "time-series-content",
    "process-specific-tab": "process-specific-content",
    "comparative-tab": "comparative-content",
};

var OVERLAY_HIDDEN = {display: "none"};
var OVERLAY_VISIBLE = {display: "flex"};

function panelClass(isActive) {
    return isActive ? "tab-panel-scroll tab-panel-active" : "tab-panel-scroll tab-panel-idle";
}

function visibleTabPanel() {
    return document.querySelector(".tab-panel-active");
}

function panelIsPlaceholder(el) {
    if (!el || !el.children || !el.children.length) {
        return true;
    }
    for (var i = 0; i < el.children.length; i++) {
        var cls = String(el.children[i].className || "");
        if (cls.indexOf("empty-process-specific-content") !== -1) {
            return true;
        }
        if (cls.indexOf("empty-comparative-content") !== -1) {
            return true;
        }
        if (cls.indexOf("empty-time-series-content") !== -1) {
            return true;
        }
    }
    return false;
}

function overlayStyleForTab(tab) {
    if (tab === "time-series-tab") {
        return OVERLAY_HIDDEN;
    }
    var panel = document.getElementById(TAB_PANEL_IDS[tab]);
    return panelIsPlaceholder(panel) ? OVERLAY_VISIBLE : OVERLAY_HIDDEN;
}

function resizePanelGraphs(panel) {
    if (!panel || !window.Plotly || !window.Plotly.Plots) {
        return;
    }
    var nodes = panel.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < nodes.length; i++) {
        var gd = nodes[i];
        if (!gd || !gd.offsetWidth || !gd.offsetHeight) {
            continue;
        }
        try {
            window.Plotly.Plots.resize(gd);
        } catch (err) {
            /* Hidden or not-yet-laid-out graphs can throw during prefetch. */
        }
    }
}

function syncTabPanelHeight(activeTab) {
    var area = document.getElementById("tab-content-area");
    var panel = TAB_PANEL_IDS[activeTab]
        ? document.getElementById(TAB_PANEL_IDS[activeTab])
        : visibleTabPanel();

    if (area) {
        area.classList.add("tab-area-locked");
    }
    resizePanelGraphs(panel);
}

function scheduleTabPanelSync(activeTab) {
    syncTabPanelHeight(activeTab);
    setTimeout(function () { syncTabPanelHeight(activeTab); }, 60);
    setTimeout(function () { syncTabPanelHeight(activeTab); }, 250);
}

window.syncTabPanelHeight = syncTabPanelHeight;
window.scheduleTabPanelSync = scheduleTabPanelSync;

function requestTabPrefetch() {
    if (!window.dash_clientside || !window.dash_clientside.set_props) {
        return;
    }
    window.dash_clientside.set_props("tab-prefetch-store", {data: Date.now()});
}

function bindTabHoverPrefetch() {
    var tabs = document.getElementById("results-tabs");
    if (!tabs || window._tabPrefetchBound) {
        return;
    }
    window._tabPrefetchBound = true;
    tabs.addEventListener("pointerenter", function (event) {
        var tabEl = event.target.closest("#results-tabs .tab");
        if (!tabEl || tabEl.classList.contains("tab--selected")) {
            return;
        }
        if (window._tabPrefetchTimer) {
            clearTimeout(window._tabPrefetchTimer);
            window._tabPrefetchTimer = null;
        }
        requestTabPrefetch();
    }, true);
}

window.bindTabHoverPrefetch = bindTabHoverPrefetch;

window.addEventListener("resize", function () {
    syncTabPanelHeight();
});

if (window.ResizeObserver) {
    window.addEventListener("load", function () {
        bindTabHoverPrefetch();
        var area = document.getElementById("tab-content-area");
        if (!area) {
            return;
        }
        new ResizeObserver(function () {
            syncTabPanelHeight();
        }).observe(area);
    });
} else {
    window.addEventListener("load", bindTabHoverPrefetch);
}

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.tab_panel = Object.assign({}, window.dash_clientside.tab_panel, {
    toggleTabPanels: function (tab) {
        if (window.scheduleTabPanelSync) {
            window.scheduleTabPanelSync(tab);
        }
        return [
            panelClass(tab === "time-series-tab"),
            panelClass(tab === "process-specific-tab"),
            panelClass(tab === "comparative-tab"),
            overlayStyleForTab(tab),
        ];
    },
    afterTabBuild: function (_ts, _ps, _comp, tab) {
        if (window.scheduleTabPanelSync) {
            window.scheduleTabPanelSync(tab);
        }
        return [Date.now(), overlayStyleForTab(tab)];
    },
});
