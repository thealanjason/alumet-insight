/** Viewport sizing and tab-panel clientside helpers for the main analysis tabs. */

var TAB_PANEL_IDS = {
    "time-series-tab": "time-series-content",
    "process-specific-tab": "process-specific-content",
    "comparative-tab": "comparative-content",
};

function visibleTabPanel() {
    var ids = Object.keys(TAB_PANEL_IDS);
    for (var i = 0; i < ids.length; i++) {
        var el = document.getElementById(TAB_PANEL_IDS[ids[i]]);
        if (el && el.style.display !== "none") {
            return el;
        }
    }
    return null;
}

function syncTabPanelHeight(activeTab) {
    var area = document.getElementById("tab-content-area");
    var panel = TAB_PANEL_IDS[activeTab]
        ? document.getElementById(TAB_PANEL_IDS[activeTab])
        : visibleTabPanel();

    if (area) {
        area.classList.add("tab-area-locked");
    }
    if (!area || !panel || panel.style.display === "none") {
        return;
    }

    var marginTop = parseFloat(window.getComputedStyle(panel).marginTop) || 0;
    var height = Math.max(0, area.clientHeight - marginTop);
    panel.style.height = height + "px";
    panel.style.maxHeight = height + "px";
}

function scheduleTabPanelSync(activeTab) {
    syncTabPanelHeight(activeTab);
    setTimeout(function () { syncTabPanelHeight(activeTab); }, 60);
    setTimeout(function () { syncTabPanelHeight(activeTab); }, 250);
}

window.syncTabPanelHeight = syncTabPanelHeight;
window.scheduleTabPanelSync = scheduleTabPanelSync;

window.addEventListener("resize", function () {
    syncTabPanelHeight();
});

if (window.ResizeObserver) {
    window.addEventListener("load", function () {
        var area = document.getElementById("tab-content-area");
        if (!area) {
            return;
        }
        new ResizeObserver(function () {
            syncTabPanelHeight();
        }).observe(area);
    });
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    tab_panel: {
        toggleTabPanels: function (tab) {
            var hidden = {display: "none", marginTop: "4px"};
            var visible = {
                display: "flex",
                flexDirection: "column",
                marginTop: "4px",
                minHeight: 0,
                flex: "1 1 0",
                overflow: "hidden",
            };
            var area = document.getElementById("tab-content-area");

            if (area) {
                var height = Math.max(0, area.clientHeight - 4);
                visible.height = height + "px";
                visible.maxHeight = height + "px";
            }

            if (window.scheduleTabPanelSync) {
                window.scheduleTabPanelSync(tab);
            }

            if (tab === "time-series-tab") {
                return [visible, hidden, hidden];
            }
            if (tab === "process-specific-tab") {
                return [hidden, visible, hidden];
            }
            return [hidden, hidden, visible];
        },
        afterTabBuild: function () {
            if (window.scheduleTabPanelSync) {
                window.scheduleTabPanelSync();
            }
            return Date.now();
        },
    },
});
