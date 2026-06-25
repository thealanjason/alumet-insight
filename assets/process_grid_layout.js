/** Viewport sizing and tab-panel clientside helpers for the process-specific grid. */

function syncProcessGridHeight(activeTab) {
    var area = document.getElementById("tab-content-area");
    var panel = document.getElementById("process-specific-content");
    var isProcessTab = activeTab === "process-specific-tab"
        || (panel && panel.style.display !== "none");

    if (area) {
        area.classList.toggle("tab-area-locked", isProcessTab);
    }
    if (!area || !panel || panel.style.display === "none") {
        return;
    }

    var height = area.clientHeight;
    panel.style.height = height + "px";
    panel.style.maxHeight = height + "px";
}

function scheduleProcessGridSync(activeTab) {
    syncProcessGridHeight(activeTab);
    setTimeout(function () { syncProcessGridHeight(activeTab); }, 60);
    setTimeout(function () { syncProcessGridHeight(activeTab); }, 250);
}

window.syncProcessGridHeight = syncProcessGridHeight;
window.scheduleProcessGridSync = scheduleProcessGridSync;

window.addEventListener("resize", function () {
    syncProcessGridHeight();
});

if (window.ResizeObserver) {
    window.addEventListener("load", function () {
        var area = document.getElementById("tab-content-area");
        if (!area) {
            return;
        }
        new ResizeObserver(function () {
            syncProcessGridHeight();
        }).observe(area);
    });
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    process_grid: {
        toggleTabPanels: function (tab) {
            var hidden = {display: "none", marginTop: "10px"};
            var visible = {display: "flex", flexDirection: "column", marginTop: "10px", minHeight: 0};
            var processVisible = {
                display: "flex",
                flexDirection: "column",
                marginTop: "4px",
                minHeight: 0,
                flex: "1 1 0",
                overflow: "hidden",
            };
            var area = document.getElementById("tab-content-area");

            if (tab === "process-specific-tab" && area) {
                var height = area.clientHeight;
                processVisible.height = height + "px";
                processVisible.maxHeight = height + "px";
            }

            if (window.scheduleProcessGridSync) {
                window.scheduleProcessGridSync(tab);
            }

            if (tab === "time-series-tab") {
                return [visible, hidden, hidden];
            }
            if (tab === "process-specific-tab") {
                return [hidden, processVisible, hidden];
            }
            return [hidden, hidden, visible];
        },
        afterGridBuild: function (_children) {
            if (window.scheduleProcessGridSync) {
                window.scheduleProcessGridSync("process-specific-tab");
            }
            return Date.now();
        },
    },
});