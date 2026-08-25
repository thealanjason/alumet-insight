/**
 * Dash dcc.Upload copies filename from file.name (basename only).
 * Folder pick sets file.webkitRelativePath to "folder/file.csv".
 * Rewrite file.name in capture phase so Dash's FileReader sees the folder.
 */
(function () {
    var STORE_ID = "upload-relative-paths";

    function applyRelativeName(file) {
        var rel = file && file.webkitRelativePath;
        if (!rel || rel === file.name) {
            return;
        }
        try {
            Object.defineProperty(file, "name", {
                configurable: true,
                writable: true,
                value: rel,
            });
        } catch (_err) {
            /* File.name is native; store publish is the fallback. */
        }
    }

    function applyToFiles(files) {
        if (!files || !files.length) {
            return;
        }
        Array.prototype.forEach.call(files, applyRelativeName);
        publishStore(files);
    }

    function publishStore(files) {
        var allowed = { ".csv": 1, ".log": 1, ".txt": 1, ".toml": 1 };
        var names = Array.prototype.map.call(files, function (file) {
            return file.webkitRelativePath || file.name || "";
        }).filter(function (name) {
            var lower = name.toLowerCase();
            var dot = lower.lastIndexOf(".");
            return dot !== -1 && allowed[lower.slice(dot)];
        });
        var hasFolder = names.some(function (name) {
            return name.indexOf("/") !== -1 || name.indexOf("\\") !== -1;
        });
        if (!hasFolder) {
            return;
        }
        var tries = 0;
        var timer = setInterval(function () {
            tries += 1;
            if (window.dash_clientside && window.dash_clientside.set_props) {
                window.dash_clientside.set_props(STORE_ID, { data: names });
                clearInterval(timer);
            } else if (tries > 40) {
                clearInterval(timer);
            }
        }, 50);
    }

    function isOurUpload(node) {
        return !!(node && node.closest && node.closest("#directory-upload"));
    }

    document.addEventListener(
        "change",
        function (event) {
            var input = event.target;
            if (!input || input.type !== "file" || !input.files) {
                return;
            }
            if (!isOurUpload(input)) {
                return;
            }
            applyToFiles(input.files);
        },
        true
    );

    document.addEventListener(
        "drop",
        function (event) {
            if (!isOurUpload(event.target)) {
                return;
            }
            var transfer = event.dataTransfer;
            if (transfer && transfer.files) {
                applyToFiles(transfer.files);
            }
        },
        true
    );
})();
